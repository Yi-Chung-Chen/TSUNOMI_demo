"""Read raw SYGARD JSONL without interpolation; save dense scenario arrays."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from src.common.units import altitude_to_height, FRAME_HZ

DEFAULT_ROOT = Path('/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_aircraft_training_800_3hz')


def ingest_scenario(root, entry, output_dir):
    folder = Path(root) / entry['path']
    metadata = json.loads((folder / 'metadata.json').read_text())
    tracks = sorted(metadata['track_ids']['reference_tracks'], key=lambda t: t['source_object_id'])
    ids = [str(t['source_object_id']) for t in tracks]
    lookup = {v: i for i, v in enumerate(ids)}
    n, frames = len(ids), round(entry['duration_s'] * FRAME_HZ) + 1
    valid = np.zeros((n, frames), bool)
    pos = np.zeros((n, frames, 3), np.float32)
    heading = np.zeros((n, frames), np.float32)
    ground = np.zeros((n, frames), bool)
    fallback = np.zeros((n, frames), bool)
    row_count = 0
    with (folder / 'reference_tracks_3hz.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            a, t = lookup[str(row['source_object_id'])], int(row['frame_index'])
            if not 0 <= t < frames or valid[a, t]:
                raise ValueError(f"Invalid or duplicate frame in {entry['scenario']}: {a}, {t}")
            xyz = [*row['xyz_enu_m'][:2], float(altitude_to_height(row['altitude_m']))]
            if not np.isfinite(xyz).all() or not np.isfinite(row['heading_rad']):
                raise ValueError(f"Nonfinite row in {entry['scenario']}")
            valid[a, t] = True
            pos[a, t] = xyz
            heading[a, t] = row['heading_rad']
            ground[a, t] = row['on_ground']
            fallback[a, t] = row['altitude_fallback_used']
            row_count += 1
    if row_count != entry['reference_samples'] or n != entry['aircraft']:
        raise ValueError(f"Index count mismatch: {entry['scenario']}")
    output = Path(output_dir) / (entry['scenario'] + '.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, valid=valid, pos=pos, heading=heading,
                        on_ground=ground, alt_fallback=fallback,
                        agent_id=np.asarray(ids),
                        agent_type=np.asarray([t['adsb_reported_type'] or '' for t in tracks]),
                        start_utc_s=np.asarray(entry['start_utc_s']),
                        date=np.asarray(entry['start_utc'][:10]),
                        bounds_json=np.asarray(json.dumps(metadata['generation']['bounds'])))
    temporary.replace(output)
    return output


def _worker(args):
    return str(ingest_scenario(*args))


def summarize(paths, entries):
    types, dates, counts = Counter(), Counter(), []
    speed, height, xyz, vertical = [], [], [], []
    rows = ground_rows = fallback_rows = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            v, p = d['valid'], d['pos']
            counts.append(len(v))
            types.update(d['agent_type'].tolist())
            dates.update([str(d['date'])])
            rows += int(v.sum())
            ground_rows += int((d['on_ground'] & v).sum())
            fallback_rows += int((d['alt_fallback'] & v).sum())
            pair = v[:, 1:] & v[:, :-1]
            velocity = np.diff(p.astype(np.float64), axis=1)[pair] * FRAME_HZ
            speed.append(np.linalg.norm(velocity[:, :2], axis=-1))
            vertical.append(velocity[:, 2])
            height.append(p[..., 2][v])
            xyz.append(p[v])
    quantiles = [0, 5, 25, 50, 75, 95, 100]
    def q(values):
        return dict(zip(map(str, quantiles), np.percentile(np.concatenate(values), quantiles).tolist()))
    points = np.concatenate(xyz)
    expected = sum(e['reference_samples'] for e in entries)
    if rows != expected:
        raise ValueError(f'Valid samples {rows} != index samples {expected}')
    return dict(scenarios=len(paths), aircraft_tracks=sum(counts), valid_rows=rows,
                index_reference_samples=expected, scenarios_by_date=dict(dates),
                aircraft_per_scenario_quantiles=dict(zip(map(str, quantiles), np.percentile(counts, quantiles).tolist())),
                type_histogram=dict(types.most_common()), ground_fraction=ground_rows / rows,
                altitude_fallback_fraction=fallback_rows / rows,
                horizontal_speed_mps_quantiles=q(speed), height_m_quantiles=q(height),
                vertical_speed_mps_quantiles=q(vertical),
                position_min_m=points.min(0).tolist(), position_max_m=points.max(0).tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    entries = [json.loads(line) for line in (args.root / 'index.jsonl').read_text().splitlines()]
    tasks = [(args.root, e, args.output_dir) for e in entries]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as executor:
            paths = list(executor.map(_worker, tasks))
    else:
        paths = [_worker(t) for t in tasks]
    stats = summarize(paths, entries)
    (args.output_dir / 'stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
