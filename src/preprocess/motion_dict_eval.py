"""GT-conditioned token reconstruction, not model prediction evaluation.

Port of PRIV's autoregressive contour matching, with 3D distance, per-frame
validity, fixed prediction windows, and exact horizon-specific counts.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path
import shutil
import numpy as np
from src.common.geometry import causal_headings, match_endpoints, decode_tokens
from src.common.units import to_internal, to_metres, FRAME_HZ
from src.preprocess.windows import target_mask


def load_validation(processed_dir, split_dir, split='val', current_frame_idx=18, min_future_frames=30):
    positions, headings, grounds, validities = [], [], [], []
    cache = {}
    with (Path(split_dir) / f'{split}_windows.csv').open() as stream:
        for row in csv.DictReader(stream):
            name, start = row['scenario'], int(row['start_frame'])
            if name not in cache:
                with np.load(Path(processed_dir) / f'{name}.npz', allow_pickle=False) as data:
                    cache[name] = {k: data[k] for k in ['pos', 'heading', 'valid', 'on_ground']}
            data = cache[name]
            sl = slice(start, start + 109)
            v = data['valid'][:, sl]
            take = target_mask(v, current_frame_idx, min_future_frames)
            positions.append(data['pos'][take, sl])
            headings.append(data['heading'][take, sl])
            grounds.append(data['on_ground'][take, sl])
            validities.append(v[take])
    return tuple(map(np.concatenate, [positions, headings, grounds, validities]))


def reconstruct(dictionary, position_m, heading, valid, on_ground, horizon_s,
                autoregressive=True, current_frame_idx=18):
    shift = dictionary['meta']['shift']
    horizon = int(round(horizon_s * FRAME_HZ))
    eligible = valid[:, current_frame_idx:current_frame_idx + horizon + 1].all(-1)
    p = to_internal(position_m[eligible], dictionary['meta']['pos_scale'])
    h = heading[eligible]
    ground = on_ground[eligible]
    if dictionary['meta']['heading_source'] == 'chord':
        h = causal_headings(position_m[eligible], h, valid[eligible], ground, shift)
    previous_p = p[:, current_frame_idx].copy()
    previous_h = h[:, current_frame_idx].copy()
    outputs = []
    for offset in range(0, horizon, shift):
        length = min(shift, horizon - offset)
        start, end = current_frame_idx + offset, current_frame_idx + offset + length
        if not autoregressive:
            previous_p = p[:, start].copy()
            previous_h = h[:, start].copy()
        ids = match_endpoints(p[:, end], h[:, end], previous_p, previous_h, dictionary, endpoint_idx=length)
        decoded, heads = decode_tokens(ids, previous_p, previous_h, dictionary)
        outputs.append(decoded[:, 1:length + 1])
        previous_p = decoded[:, length]
        previous_h = heads[:, length]
    prediction = to_metres(np.concatenate(outputs, axis=1), dictionary['meta']['pos_scale'])
    gt = position_m[eligible, current_frame_idx + 1:current_frame_idx + horizon + 1]
    error = prediction - gt
    return dict(horizontal=np.linalg.norm(error[..., :2], axis=-1), vertical=np.abs(error[..., 2]),
                ground=on_ground[eligible, current_frame_idx], count=int(eligible.sum()))


def evaluate(dictionary, data, autoregressive=True, horizons=(10, 30)):
    p, h, g, v = data
    rows = []
    for horizon in horizons:
        errors = reconstruct(dictionary, p, h, v, g, horizon, autoregressive)
        for group in ['all', 'ground', 'airborne']:
            mask = np.ones(errors['count'], bool) if group == 'all' else errors['ground'] == (group == 'ground')
            row = dict(mode='autoregressive' if autoregressive else 'one_shot', horizon_s=horizon,
                       group=group, n_agents=int(mask.sum()))
            for axis in ['horizontal', 'vertical']:
                e = errors[axis][mask]
                row[f'{axis}_ade_m'] = float(e.mean()) if len(e) else None
                row[f'{axis}_fde_m'] = float(e[:, -1].mean()) if len(e) else None
            rows.append(row)
    return rows


def write_report(records, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sweep.json').write_text(json.dumps(records, indent=2) + '\n')
    rows = [{**{k: record['meta'][k] for k in ['K', 'shift', 'heading_source']}, **row}
            for record in records for row in record['metrics']]
    with (out / 'sweep.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = [(next(r['horizontal_fde_m'] for r in item['metrics']
                        if r['mode'] == 'autoregressive' and r['horizon_s'] == 30 and r['group'] == 'all'), item)
                  for item in records if item['meta']['shift'] == 6]
    best_error, best = min(candidates, key=lambda pair: pair[0])
    target = Path('assets') / f'motion_dict_K{best["meta"]["K"]}_s6.pkl'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best['path'], target)
    selection = dict(path=str(target), source=best['path'], meta=best['meta'], horizontal_fde_30s_m=best_error)
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    Path('docs').mkdir(exist_ok=True)
    lines = ['# Motion dictionary validation', '',
             'This is GT-conditioned token reconstruction error, not prediction error.', '',
             f'Selected shift=6, K={best["meta"]["K"]}, heading={best["meta"]["heading_source"]}.',
             f'30 s autoregressive horizontal FDE: {best_error:.4f} m. Target <10 m: {"met" if best_error < 10 else "MISSED; continue with best dictionary as approved"}.', '',
             'Provisional advisor decisions: tolerance starts at 0.5 internal units and retries ×0.8; token zero exactly stationary; training segments filtered at 200 m/s horizontal / 40 m/s vertical; seeded shared pool capped at 500k per shift. See ../SYGARD_QA_LOG.md.', '',
             'Chord heading is computed from the preceding segment only (causal for history/input), with data fallback on ground or speed ≤10 m/s. All token subframes are retained at 3 Hz. An incomplete final horizon token is matched at the exact requested subframe, with no GT beyond the horizon.', '',
             '| shift | heading | K | mode | horizon s | group | agents | hADE m | hFDE m | vADE m | vFDE m |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        numbers = [f'{r[k]:.4f}' if r[k] is not None else 'NA' for k in ['horizontal_ade_m', 'horizontal_fde_m', 'vertical_ade_m', 'vertical_fde_m']]
        lines.append('| ' + ' | '.join(map(str, [r['shift'], r['heading_source'], r['K'], r['mode'], r['horizon_s'], r['group'], r['n_agents'], *numbers])) + ' |')
    lines += ['', '## Selected dictionary metadata', '', '```json', json.dumps(best['meta'], indent=2), '```', '',
              'Full sweep and per-dictionary filtering metadata: data/motion_dict/eval/sweep.json (ignored generated output).', '']
    Path('docs/motion_dict.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with target.open('rb') as stream:
        d = pickle.load(stream)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for traj, dz in zip(d['traj'], d['dz']):
        xy = to_metres(traj[:, :2], d['meta']['pos_scale'])
        axes[0].plot(xy[:, 0], xy[:, 1], alpha=.15, lw=.5)
        axes[1].plot(np.arange(len(dz)) / FRAME_HZ, to_metres(dz, d['meta']['pos_scale']), alpha=.15, lw=.5)
    axes[0].set(xlabel='Local east / forward (m)', ylabel='Local north / left (m)', title='Token arrow fan', aspect='equal')
    axes[1].set(xlabel='Time (s)', ylabel='Relative height (m)', title='Vertical token tracks')
    axes[0].scatter(0, 0, color='black', s=8)
    end = to_metres(d['traj'][:, -1, :2], d['meta']['pos_scale'])
    head = d['traj'][:, -1, 2]
    axes[0].quiver(end[:, 0], end[:, 1], np.cos(head), np.sin(head), alpha=.35, scale=35)
    fig.tight_layout()
    fig.savefig('docs/motion_dict_fan.png', dpi=150)
    plt.close(fig)
    print(json.dumps(selection, indent=2))
    return selection


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dict-path', type=Path)
    p.add_argument('--dictionary-dir', type=Path, default=Path('data/motion_dict'))
    p.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--split-dir', type=Path, default=Path('data/splits'))
    p.add_argument('--output-dir', type=Path, default=Path('data/motion_dict/eval'))
    p.add_argument('--autoregressive', action='store_true')
    args = p.parse_args()
    data = load_validation(args.processed_dir, args.split_dir)
    paths = [args.dict_path] if args.dict_path else sorted(args.dictionary_dir.glob('*.pkl'))
    records = []
    for path in paths:
        with path.open('rb') as stream:
            dictionary = pickle.load(stream)
        metrics = evaluate(dictionary, data, True)
        if not args.autoregressive:
            metrics += evaluate(dictionary, data, False)
        records.append(dict(path=str(path), meta=dictionary['meta'], metrics=metrics))
        print(f'Evaluated {path}: {metrics[3]}', flush=True)
    if args.dict_path:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'single.json').write_text(json.dumps(records, indent=2) + '\n')
    else:
        write_report(records, args.output_dir)


if __name__ == '__main__':
    main()
