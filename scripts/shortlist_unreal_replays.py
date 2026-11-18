"""Screen Joshua's native mixed scenarios and prepare timestamped BEV previews."""
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval import demo_gif as visual
from src.integrated.data import observation_graph
from src.integrated.forecast import infer, load_model, medoid_index
from src.integrated.geometry import forecast_intent, inside_zone
from src.integrated.presentation import cached_basemap
from src.integrated.scoring import FrozenStateScorer

FAMILIES = {'circle': 'Circling', 'hover_turn': 'Hover into departure',
            'two_crossings': 'Two crossing drones'}
HZ = 3
HISTORY = 18
CLIP = 90


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def read_scene(folder, entry):
    reference = read_jsonl(folder / 'reference_tracks_3hz.jsonl')
    noisy = read_jsonl(folder / 'noisy_tracks_3hz.jsonl')
    key = lambda row: (row['source_object_id'], row['frame_index'])
    by_key = {key(row): row for row in noisy}
    assert len(by_key) == len(noisy) == len(reference) == entry['reference_rows']
    assert {key(row) for row in reference} == by_key.keys()
    types = {row['source_object_id']: row['vehicle_type'] for row in reference}
    ids = sorted(types)
    index = {name: i for i, name in enumerate(ids)}
    drone = np.array([types[name] == 'drone' for name in ids])
    assert drone.sum() == entry['drone_count']
    shape = (len(ids), entry['duration_s'] * HZ + 1)
    clean = np.zeros((*shape, 3))
    observed = np.zeros_like(clean)
    valid = np.zeros(shape, dtype=bool)
    for row in reference:
        name, frame = key(row)
        matched = by_key[name, frame]
        assert matched['vehicle_type'] == row['vehicle_type']
        assert abs(row['utc_s'] - matched['utc_s']) < 1e-6
        assert abs(row['utc_s'] - entry['start_utc_s'] - frame / HZ) < 1e-6
        agent = index[name]
        clean[agent, frame] = [*row['xyz_enu_m'][:2], row['altitude_m'] - 166]
        observed[agent, frame] = clean[agent, frame] + (
            np.asarray(matched['xyz_enu_m']) - row['xyz_enu_m'])
        valid[agent, frame] = True
    assert valid.sum() == len(reference)
    return clean, observed, valid, ids, drone


def score(scorer, position, valid):
    agents, frames = np.where(valid[:, 6:] & valid[:, :-6])
    frames += 6
    values = np.full(valid.shape, np.nan)
    values[agents, frames] = scorer.states(
        position[agents, frames], position[agents, frames - 6])['state_nll']
    return values


def scene_windows(entry, data, scores, cutoff, phases):
    _, _, valid, ids, drone = data
    for start in range(HISTORY, valid.shape[1] - CLIP, 6):
        end = start + CLIP
        continuous = valid[:, start-HISTORY:end+1].all(axis=1)
        if not continuous[drone].all() or not (continuous & ~drone).any():
            continue
        window = slice(start, end + 1)
        aircraft_scores = scores[~drone, window]
        aircraft_flags = np.mean(aircraft_scores[np.isfinite(aircraft_scores)] > cutoff)
        drone_flags = np.mean(scores[drone, window] > cutoff, axis=1)
        if entry['family'] == 'circle' and not any(
                phase['behavior'] == 'circle' and phase['start_s'] * HZ <= start
                and end <= phase['end_s'] * HZ for phase in phases):
            continue
        transition = any(start + 15 <= phase['start_s'] * HZ <= end - 15
                         for phase in phases if phase['behavior'] == 'turn_and_depart')
        yield dict(scenario=entry['scenario'], family=entry['family'], start=start, end=end,
                   drone_ids=[ids[i] for i in np.flatnonzero(drone)],
                   aircraft_ids=[ids[i] for i in np.flatnonzero(continuous & ~drone)],
                   min_drone_flag_fraction=float(drone_flags.min()),
                   aircraft_flag_fraction=float(aircraft_flags),
                   phase_change=bool(transition))


def preview(root, out, pick, data, scores, cutoff, forecasts, zone):
    clean, observed, valid, ids, drone = data
    start, end = pick['start'], pick['end']
    pos, active = forecasts
    medoid = medoid_index(pos)
    points = np.concatenate([observed[:, start:end+1, :2][valid[:, start:end+1]],
                             pos[medoid, :, :, :2].reshape(-1, 2),
                             np.asarray(zone['polygon_enu_m'])])
    bounds = visual.view_bounds(points, aspect=1.8)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0b1622')
    ax.set_facecolor('#132434')
    try:
        image, zoom, pb = cached_basemap(bounds, root / 'data/basemap_cache')
    except FileNotFoundError:
        zoom = None
        ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect='equal',
               xlabel='East (m)', ylabel='North (m)')
        ax.tick_params(colors='#a5b7c9')
        ax.xaxis.label.set_color('#a5b7c9')
        ax.yaxis.label.set_color('#a5b7c9')
        ax.grid(color='#294052', lw=.5)
    else:
        ax.imshow(image, extent=(pb[0,0], pb[1,0], pb[1,1], pb[0,1]))
        ax.set(xlim=(pb[0,0], pb[1,0]), ylim=(pb[1,1], pb[0,1]))
        ax.axis('off')
        fig.text(.94,.017,'© OpenStreetMap contributors',color='#a5b7c9',ha='right',size=8)

    def project(xy):
        return xy if zoom is None else visual.enu_to_pixel(xy, zoom)
    for agent in range(len(ids)):
        track = observed[agent, start:end+1, :2].copy()
        track[~valid[agent, start:end+1]] = np.nan
        xy = project(track)
        color = '#4cc9f0' if drone[agent] else '#80ed99'
        ax.plot(*xy.T, color=color, lw=2 if drone[agent] else 1, alpha=.85 if drone[agent] else .6)
        if drone[agent]:
            flagged = scores[agent, start:end+1] > cutoff
            ax.scatter(*xy[flagged].T, s=11, color='#ff9f43')
            ax.annotate(ids[agent], xy[0], xytext=(7, 7), textcoords='offset points', color='white')
        elif agent in active:
            ax.scatter(*xy[0], s=13, color=color)
    for i, agent in enumerate(active):
        xy = project(np.vstack([observed[agent, start, :2], pos[medoid, i, :, :2]]))
        ax.plot(*xy.T, '--', color='#ffd166' if drone[agent] else '#80ed99', alpha=.8, lw=1.3)
    corners = project(np.array([*zone['polygon_enu_m'], zone['polygon_enu_m'][0]]))
    ax.plot(*corners.T, ':', color='#c77dff', lw=1.5)
    fig.subplots_adjust(top=.84, bottom=.16)
    fig.text(.06,.93,f"{pick['scenario']}  /  {FAMILIES[pick['family']]}",color='white',size=21)
    fig.text(.06,.885,f"Scenario t={start/3:.0f}–{end/3:.0f}s   |   Observed tracks and forecast issued at t={start/3:.0f}s",color='#a5b7c9')
    fig.text(.06,.04,'Cyan: drone    Green: aircraft    Orange: flagged observation    Dashed: model medoid    Purple: demo zone',color='#dce5f0')
    fig.savefig(out / f"{pick['scenario']}.png", dpi=145)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    dataset = root.parent / 'SYGARD-Dataset/ord_mixed_100_3hz'
    out = root / 'outputs/unreal_replay_shortlist'
    out.mkdir(exist_ok=True)
    entries = read_jsonl(dataset / 'index.jsonl')
    assert len(entries) == len({entry['scenario'] for entry in entries}) == 100
    assert sum(entry['drone_count'] for entry in entries) == 120
    scorer = FrozenStateScorer(root / 'outputs/anomaly_state_smoothed')
    audit, candidates = [], []
    for entry in entries:
        folder = dataset / entry['path']
        data = read_scene(folder, entry)
        clean, observed, valid, _, drone = data
        raw, noisy = (score(scorer, p, valid) for p in (clean, observed))
        receipt = dict(scenario=entry['scenario'], family=entry['family'], start_utc=entry['start_utc'])
        for name, scores in [('reference', raw), ('noisy', noisy)]:
            for role, mask in [('drone', drone), ('aircraft', ~drone)]:
                values = scores[mask]
                receipt[f'{name}_{role}_states'] = int(np.isfinite(values).sum())
                receipt[f'{name}_{role}_flags'] = int((values > scorer.cutoff).sum())
        audit.append(receipt)
        if entry['family'] in FAMILIES:
            metadata = json.loads((folder / 'metadata.json').read_text())
            phases = [phase for track in metadata['track_ids']['reference_tracks']
                      for phase in track.get('phases', [])]
            candidates.extend(scene_windows(entry, data, noisy, scorer.cutoff, phases))
    with (out / 'screening.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(audit)
    print(f'Screened {len(entries)} scenarios, {len(candidates)} eligible candidate windows', flush=True)

    config = json.loads((root / 'configs/integrated_demo/v2_anchor.json').read_text())
    zone = config['zone']
    torch.set_num_threads(1)
    network = load_model(root, config['checkpoint'])
    by_id = {entry['scenario']: entry for entry in entries}
    previous_path = out / 'shortlist.json'
    previous = json.loads(previous_path.read_text())['picks'] if previous_path.exists() else []
    picks = []
    for family in FAMILIES:
        subset = [candidate for candidate in candidates if candidate['family'] == family]
        pick = max(subset, key=lambda c: (c['min_drone_flag_fraction'] >= .9,
                   c['phase_change'] if family == 'hover_turn' else True,
                   c['aircraft_flag_fraction'] <= .05, len(c['aircraft_ids']),
                   c['min_drone_flag_fraction'], -c['aircraft_flag_fraction']))
        entry = by_id[pick['scenario']]
        data = read_scene(dataset / entry['path'], entry)
        clean, observed, valid, ids, drone = data
        start, end = pick['start'], pick['end']
        active = np.flatnonzero(valid[:, start-HISTORY:start+1].all(axis=1))
        graph = observation_graph(observed[active], valid[active], start, network.token_processor.meta['pos_scale'])
        saved = out / f"{pick['scenario']}_rollout.npz"
        if any(all(old[k] == pick[k] for k in ('scenario', 'start', 'end')) for old in previous) and saved.exists():
            with np.load(saved) as arrays:
                np.testing.assert_array_equal(arrays['ids'], np.array(ids)[active])
                assert int(arrays['frame']) == start
                forecast = {key: arrays[key] for key in ('position', 'heading')}
        else:
            forecast = infer(network, graph, 817)
        pos = forecast['position']
        assert pos.shape == (32, len(active), CLIP, 3) and np.isfinite(pos).all()
        evidence = forecast_intent(pos, zone)
        future = clean[active, start+1:end+1]
        complete_air = valid[active, start+1:end+1].all(axis=1) & ~drone[active]
        error = np.linalg.norm(pos[medoid_index(pos), complete_air, :, :2] - future[complete_air, :, :2], axis=-1)
        pick.update(start_utc=entry['start_utc'],
                    aircraft_medoid_horizontal_ADE_m=float(error.mean()),
                    drone_predicted_zone_samples={ids[a]: e['entry_samples'] for a, e in zip(active, evidence) if drone[a]},
                    observed_demo_zone_entry=bool(inside_zone(observed[drone, start:end+1], zone).any()))
        t0 = datetime.fromisoformat(entry['start_utc'])
        pick.update(replay_from_utc=(t0 + timedelta(seconds=(start-HISTORY)/3)).isoformat(),
                    display_from_utc=(t0 + timedelta(seconds=start/3)).isoformat(),
                    display_to_utc=(t0 + timedelta(seconds=end/3)).isoformat())
        preview(root, out, pick, data, score(scorer, observed, valid), scorer.cutoff, (pos, active), zone)
        np.savez_compressed(out / f"{pick['scenario']}_rollout.npz", **forecast, ids=np.array(ids)[active], frame=start)
        picks.append(pick)
        print(json.dumps(pick), flush=True)
    (out / 'shortlist.json').write_text(json.dumps(dict(
        screened=len(entries), candidate_windows=len(candidates), cutoff=scorer.cutoff, picks=picks), indent=2)+'\n')


if __name__ == '__main__':
    main()
