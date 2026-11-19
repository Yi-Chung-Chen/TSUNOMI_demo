"""Find original mixed scenes with repeatable warnings before first zone entry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from scripts.render_unreal_shortlist import forecast_clip
from scripts.shortlist_unreal_replays import read_jsonl, read_scene
from src.integrated.forecast import load_model
from src.integrated.geometry import distance_to_zone, forecast_intent, inside_zone
from src.integrated.scoring import FrozenStateScorer

HZ = 3
STRIDE = 6
DISTANCE_HORIZON = np.arange(1, 91) / HZ


def screen(dataset, entries, scorer, zone):
    candidates = []
    geometry_count = 0
    for entry in entries:
        clean, observed, valid, ids, drone = read_scene(dataset / entry['path'], entry)
        reference_inside = inside_zone(clean, zone) & valid
        observed_inside = inside_zone(observed, zone) & valid
        entrants = np.flatnonzero(drone & reference_inside.any(axis=1) & observed_inside.any(axis=1))
        for agent in entrants:
            geometry_count += 1
            first_ref = int(np.flatnonzero(reference_inside[agent])[0])
            first_obs = int(np.flatnonzero(observed_inside[agent])[0])
            limit = min(first_ref, first_obs) - 6 * HZ
            latest = limit // STRIDE * STRIDE
            trace = scorer.observation_trace(observed[agent:agent+1], valid[agent:agent+1])['state_nll'][0]
            runs = []
            for frame in range(max(18, latest - 18), latest + 1, STRIDE):
                if not valid[agent, frame-18:frame+1].all() or not trace[frame] > scorer.cutoff:
                    continue
                if reference_inside[agent, frame] or observed_inside[agent, frame]:
                    continue
                if np.any((observed_inside & drone[:, None])[:, :frame+1]):
                    continue
                continuous = valid[:, frame-18:min(valid.shape[1], first_ref+12*HZ+1)].all(axis=1)
                aircraft = np.count_nonzero(continuous & ~drone)
                if not aircraft:
                    continue
                velocity = (observed[agent, frame] - observed[agent, frame-6]) / 2
                cv = observed[agent, frame] + DISTANCE_HORIZON[:, None] * velocity
                gap = float(distance_to_zone(cv, zone).min())
                runs.append(dict(frame=frame, cv_gap_m=gap, continuous_aircraft=int(aircraft)))
            pairs = [(a,b) for a,b in zip(runs,runs[1:]) if b['frame']-a['frame']==STRIDE]
            if not pairs:
                continue
            a,b = min(pairs,key=lambda pair: (max(x['cv_gap_m'] for x in pair), -pair[0]['continuous_aircraft'], pair[0]['frame']))
            # CV prioritizes which examples to verify. The model issues the actual warnings.
            candidates.append(dict(scenario=entry['scenario'], family=entry['family'],
                drone_id=ids[agent], first_ref=first_ref, first_obs=first_obs,
                pair=[a['frame'],b['frame']], eligible=[r['frame'] for r in runs],
                continuous_aircraft=min(a['continuous_aircraft'],b['continuous_aircraft']),
                cv_gap_m=max(a['cv_gap_m'],b['cv_gap_m'])))
    return sorted(candidates,key=lambda c: (c['cv_gap_m'] > 0, -c['continuous_aircraft'], c['cv_gap_m'], c['scenario'])), geometry_count


def verify(candidate):
    torch.set_num_threads(1)
    name = candidate['scenario']
    entry = INDEX[name]
    _, observed, valid, ids, drone = read_scene(DATASET / entry['path'], entry)
    target = ids.index(candidate['drone_id'])
    results = []
    for frame in candidate['frames_to_check']:
        active = np.flatnonzero(valid[:, frame-18:frame+1].all(1))
        if target not in active:
            continue
        forecasts, _ = forecast_clip(MODEL, observed, valid, active, frame, frame, STRIDE, name)
        evidence = forecast_intent(forecasts[18]['position'], ZONE)
        results.append(dict(frame=frame, entry_samples=evidence[list(active).index(target)]['entry_samples']))
    eligible = set(candidate['eligible'])
    valid_pairs = [[a['frame'],b['frame']] for a,b in zip(results,results[1:])
              if a['frame'] in eligible and b['frame'] in eligible and b['frame']-a['frame']==STRIDE
              and min(a['entry_samples'], b['entry_samples']) >= 8]
    return dict(candidate, results=results, qualifying_pairs=valid_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=12)
    args = parser.parse_args()
    global DATASET, INDEX, MODEL, ZONE
    root = Path(__file__).resolve().parents[1]
    DATASET = root.parent/'SYGARD-Dataset/ord_mixed_100_3hz'
    INDEX = {e['scenario']:e for e in read_jsonl(DATASET/'index.jsonl')}
    out = root/'outputs/unreal_replay_shortlist/early_warning'
    out.mkdir(exist_ok=True)
    config = json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())
    ZONE = config['zone']
    scorer = FrozenStateScorer(root/'outputs/anomaly_state_smoothed')
    candidates, entrants = screen(DATASET, INDEX.values(), scorer, ZONE)
    print(f'Screened {len(INDEX)} scenes; {entrants} entries with both reference and noisy observations; {len(candidates)} eligible drone cases',flush=True)
    (out/'screening.json').write_text(json.dumps(dict(scenes=len(INDEX), entrants=entrants,candidates=candidates),indent=2)+'\n')
    selected = [dict(c, frames_to_check=c['pair']) for c in candidates if c['scenario']!='scenario_000073'][:args.limit]
    circle = next((c for c in candidates if c['scenario']=='scenario_000073'), None)
    if circle is None:
        circle = dict(scenario='scenario_000073',drone_id='~drone01', first_ref=326,first_obs=327,eligible=[])
    selected.insert(0,dict(circle, frames_to_check=sorted(set(circle['eligible']) | set(range(294,325,6)))))
    torch.set_num_threads(1)
    MODEL = load_model(root,config['checkpoint'])
    with ProcessPoolExecutor(max_workers=4) as pool:
        with (out/'verified.jsonl').open('w') as f:
            for result in pool.map(verify, selected):
                f.write(json.dumps(result)+'\n')
                f.flush()
                print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
