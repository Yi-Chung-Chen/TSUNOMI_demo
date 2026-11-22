"""Render a complete source scenario in D12, with forecasts every two seconds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.render_unreal_shortlist import ShortlistRenderer, forecast_clip
from scripts.shortlist_unreal_replays import read_jsonl, read_scene
from src.eval import demo_gif as visual
from src.integrated.forecast import load_model
from src.integrated.future_groups import replay_object_groups
from src.integrated.geometry import forecast_intent
from src.integrated.presentation import cached_basemap, load_presentation
from src.integrated.scenario_status import latch_intrusions, object_reason, scenario_status
from src.integrated.scoring import COMPONENTS, FrozenStateScorer
from src.integrated.story import write_json
from src.integrated.threat import explain_state


def init_forecast_worker(model, observed, valid, name, cache):
    global _forecast_inputs
    _forecast_inputs = (model, observed, valid, name, cache)


def forecast_at(frame):
    model, observed, valid, name, cache = _forecast_inputs
    members = np.flatnonzero(valid[:, frame-18:frame+1].all(axis=1))
    if not len(members):
        return frame, None
    path = cache / f'{frame:04d}.npz'
    if path.exists():
        with np.load(path) as data:
            assert np.array_equal(data['members'], members)
            position, medoid = data['position'], int(data['medoid'])
    else:
        result, _ = forecast_clip(model, observed, valid, members, frame, frame, 6, name)
        position, medoid = result[18]['position'], result[18]['medoid']
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, position=position, medoid=medoid, members=members)
        temporary.replace(path)
    full = np.full((32, len(valid), 90, 3), np.nan, dtype=position.dtype)
    full[:, members] = position
    return frame, dict(position=full, medoid=medoid, members=members)


def build_forecasts(model, observed, valid, name, cache, workers=1):
    cache.mkdir(parents=True, exist_ok=True)
    inputs = (model, observed, valid, name, cache)
    frames = range(18, valid.shape[1], 6)
    if workers == 1:
        init_forecast_worker(*inputs)
        return {frame: result for frame,result in map(forecast_at, frames) if result is not None}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'),
                             initializer=init_forecast_worker, initargs=inputs) as executor:
        return {frame: result for frame,result in executor.map(forecast_at, frames) if result is not None}


def build_timeline(observed, valid, ids, labels, forecasts, scorer, medians, zone):
    trace = scorer.observation_trace(observed, valid)
    rows = []
    for frame in range(valid.shape[1]):
        origin = (frame // 6) * 6
        forecast = forecasts.get(origin)
        entries = {}
        if forecast is not None:
            members = forecast['members']
            evidence = forecast_intent(forecast['position'][:, members, frame-origin:], zone)
            entries = dict(zip(members, evidence))
        objects = []
        for i, name in enumerate(ids):
            present = bool(valid[i, frame])
            score = trace['state_nll'][i, frame]
            if present and np.isfinite(score):
                parts = {key: trace[key][i, frame] for key in COMPONENTS[:3]}
                explanation = explain_state(parts, medians)
                obj_score = float(score)
            else:
                explanation, obj_score = [], None
            anomalous = obj_score is not None and obj_score > scorer.cutoff
            obj = dict(id=name, label=labels[i], observed=present, current_state_nll=obj_score,
                       anomalous=anomalous, explanation=explanation, entry=entries.get(i) if present else None,
                       forecast_origin_frame=origin if present and i in entries else None)
            obj['anomaly_reason'] = object_reason(obj) if anomalous else None
            objects.append(obj)
        present = np.flatnonzero(valid[:, frame])
        state = scenario_status([objects[i] for i in present], observed[present, frame], zone)
        rows.append(dict(frame=frame, time_s=frame/3, objects=objects, scenario=state))
    return latch_intrusions(rows, observed, zone)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario')
    parser.add_argument('--workers',type=int,default=1,help='Number of CPU forecast workers')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = root.parent/'SYGARD-Dataset/ord_mixed_100_3hz'
    entry = next(row for row in read_jsonl(dataset/'index.jsonl') if row['scenario'] == args.scenario)
    out = root/'outputs/unreal_replay_shortlist/d12_full_scenes'/args.scenario
    out.mkdir(parents=True, exist_ok=True)
    _, observed, valid, ids, drone = read_scene(dataset/entry['path'], entry)
    labels = ['']*len(ids)
    for mask, name in [(drone,'Drone'), (~drone,'Aircraft')]:
        for number, index in enumerate(np.flatnonzero(mask),1):
            labels[index] = f'{name} {number:02}'
    config = json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())
    torch.set_num_threads(1)
    model = load_model(root,config['checkpoint'])
    scorer_dir = root/'outputs/anomaly_state_smoothed'
    scorer = FrozenStateScorer(scorer_dir)
    with (scorer_dir/'summary.csv').open() as stream:
        medians = {row['score']: float(row['median']) for row in csv.DictReader(stream)
                   if row['group']=='aircraft-airborne (2026-10-04)' and row['score'] in COMPONENTS}
    forecasts = build_forecasts(model, observed, valid, args.scenario, out/'forecasts', args.workers)
    positions = np.where(valid[..., None], observed, np.nan)
    zone = config['zone']
    timeline = build_timeline(positions,valid,ids,labels,forecasts,scorer,medians,zone)
    points = np.concatenate([observed[valid][:,:2],np.asarray(zone['polygon_enu_m'])])
    try:
        image, zoom, pb = cached_basemap(visual.view_bounds(points, aspect=1.61),root/'data/basemap_cache')
    except FileNotFoundError:
        airport = load_presentation(root,'B_nighttime_approach',True,True,True)
        image,zoom,pb = (airport[k] for k in ('map_image','zoom','pb'))
    data = dict(name=args.scenario,manifest=dict(zone=zone,scene=dict(context='daytime',scripted_noise_sigma_m_per_axis=1)),
                stream=dict(observed=positions), timeline=timeline, forecasts=forecasts,
                grouped=True, all_anomalies=True, intrusion_memory=True, object_labels=labels,
                drone_agents=np.flatnonzero(drone), map_image=image, zoom=zoom, pb=pb,
                preserve_forecast_geometry=True, max_why_lines=8, hide_icon_without_heading=True,
                object_groups=replay_object_groups(timeline,forecasts,zone))
    write_json(out/'timeline.json',timeline)
    renderer = ShortlistRenderer(data)
    mp4 = out/f'{args.scenario}.mp4'
    tmp = out/f'{args.scenario}.tmp.mp4'
    command = ['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900',
               '-r','9','-i','-','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(tmp)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for frame in range(len(timeline)):
            encoder.stdin.write(renderer.frame(frame).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg could not finish the replay')
    renderer.close()
    subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(tmp),'-f','null','-'],check=True)
    tmp.replace(mp4)
    write_json(out/'acceptance.json',dict(frames=len(timeline), source_interval_s=[0,entry['duration_s']],
               forecast_origins=len(forecasts), playback_speed=3,
               statuses=Counter(row['scenario']['status'] for row in timeline)))
    print(mp4,flush=True)


if __name__ == '__main__':
    main()
