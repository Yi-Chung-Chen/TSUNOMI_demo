"""Replay shortlist clips in the existing D12 interface with refreshed observed-history forecasts."""
import json
from pathlib import Path
import argparse
import subprocess

import numpy as np
import torch

from scripts.shortlist_unreal_replays import read_jsonl, read_scene
from src.eval import demo_gif as visual
from src.integrated.geometry import forecast_intent
from src.integrated.future_groups import replay_object_groups
from src.integrated.forecast import infer, load_model, medoid_index
from src.integrated.data import observation_graph, stable_seed
from src.integrated.presentation import cached_basemap
from src.integrated.scenario_status import object_reason, scenario_status, latch_intrusions
from src.integrated.scoring import COMPONENTS, FrozenStateScorer
from src.integrated.story import write_json
from src.integrated.track_style import TrackStyleRenderer


class ShortlistRenderer(TrackStyleRenderer):
    def __init__(self, data, dpi=100):
        super().__init__(data, dpi)
        for text in self.fig.texts:
            if text.get_text().startswith('DAYTIME HOBBYIST'):
                text.set_text(f"{data['name']}  ·  MIXED ORD REPLAY  ·  1 m SIMULATED TRACKER NOISE")
            elif text.get_text().startswith('Display-smoothed icons'):
                text.set_text('Forecast updates every 2 s · red paths enter demo zone · endpoint % = sample share')
        self.fig.canvas.draw()
        self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)


def forecast_clip(model, observed, valid, agents, start, end, stride, name):
    forecasts, diagnostics = {}, []
    scale = model.token_processor.meta['pos_scale']
    for frame in range(start, end + 1, stride):
        graph = observation_graph(observed[agents], valid[agents], frame, scale)
        begin = frame - 18
        # At issuance, the first two seconds of this history are already known.
        # Use their chord for a new track that lacks pre-window direction.
        have_prior = (valid[agents, begin-6:begin+1].all(axis=1) if begin >= 6
                      else np.zeros(len(agents), dtype=bool))
        chord = observed[agents, begin+6, :2] - observed[agents, begin, :2]
        use_chord = ~have_prior & (np.linalg.norm(chord, axis=1) > 2.)
        graph['agent']['heading'][use_chord, 0] = torch.from_numpy(
            np.arctan2(chord[use_chord, 1], chord[use_chord, 0]))
        with torch.no_grad():
            tokens = model.token_processor.tokenize_agent(graph)
        drift = ((tokens['gt_pos'][:, -1] - tokens['gt_pos_raw'][:, -1]) * scale).numpy()
        seed = stable_seed(817, name, frame)
        paths = infer(model, graph, seed)['position']
        origin = frame - (start - 18)
        forecasts[origin] = dict(position=paths, medoid=medoid_index(paths))
        diagnostics.append(dict(frame=frame, reconstructed_offset_m=drift.tolist(),
                                initial_heading_from_history=use_chord.tolist()))
        print(f'{name}: forecast t={frame/3:g}s', flush=True)
    return forecasts, diagnostics


def assemble(root, dataset, entry, pick, model, scorer, zone, stride):
    _, observed, valid, ids, _ = read_scene(dataset / entry['path'], entry)
    start, end = pick['start'], pick['end']
    retained = pick['drone_ids'] + pick['aircraft_ids']
    agents = [ids.index(name) for name in retained]
    n_drone = len(pick['drone_ids'])
    labels = [f'Drone {i+1:02d}' for i in range(n_drone)]
    labels += [f'Aircraft {i+1:02d}' for i in range(len(pick['aircraft_ids']))]
    forecasts, diagnostics = forecast_clip(
        model, observed, valid, agents, start, end, stride, entry['scenario'])
    # Keep six seconds of observations as history for the first visible frame.
    issue = 18
    first = start - issue
    positions = observed[agents, first:end+1]
    assert valid[agents, first:end+1].all()
    trace = scorer.observation_trace(observed, valid)
    rows = []
    # Carry earlier observed incidents into the clip, even if the drone has
    # exited before the camera recording begins.
    incident_positions = np.where(valid[agents, :end+1, None], observed[agents, :end+1], np.nan)
    for frame in range(end+1):
        i = frame - first
        origin = issue + ((i-issue)//stride)*stride if i >= issue else None
        evidence = (forecast_intent(forecasts[origin]['position'][:, :, i-origin:], zone)
                    if origin is not None else [None]*len(retained))
        objects = []
        for agent, name, label, entry_evidence in zip(agents, retained, labels, evidence):
            score = trace['state_nll'][agent, frame]
            components = [trace[key][agent, frame] for key in COMPONENTS[:3]]
            driver = ['Location / height', 'Speed', 'Travel direction'][int(np.argmax(components))]
            anomalous = bool(np.isfinite(score) and score > scorer.cutoff)
            obj = dict(id=name, label=label, current_state_nll=float(score) if np.isfinite(score) else None,
                       anomalous=anomalous, explanation=[dict(factor=driver)],
                       entry=entry_evidence, forecast_origin_frame=origin)
            obj['anomaly_reason'] = object_reason(obj) if anomalous else None
            objects.append(obj)
        status = scenario_status(objects, incident_positions[:, frame], zone)
        rows.append(dict(frame=i, time_s=frame/3, objects=objects, scenario=status))
    rows = latch_intrusions(rows, incident_positions, zone)[first:]
    points = np.concatenate((positions[..., :2].reshape(-1,2),
                             np.concatenate([prediction['position'][prediction['medoid'], ..., :2].reshape(-1, 2)
                                             for prediction in forecasts.values()]),
                             np.asarray(zone['polygon_enu_m'])))
    try:
        image, zoom, pb = cached_basemap(visual.view_bounds(points, aspect=1.61),
                                         root/'data/basemap_cache')
    except FileNotFoundError:
        # Use the cached shortlist viewport if the longer refreshed forecasts
        # reach outside local map coverage. Keep all displayed observations in view.
        saved = root/'outputs/unreal_replay_shortlist'/f"{pick['scenario']}_rollout.npz"
        with np.load(saved) as array:
            original = array['position']
            initial_paths = original[medoid_index(original),..., :2].reshape(-1,2)
        previous_points = np.concatenate([
            observed[:,start:end+1,:2][valid[:,start:end+1]],
            initial_paths, np.asarray(zone['polygon_enu_m'])])
        image, zoom, pb = cached_basemap(visual.view_bounds(previous_points, aspect=1.8),
                                         root/'data/basemap_cache')
    return dict(name=pick['scenario'], issue_s=start/3, playback_start_frame=issue,
                manifest=dict(zone=zone, scene=dict(context='daytime',scripted_noise_sigma_m_per_axis=1)),
                stream=dict(observed=positions), timeline=rows, forecasts=forecasts, grouped=True,
                all_anomalies=True, intrusion_memory=True, object_labels=labels, preserve_forecast_geometry=True,
                drone_agents=tuple(range(n_drone)), map_image=image, zoom=zoom, pb=pb,
                object_groups=replay_object_groups(rows, forecasts, zone)), dict(zip(labels, retained)), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', action='append')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root/'outputs/unreal_replay_shortlist'
    out = source/'d12_replay'
    out.mkdir(exist_ok=True)
    dataset = root.parent/'SYGARD-Dataset/ord_mixed_100_3hz'
    entries = {e['scenario']:e for e in read_jsonl(dataset/'index.jsonl')}
    picks = json.loads((source/'shortlist.json').read_text())['picks']
    if args.scenario:
        picks = [pick for pick in picks if pick['scenario'] in args.scenario]
        if len(picks) != len(set(args.scenario)):
            parser.error('Unknown shortlist scenario')
    config = json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())
    torch.set_num_threads(1)
    model = load_model(root, config['checkpoint'])
    scorer = FrozenStateScorer(root/'outputs/anomaly_state_smoothed')
    for pick in picks:
        name = pick['scenario']
        data, labels, diagnostics = assemble(root, dataset, entries[name], pick, model, scorer,
                                             config['zone'], config['forecast_stride_frames'])
        renderer = ShortlistRenderer(data)
        first = data['playback_start_frame']
        assert np.isfinite(renderer.icon_headings[:,first]).all()
        video = out/f'{name}.mp4'
        gif = out/f'{name}.gif'
        palette = visual.fixed_palette(data['map_image'], extra_colours=['#ff9f43','#ffd166','#ff626b','#132434','#294052'])
        frames = []
        for frame in range(first,len(data['timeline'])):
            image = renderer.frame(frame)
            frames.append(image.quantize(palette=palette,dither=0))
            if frame == first:
                image.save(out/f'{name}_opening.png')
        renderer.close()
        visual.save_gif(frames,gif,1)
        # Use the D12 presentation pace (3x), without the GIF's end hold.
        subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-i',str(gif),
                        '-vf','fps=9','-frames:v',str(len(frames)),'-c:v','libx264',
                        '-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
        forecasts = data['forecasts']
        np.savez_compressed(out/f'{name}_forecasts.npz',
                            agents=np.array(list(labels.values())),
                            frames=np.array([d['frame'] for d in diagnostics]),
                            position=np.stack([f['position'] for f in forecasts.values()]))
        write_json(out/f'{name}.json',dict(labels=labels, checkpoint=config['checkpoint'],
                   forecast_stride_s=config['forecast_stride_frames']/3, playback_speed=3,
                   display_window_s=[pick['start']/3,pick['end']/3], origins=diagnostics,
                   visible_timeline=data['timeline'][first:]))
        print(f'{name}: encoded and decoded {len(frames)} source frames',flush=True)


if __name__ == '__main__':
    main()
