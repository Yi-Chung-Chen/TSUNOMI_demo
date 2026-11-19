"""Replay shortlist clips in the existing D12 interface from saved forecasts."""
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.shortlist_unreal_replays import read_jsonl, read_scene
from src.eval import demo_gif as visual
from src.integrated.geometry import forecast_intent
from src.integrated.future_groups import replay_object_groups
from src.integrated.forecast import medoid_index
from src.integrated.presentation import cached_basemap, load_presentation
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
                text.set_text(f"Forecast issued at t={data['issue_s']:.0f} s · red paths enter demo zone · endpoint % = sample share")
        self.fig.canvas.draw()
        self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)


def assemble(root, dataset, entry, pick, saved, scorer, zone):
    _, observed, valid, ids, _ = read_scene(dataset / entry['path'], entry)
    start, end = pick['start'], pick['end']
    retained = pick['drone_ids'] + pick['aircraft_ids']
    agents = [ids.index(name) for name in retained]
    n_drone = len(pick['drone_ids'])
    labels = [f'Drone {i+1:02d}' for i in range(n_drone)]
    labels += [f'Aircraft {i+1:02d}' for i in range(len(pick['aircraft_ids']))]
    with np.load(saved) as array:
        assert int(array['frame']) == start
        order = [list(array['ids']).index(name) for name in retained]
        raw = array['position']
        forecast = dict(position=raw[:, order], medoid=medoid_index(raw))
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
        origin = issue if issue <= i < issue + forecast['position'].shape[2] else None
        evidence = (forecast_intent(forecast['position'][:, :, i-issue:], zone)
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
                             forecast['position'][..., :2].reshape(-1,2),
                             np.asarray(zone['polygon_enu_m'])))
    try:
        image, zoom, pb = cached_basemap(visual.view_bounds(points, aspect=1.61),
                                         root/'data/basemap_cache')
    except FileNotFoundError:
        # Reuse the existing airport view if a peripheral map tile is missing.
        airport = load_presentation(root, 'B_nighttime_approach', True, True, True)
        image, zoom, pb = (airport[key] for key in ('map_image','zoom','pb'))
        px = visual.enu_to_pixel(positions[..., :2], zoom)
        if not ((px >= pb[0]) & (px <= pb[1])).all():
            raise
    forecasts = {issue: forecast}
    return dict(name=pick['scenario'], issue_s=start/3, playback_start_frame=issue,
                manifest=dict(zone=zone, scene=dict(context='daytime',scripted_noise_sigma_m_per_axis=1)),
                stream=dict(observed=positions), timeline=rows, forecasts=forecasts, grouped=True,
                all_anomalies=True, intrusion_memory=True, object_labels=labels,
                drone_agents=tuple(range(n_drone)), map_image=image, zoom=zoom, pb=pb,
                object_groups=replay_object_groups(rows, forecasts, zone)), dict(zip(labels, retained))


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'outputs/unreal_replay_shortlist'
    out = source/'scenario_videos'
    out.mkdir(exist_ok=True)
    dataset = root.parent/'SYGARD-Dataset/ord_mixed_100_3hz'
    entries = {e['scenario']:e for e in read_jsonl(dataset/'index.jsonl')}
    picks = json.loads((source/'shortlist.json').read_text())['picks']
    zone = json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())['zone']
    scorer = FrozenStateScorer(root/'outputs/anomaly_state_smoothed')
    for pick in picks:
        name = pick['scenario']
        data, labels = assemble(root, dataset, entries[name], pick, source/f'{name}_rollout.npz', scorer, zone)
        renderer = ShortlistRenderer(data)
        first = data['playback_start_frame']
        assert np.isfinite(renderer.icon_headings[:,first]).all()
        video = out/f'{name}.mp4'
        args = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
                '-s','1600x900','-r','3','-i','-','-vf','fps=15','-c:v','libx264',
                '-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            for frame in range(first,len(data['timeline'])):
                image = renderer.frame(frame)
                encoder.stdin.write(image.tobytes())
                if frame == first:
                    image.save(out/f'{name}_opening.png')
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f'Video encoding failed for {name}')
        renderer.close()
        subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
        write_json(out/f'{name}.json',dict(labels=labels, forecast_issued_s=pick['start']/3,
                   display_window_s=[pick['start']/3,pick['end']/3],
                   visible_timeline=data['timeline'][first:]))
        print(f'{name}: encoded and decoded {len(data["timeline"])-first} source frames',flush=True)


if __name__ == '__main__':
    main()
