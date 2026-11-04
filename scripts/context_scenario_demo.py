"""Full D12-style clock-context scenario pair, with one shared real model run."""
from copy import deepcopy
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
import torch
from src.eval.context_density import ContextDensity
from src.eval.anomaly_state import states_at
from src.eval import demo_gif as visual
from src.integrated.data import observation_graph,stable_seed,verify_lock
from src.integrated.forecast import load_model,infer,medoid_index
from src.integrated.geometry import forecast_intent
from src.integrated.presentation import load_presentation
from src.integrated.scenario_status import scenario_status,latch_intrusions,FACTOR_REASONS
from src.integrated.future_groups import replay_object_groups
from src.integrated.track_style import TrackStyleRenderer
from src.integrated.story import write_json

FACTORS=['location_height_nll','speed_nll','course_nll']
NAMES=['Location / height','Speed','Travel direction']


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def protected(root):
    return {str(p.relative_to(root)):digest(p) for d in ['outputs/integrated_demo','outputs/anomaly_context_time','outputs/context_flip_examples','outputs/context_flip_animation','data/splits']
            for p in (root/d).rglob('*') if p.is_file()}


class ContextScenarioRenderer(TrackStyleRenderer):
    def object_evidence_text(self,obj):
        score=obj['current_state_nll']
        return '—' if score is None else f'{score:.2f}'

    def __init__(self,data,dpi=100):
        super().__init__(data,dpi)
        period='DAYTIME 12:00' if data['clock_label']==0 else 'NIGHTTIME 22:00'
        for text in self.fig.texts:
            value=text.get_text()
            if value.startswith(('DAYTIME HOBBYIST','NIGHTTIME APPROACH')):
                text.set_text(f'{period} LOCAL  ·  SAME TRACKS / CONTEXT CHANGED  ·  SCRIPTED DRONE + REFERENCE AIRCRAFT')
                text.set_fontsize(10)
            elif value=='ZONE ENTRY':text.set_text('SCORE')
            elif value.startswith('Orange: current object anomaly.'):
                text.set_text(f"Context-conditioned anomaly cutoff {data['manifest']['cutoff_raw']:.2f} · endpoint % = forecast sample share · synthetic clock-pair example")
        self.fig.canvas.draw();self.background=self.fig.canvas.copy_from_bbox(self.fig.bbox)


def create_data(fixed,stream,forecasts,model,cutoff,clock):
    observed=stream['observed'];n,t,_=observed.shape
    trace={key:np.full((n,t),np.nan) for key in [*FACTORS,'state_nll']}
    daytrace={key:np.full((n,t),np.nan) for key in FACTORS}
    agents,frames=np.nonzero(stream['valid'][:,6:]&stream['valid'][:,:-6]);frames+=6
    states=states_at(dict(pos=observed,valid=stream['valid']),agents,frames)
    scores=model.score_context(states,clock);days=model.score_context(states,0)
    for key in trace:trace[key][agents,frames]=scores[key]
    for key in daytrace:daytrace[key][agents,frames]=days[key]
    rows=[]
    for frame,time in enumerate(stream['times']):
        origins=[f for f in forecasts if f<=frame];origin=max(origins) if origins else None
        entries=forecast_intent(forecasts[origin]['position'][:,:,frame-origin:],fixed['manifest']['zone']) if origin is not None else [None,None]
        objects=[]
        for agent,name in enumerate(['scripted_drone','aa4e47']):
            value=trace['state_nll'][agent,frame];valid=bool(np.isfinite(value));anomaly=bool(valid and value>cutoff)
            differences=[trace[k][agent,frame]-daytrace[k][agent,frame] for k in FACTORS]
            driver=int(np.argmax(differences)) if valid else 0
            explanation=[dict(factor=NAMES[driver])]
            reason=FACTOR_REASONS[NAMES[driver]] if anomaly else None
            objects.append(dict(id=name,current_state_nll=float(value) if valid else None,
                                current_exceeds_cutoff=anomaly,anomalous=anomaly,anomaly_reason=reason,
                                explanation=explanation,entry=entries[agent],forecast_origin_frame=origin))
        status=scenario_status(objects,observed[:,frame],fixed['manifest']['zone'])
        if frame>=6:
            period='daytime (07–19)' if clock==0 else 'nighttime (19–07)'
            status['reasons'].append(f'States are evaluated against {period} local reference data.')
        rows.append(dict(frame=frame,time_s=float(time),objects=objects,scenario=status))
    rows=latch_intrusions(rows,observed,fixed['manifest']['zone'])
    result=dict(fixed);result['manifest']=deepcopy(fixed['manifest'])
    result['manifest']['scene'].update(context='daytime' if clock==0 else 'nighttime',scripted_noise_sigma_m_per_axis=0)
    result['manifest']['cutoff_raw']=cutoff
    result.update(stream=stream,timeline=rows,forecasts=forecasts,clock_label=clock,groups={},grouped=True,all_anomalies=True,intrusion_memory=True)
    result['object_groups']=replay_object_groups(rows,forecasts,result['manifest']['zone'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='Read existing scene/forecasts, never execute the model')
    parser.add_argument('--output-dir')
    parser.add_argument('--playback-start-frame',type=int,default=18,help='Keep preceding frames as observed history; start visible replay here')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];source_run=root/'outputs/context_scenario_demo'
    out=root/(args.output_dir or ('outputs/context_scenario_demo_history_heading' if args.render_only else 'outputs/context_scenario_demo'))
    if args.render_only and out.resolve()==source_run.resolve():raise SystemExit('Render-only output must preserve the original scene directory')
    out.mkdir(exist_ok=True);verify_lock(root);before=protected(root)
    if args.render_only:
        before.update({str(p.relative_to(root)):digest(p) for p in source_run.rglob('*') if p.is_file()})
    marker=out/'inference_attempt.json'
    config=json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())
    metadata=json.loads((root/'outputs/context_flip_examples/examples.json').read_text())
    cutoff=metadata['cutoff'];density=ContextDensity.load(root/'outputs/anomaly_context_time/state_density.npz')
    if args.render_only:
        previous=json.loads((source_run/'inference_attempt.json').read_text())
        assert previous['status']=='complete' and previous['attempt']==1
        with np.load(source_run/'observations.npz') as a:stream={k:a[k].copy() for k in a.files}
        observed=stream['observed'];times=stream['times'];forecasts={}
        for frame in [18,24,30]:
            with np.load(source_run/'forecasts'/f'f{frame:04d}.npz') as a:
                forecasts[frame]={k:a[k].copy() for k in ['position','heading','medoid']}
        inference_runs=0
    else:
        if marker.exists():raise SystemExit('Shared inference already attempted; do not silently rerun or tune')
        with np.load(root/'outputs/context_flip_examples/synthetic_drone_track.npz') as a:drone=a['position'].copy();times=a['time_s'].copy()
        source=root/'outputs/integrated_demo/D11/D4_anchor/B_nighttime_approach/observations.npz'
        with np.load(source) as a:
            observed=np.stack([drone,a['observed'][1,:len(times)]]);reference=np.stack([drone,a['reference'][1,:len(times)]])
        stream=dict(observed=observed,reference=reference,valid=np.ones(observed.shape[:2],bool),times=times)
        np.savez_compressed(out/'observations.npz',**stream)
        write_json(marker,dict(status='running',attempt=1,device='CPU',checkpoint=config['checkpoint'],seed=20261029))
        torch.set_num_threads(1);network=load_model(root,config['checkpoint']);forecasts={};(out/'forecasts').mkdir(exist_ok=True)
        for frame in [18,24,30]:
            graph=observation_graph(observed,stream['valid'],frame,network.token_processor.meta['pos_scale'])
            seed=stable_seed(20261029,'context_clock_pair',frame)
            pred=infer(network,graph,seed);pred['medoid']=medoid_index(pred['position']);forecasts[frame]=pred
            np.savez_compressed(out/'forecasts'/f'f{frame:04d}.npz',**pred,seed=seed)
            print(json.dumps(dict(forecast_frame=frame,seed=seed,samples=len(pred['position']))),flush=True)
        del network;inference_runs=1
        write_json(marker,dict(status='complete',attempt=1,device='CPU',checkpoint=config['checkpoint'],seed=20261029,origins=[18,24,30],shared_across_clocks=True))
    fixed=load_presentation(root,'B_nighttime_approach',True,True,True)
    cases=[create_data(fixed,stream,forecasts,density,cutoff,c) for c in [0,1]]
    start=args.playback_start_frame
    if not 18<=start<len(times):raise ValueError('Playback requires at least6s of existing past history')
    render_frames=list(range(start,len(times)))
    for data in cases:data['playback_start_frame']=start
    np.testing.assert_array_equal(cases[0]['stream']['observed'],cases[1]['stream']['observed'])
    assert cases[0]['forecasts'] is cases[1]['forecasts']
    assert all(not r['objects'][0]['anomalous'] for r in cases[0]['timeline'][6:])
    assert all(r['objects'][0]['anomalous'] for r in cases[1]['timeline'][6:])
    assert all(not r['objects'][1]['anomalous'] for d in cases for r in d['timeline'][6:])
    if args.render_only:
        for label,data in zip(['daytime','nighttime'],cases):
            assert data['timeline']==json.loads((source_run/label/'timeline.json').read_text())
            assert json.loads(json.dumps(data['object_groups']))==json.loads((source_run/label/'future_groups.json').read_text())
    receipt=dict(stage='full_scene_clock_pair',render_only=args.render_only,shared_inference_attempts=inference_runs,device='CPU',forecast_origins=3,
                 checkpoint=config['checkpoint'],samples=32,top_k=5,temperature=1.,cutoff=cutoff,
                 identical_tracks=True,identical_forecasts=True,unchanged_zone_rules_latch=True,scenes=[])
    receipt.update(playback_start_frame=start,playback_start_s=float(times[start]),history_frames=18,source_frames=len(times),rendered_frames=len(render_frames))
    for label,data in zip(['daytime','nighttime'],cases):
        directory=out/label;directory.mkdir(exist_ok=True);renderer=ContextScenarioRenderer(data)
        palette=visual.fixed_palette(data['map_image'],extra_colours=['#ff9f43','#ffd166','#ff626b','#132434','#294052'])
        frames=[]
        assert np.isfinite(renderer.icon_headings[:,start]).all(),'Initial directional icons require supplied moving history'
        for frame in render_frames:frames.append(renderer.frame(frame).quantize(palette=palette,dither=0))
        write_json(directory/'layout_checks.json',renderer.layout_receipt);renderer.close()
        gif=directory/f'{label}_scenario.gif'
        durations=np.diff(np.rint(np.arange(len(frames)+1)*100/3).astype(int)*10).tolist()
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=1)
        mp4=directory/f'{label}_scenario.mp4'
        subprocess.run([shutil.which('ffmpeg'),'-y','-loglevel','error','-threads','1','-i',str(gif),'-vf','fps=15',
                        '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
        subprocess.run([shutil.which('ffmpeg'),'-v','error','-threads','1','-i',str(mp4),'-f','null','-'],check=True)
        renderer=ContextScenarioRenderer(data,dpi=200)
        for frame in sorted({start,min(start+1,len(times)-1),min(start+3,len(times)-1),len(times)-1}):
            name=f'review_f{frame:04d}.png' if args.render_only else f'review_t{round(frame/3):02d}.png'
            renderer.frame(frame).save(directory/name)
        renderer.close();write_json(directory/'timeline.json',data['timeline']);write_json(directory/'future_groups.json',data['object_groups'])
        counts={s:sum(r['scenario']['status']==s for r in data['timeline'][start:]) for s in sorted({r['scenario']['status'] for r in data['timeline'][start:]})}
        receipt['scenes'].append(dict(scene=label,frames=len(render_frames),source_frames=len(times),statuses=counts,final_drone_score=data['timeline'][-1]['objects'][0]['current_state_nll'],
                                      background_normal=True,mp4_full_decode=True,map_bounds=data['pb'].tolist()))
        print(json.dumps(receipt['scenes'][-1]),flush=True)
    assert before=={name:digest(root/name) for name in before};verify_lock(root);receipt['protected_files_unchanged']=len(before)
    write_json(out/'acceptance.json',receipt)
    (out/'README.md').write_text('''# Full-scene clock-context demo

Two separate full-map scenario animations in D12 style: synthetic drone, existing reference aircraft, unchanged fictional protected zone, model forecasts and right-side scenario/WHY/object cards. Object card shows conditional anomaly score, cutoff12.64. Day/night have exactly the same observed tracks and32 model futures; only clock-conditioned state scores differ. No night bonus or new threat rule.

Saved synthetic10s track is unchanged (height30m,speed25m/s), and the background reuses the first10s of accepted D11 B observations/reference. Foreground is a clean script; background retains its original3m scripted noise. One CPU anchor inference run at6/8/10s,32samples/topk5/temp1,seed20261029; forecasts shared across both clocks. No6s-history forecasts before6s. Current scores need2s history. No training, network, packages or previous demo changes. No fabricated or borrowed drone predictions. The prior local split-screen video remains available separately.

Normal/anomaly refers to aircraft-reference kinematics, not a drone identity or safety classifier. Showcase selected earlier for clock-dependent score flip, not evaluation performance. Daytime07–19/night19–07Chicago are counterfactual conditions. Context is in the anomaly detector, not the neural forecasting model; forecast probabilities are sample shares. Weather unused. Existing entry>=8/32 and observed intrusion latch rules remain unchanged.

MP4/GIF1600x900,real-time3Hz source, MP4 encoded15fps. Review PNGs3200x1800. Full unchanged D12 airport map bounds; recent10s faded history, display-smoothed positions and model-directed heading. Same joint medoid/grouping rules, groups for anomalous objects. The first6s remain available as observed history; visible playback starts at6s with directional icons and forecasts, ending at10s (4s visible motion). No collecting-observation frames are shown. Render source: scripts.context_scenario_demo; inference guard prevents accidental repeated runs.
''')
    if args.render_only:
        with (out/'README.md').open('a') as f:
            f.write('\nHistory-based initialization: playback starts at6s, after the existing6s observed history. First visible frame uses history-derived directional aircraft/drone icons, then circular easing follows issued forecasts. The saved10s source leaves4s visible motion; no synthetic prehistory or future observations are used. Render-only reads saved inputs and forecasts with ZERO new inference. Timelines/groups match original exactly; old videos are preserved.\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
