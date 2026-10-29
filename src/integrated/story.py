"""First-pass story execution and acceptance, with no outcome-driven tuning."""
import csv
import json
from pathlib import Path
import numpy as np

from src.integrated.data import scenario_stream,observation_graph,stable_seed
from src.integrated.forecast import infer,medoid_index,constant_velocity
from src.integrated.geometry import forecast_intent,inside_zone
from src.integrated.scoring import future_anomaly_evidence,COMPONENTS
from src.integrated.threat import AnomalyLatch,threat_level,explain_state


def clean_json(value):
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(clean_json(value),indent=2,allow_nan=False)+'\n')


def evaluate_story(scene,stream,timeline):
    actual=np.flatnonzero(inside_zone(stream['reference'][0],scene['zone']))
    entry=float(stream['times'][actual[0]]) if len(actual) else None
    anomaly=[r['time_s'] for r in timeline if r['objects'][0]['current_exceeds_cutoff']]
    future=[r['time_s'] for r in timeline if r['objects'][0]['future_anomaly_horizon_s'] is not None]
    high=[r['time_s'] for r in timeline if r['objects'][0]['threat']['level']=='High']
    model_high=[r['time_s'] for r in timeline if r['objects'][0]['threat']['level']=='High'
        and r['objects'][0]['threat']['rule_points'].get('zone')==scene['threat']['entry_points']
        and r['objects'][0]['entry'] is not None and r['objects'][0]['entry']['entry_samples']>=8
        and entry is not None and r['time_s']<entry]
    low=[r['time_s'] for r in timeline if r['objects'][0]['anomalous'] and r['objects'][0]['threat']['level']=='Low']
    normal=sum(o['threat']['level']=='Normal traffic' for r in timeline for o in r['objects'][1:])
    return dict(reference_entry_s=entry,first_current_anomaly_s=min(anomaly) if anomaly else None,
        first_future_anomaly_warning_s=min(future) if future else None,
        first_high_s=min(high) if high else None,first_model_based_preentry_high_s=min(model_high) if model_high else None,
        model_warning_lead_s=entry-min(model_high) if model_high else None,
        anomalous_low_frames=len(low),background_normal_frames=normal,
        max_drone_threat_score=max((r['objects'][0]['threat']['score'] or 0) for r in timeline),
        level_counts={level:sum(r['objects'][0]['threat']['level']==level for r in timeline)
            for level in ['Normal traffic','Low','Medium','High','Insufficient data']})


def run_story(root,config,scene,model,scorer,output,resume=False):
    stream=scenario_stream(root,config,scene);directory=output/scene['id'];directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'observations.npz',**stream)
    observed,valid=stream['observed'],stream['valid'];trace=scorer.observation_trace(observed,valid)
    np.savez_compressed(directory/'current_scores.npz',**trace)
    with (root/config['density_dir']/'summary.csv').open() as f:
        medians={r['score']:float(r['median']) for r in csv.DictReader(f)
            if r['group']=='aircraft-airborne (2026-10-04)' and r['score'] in COMPONENTS}
    latches=[AnomalyLatch(scorer.cutoff,config['latch_s']) for _ in observed]
    timeline=[];forecast_records=[];last=None;last_frame=None;last_future=None;medoid=None
    (directory/'forecasts').mkdir(exist_ok=True)
    for frame,time_s in enumerate(stream['times']):
        if frame>=18 and frame%config['forecast_stride_frames']==0:
            seed=stable_seed(config['seed'],scene['id'],frame)
            graph=observation_graph(observed,valid,frame,model.token_processor.meta['pos_scale'])
            path=directory/'forecasts'/f'f{frame:04d}.npz'
            if resume and path.exists():
                with np.load(path) as a:
                    assert int(a['seed'])==seed;last={key:a[key].copy() for key in ['position','heading']}
            else:
                last=infer(model,graph,seed)
            medoid=medoid_index(last['position']);cv=constant_velocity(observed,frame)
            future=scorer.futures(last['position'],observed[:,frame]);last_future=future_anomaly_evidence(future['state_nll'],scorer.cutoff)
            np.savez_compressed(path,**last,cv=cv,medoid=medoid,seed=seed,current_frame=frame,
                **{f'future_{key}':value for key,value in future.items()})
            entry=forecast_intent(last['position'],config['zone']);cv_entry=forecast_intent(cv[None],config['zone'],min_samples=1)
            forecast_records.append(dict(frame=frame,time_s=float(time_s),seed=seed,medoid_index=medoid,
                prediction_file=str(path.relative_to(output)),future_anomaly=last_future,entry=entry,cv_entry=cv_entry,
                medoid_endpoints=last['position'][medoid,:,-1].tolist()))
            last_frame=frame
            print(json.dumps(dict(scene=scene['id'],forecast_frame=frame,medoid=medoid,entry_samples=[e['entry_samples'] for e in entry])),flush=True)
        if last is not None:
            age=frame-last_frame
            entries=forecast_intent(last['position'][:,:,age:],config['zone'])
        else:entries=[None]*len(observed)
        objects=[]
        for agent,object_id in enumerate(stream['object_ids']):
            raw=float(trace['state_nll'][agent,frame]);latched=latches[agent].update(float(time_s),raw)
            previous=observed[agent,frame-6] if frame>=6 else None
            older=observed[agent,frame-12] if frame>=12 else None
            threat=threat_level(latched,raw,observed[agent,frame],previous,older,entries[agent],
                scene['context'],config['zone'],config['threat'])
            parts={key:float(trace[key][agent,frame]) for key in COMPONENTS}
            horizon=None
            if last_future is not None:
                age_s=(frame-last_frame)/3
                counts=last_future['exceedance_counts'][agent]
                hits=[(i+1)*2-age_s for i,c in enumerate(counts) if c>=8 and (i+1)*2>age_s]
                horizon=min(hits) if hits else None
            objects.append(dict(id=str(object_id),current_state_nll=raw,z_state=float(trace['z_state'][agent,frame]),
                current_exceeds_cutoff=bool(np.isfinite(raw) and raw>scorer.cutoff),anomalous=latched,
                threat=threat,entry=entries[agent],future_anomaly_horizon_s=horizon,
                score_components=parts,explanation=explain_state(parts,medians) if np.isfinite(raw) else [],
                forecast_origin_frame=last_frame,medoid_index=medoid,
                intent=entries[agent]['relation'] if entries[agent] is not None else 'Awaiting observed history'))
        timeline.append(dict(frame=frame,time_s=float(time_s),objects=objects))
    evaluation=evaluate_story(dict(scene,zone=config['zone'],threat=config['threat']),stream,timeline)
    manifest=dict(scene=scene,zone=config['zone'],noise_label=f"Scripted Gaussian sigma={scene['scripted_noise_sigma_m_per_axis']}m per axis",
        source='synthetic foreground + time-shifted aircraft reference; simulated tracker observations',
        cutoff_raw=scorer.cutoff,cutoff_z=scorer.z_cutoff,forecast_count=len(forecast_records),
        anomaly_context='location/height,speed,course only; no weather/time-of-day-conditioned density',
        threat_context='scripted day/night and illustrative protected volume',evaluation=evaluation)
    write_json(directory/'timeline.json',timeline);write_json(directory/'forecasts.json',forecast_records)
    write_json(directory/'manifest.json',manifest)
    rows=[]
    for r in timeline:
        for o in r['objects']:
            rows.append(dict(time_s=r['time_s'],object_id=o['id'],current_state_nll=o['current_state_nll'],
                anomalous=o['anomalous'],threat_level=o['threat']['level'],threat_score=o['threat']['score'],
                entry_samples=o['entry']['entry_samples'] if o['entry'] else None,
                future_anomaly_horizon_s=o['future_anomaly_horizon_s'],
                **{key:o['score_components'][key] for key in COMPONENTS[:3]}))
    with (directory/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(clean_json(rows))
    return manifest
