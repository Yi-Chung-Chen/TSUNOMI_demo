"""Train-only clock context experiment; no neural inference, network or demo edits."""
import csv
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.eval.anomaly_state import load_processed,states_at,enu_bounds
from src.eval.anomaly_score import auroc
from src.eval.context_density import ContextDensity,CONTEXTS,TIMEZONE,context_from_utc,JOINT_PRIOR_STATES,LOCAL_PRIOR_STATES
from src.integrated.data import verify_lock


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def audit_and_fit(root,source,out):
    entries={x['scenario']:x for x in map(json.loads,(source/'index.jsonl').read_text().splitlines())}
    train=(root/'data/splits/train_manifest.txt').read_text().splitlines()
    model=None;coverage=[];audit=[];hours=np.zeros(24,dtype=np.int64)
    for i,name in enumerate(train):
        d=load_processed(root/'data/processed'/f'{name}.npz');e=entries[name]
        assert str(d['date']) in ['2026-09-28','2026-09-29','2026-09-30','2026-10-01','2026-10-02']
        start=float(d['start_utc_s']);assert start==e['start_utc_s']
        meta=json.loads((source/e['path']/'metadata.json').read_text())
        assert meta['start_utc_s']==start
        assert datetime.fromisoformat(e['start_utc']).timestamp()==start
        with (source/e['path']/'reference_tracks_3hz.jsonl').open() as f:row=json.loads(next(f))
        assert abs(row['utc_s']-start-row['frame_index']/3)<1e-5
        assert abs(row['elapsed_s']-row['frame_index']/3)<1e-5
        assert row['source_times_utc_s'][0]<=row['utc_s']<=row['source_times_utc_s'][1]
        audit.append(dict(scenario=name,start_utc=e['start_utc'],start_utc_s=start,
                          metadata_sha256=digest(source/e['path']/'metadata.json'),
                          reference_first_utc_s=row['utc_s'],source='original ADS-B route interpolation; sampled traffic'))
        if model is None:model=ContextDensity(enu_bounds(json.loads(str(d['bounds_json']))))
        bounds=enu_bounds(json.loads(str(d['bounds_json'])));np.testing.assert_array_equal(bounds,model.global_model.bounds)
        agents,frames=np.nonzero(d['valid'][:,6:]&d['valid'][:,:-6]);frames+=6
        states=states_at(d,agents,frames);utc=start+frames/3;labels=context_from_utc(utc)
        model.add(states,utc)
        for label,context in enumerate(CONTEXTS):
            coverage.append(dict(scenario=name,date=str(d['date']),context=context,
                                 eligible_states=int((labels==label).sum()),scenario_start_context=CONTEXTS[int(context_from_utc(start))]))
        local_start=datetime.fromtimestamp(start,timezone.utc).astimezone(ZoneInfo(TIMEZONE))
        local_hours=((local_start.hour*3600+local_start.minute*60+local_start.second+frames/3)//3600).astype(int)%24
        hours+=np.bincount(local_hours,minlength=24)
        if (i+1)%100==0:print(json.dumps(dict(fit_scenarios=i+1)),flush=True)
    model.finish();model.save(out/'state_density.npz')
    # Exact global baseline reproduction prevents a hidden geometry/binning change.
    with np.load(root/'outputs/anomaly_state_smoothed/state_density.npz') as frozen:
        for key in ['p_joint','p_speed','p_course']:np.testing.assert_array_equal(getattr(model.global_model,key),frozen[key])
    write_csv(out/'coverage.csv',coverage);write_csv(out/'timestamp_audit.csv',audit)
    return model,entries,coverage,hours


def score_rows(root,model,entries):
    with (root/'outputs/anomaly_state_smoothed/token_scores.csv').open() as f:source_rows=list(csv.DictReader(f))
    states=dict(position=np.array([[float(r[k]) for k in ['x_m','y_m','height_m']] for r in source_rows]),
                speed=np.array([float(r['speed_mps']) for r in source_rows]),
                course=np.array([float(r['course_rad']) for r in source_rows]))
    drone_meta=json.loads((root.parent/'SYGARD-Dataset/ord_drone_2026-09-28_1800z_3hz/metadata.json').read_text())
    drone_start=datetime.fromisoformat(drone_meta['interval_utc'][0].replace('Z','+00:00')).timestamp()
    utc=np.array([(drone_start if r['source']=='drone' else entries[r['scenario']]['start_utc_s'])+int(r['token_start_frame'])/3 for r in source_rows])
    labels=context_from_utc(utc)
    baseline=model.global_model.score(states);actual=model.score(states,utc)
    day=model.score_context(states,0);night=model.score_context(states,1)
    for key in ['location_height_nll','speed_nll','course_nll','state_nll']:
        np.testing.assert_allclose(baseline[key],[float(r[key]) for r in source_rows],atol=1e-10,rtol=0)
    rows=[]
    for i,r in enumerate(source_rows):
        row={k:r[k] for k in ['source','date','scenario','agent_id','window_start_frame','token_start_frame','on_ground','x_m','y_m','height_m','speed_mps','course_rad']}
        row.update(utc_s=float(utc[i]),local_iso=datetime.fromtimestamp(utc[i],timezone.utc).astimezone(ZoneInfo(TIMEZONE)).isoformat(),context=CONTEXTS[labels[i]])
        for key in ['location_height_nll','speed_nll','course_nll','state_nll']:
            for name,scores in [('global',baseline),('context',actual),('same_state_day',day),('same_state_night',night)]:row[f'{name}_{key}']=float(scores[key][i])
        cell,height=baseline['cell'][i],baseline['height_bin'][i]
        row.update(day_local_support=int(model.models[0].pooled_joint[cell,height]),night_local_support=int(model.models[1].pooled_joint[cell,height]))
        rows.append(row)
    return rows


def evaluate(rows):
    calibration=[r for r in rows if r['source']=='aircraft' and r['date']=='2026-10-03' and r['on_ground']=='False']
    cutoffs={name:float(np.percentile([r[f'{name}_state_nll'] for r in calibration],95)) for name in ['global','context']}
    positives=[r for r in rows if r['source']=='drone'];summary=[]
    for date in ['2026-10-03','2026-10-04']:
        for ground in ['False','True']:
            for context in ['all',*CONTEXTS]:
                group=[r for r in rows if r['source']=='aircraft' and r['date']==date and r['on_ground']==ground and (context=='all' or r['context']==context)]
                if not group:continue
                for name in ['global','context']:
                    values=np.array([r[f'{name}_state_nll'] for r in group]);p=np.array([r[f'{name}_state_nll'] for r in positives])
                    summary.append(dict(date=date,ground=ground,clock_group=context,model=name,tokens=len(values),
                                        mean_state_nll=float(values.mean()),median_state_nll=float(np.median(values)),
                                        p95_state_nll=float(np.percentile(values,95)),cutoff_10_03_airborne_p95=cutoffs[name],
                                        fraction_above_cutoff=float(np.mean(values>cutoffs[name])),
                                        drone_fraction_above_cutoff=float(np.mean(p>cutoffs[name])),auroc_drone_positive=auroc(p,values)))
    return cutoffs,summary


def plots(out,model,rows,hours,cutoffs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(np.arange(24),hours,color=['#e6ab02' if 7<=h<19 else '#7570b3' for h in range(24)])
    axes[0].set(xlabel='Chicago local clock hour',ylabel='Eligible train states (3 Hz)',title='Training coverage; sampled traffic')
    ratio=np.log2(model.models[0].p_joint[:-1].sum(1)/model.models[1].p_joint[:-1].sum(1)).reshape(model.global_model.ny,model.global_model.nx)
    vmax=max(1,float(np.percentile(np.abs(ratio),98)))
    image=axes[1].imshow(ratio,origin='lower',cmap='coolwarm',vmin=-vmax,vmax=vmax,extent=[*model.global_model.bounds[:,0],*model.global_model.bounds[:,1]])
    axes[1].set(title='Learned location probability: log2(day / night)',xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(image,ax=axes[1],label='Positive = more typical by day');fig.tight_layout();fig.savefig(out/'context_density.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,date in zip(axes,['2026-10-03','2026-10-04']):
        air=[r for r in rows if r['source']=='aircraft' and r['date']==date and r['on_ground']=='False']
        delta=np.array([r['same_state_night_state_nll']-r['same_state_day_state_nll'] for r in air])
        ax.hist(delta,bins=50,color='#407ca2');ax.axvline(0,color='black',lw=1)
        ax.set(title=f'{date}: identical aircraft states, changed clock only',xlabel='NLL(night) − NLL(day)',ylabel='Token rows')
    fig.tight_layout();fig.savefig(out/'same_state_context_effect.png',dpi=180);plt.close(fig)
    # Fixed 10th/90th quantile examples from calibration day only, with >=100
    # pooled states in BOTH contexts. Illustrative selection, not test performance.
    eligible=[r for r in rows if r['source']=='aircraft' and r['date']=='2026-10-03' and r['day_local_support']>=100 and r['night_local_support']>=100]
    eligible=sorted(eligible,key=lambda r:r['same_state_night_state_nll']-r['same_state_day_state_nll'])
    examples=[eligible[round(q*(len(eligible)-1))] for q in [.1,.9]] if eligible else []
    if examples:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,row in zip(axes,examples):
            bottom=np.zeros(2)
            for factor,label,color in [('location_height_nll','Location / height','#4c78a8'),('speed_nll','Speed','#f58518'),('course_nll','Direction','#54a24b')]:
                values=[row[f'same_state_{context}_{factor}'] for context in ['day','night']]
                ax.bar(['07–19 local','19–07 local'],values,bottom=bottom,label=label,color=color);bottom+=values
            ax.axhline(cutoffs['context'],color='red',ls='--',label='10-03 p95 cutoff')
            ax.set(title=f"Same state: {row['scenario']} / {row['agent_id']}\nheight {float(row['height_m']):.0f}m; speed {float(row['speed_mps']):.1f}m/s",ylabel='Conditional state NLL')
            ax.legend(fontsize=8)
        fig.suptitle('Illustrative 10-03 10th/90th-percentile context effects; not selected on 10-04')
        fig.tight_layout();fig.savefig(out/'paired_examples.png',dpi=180);plt.close(fig)
    write_csv(out/'paired_examples.csv',examples) if examples else None
    return examples


def main():
    started=time.monotonic();root=Path(__file__).resolve().parents[1];out=root/'outputs/anomaly_context_time';out.mkdir(exist_ok=True)
    source=root.parent/'SYGARD-Dataset/ord_aircraft_training_800_3hz'
    lock=verify_lock(root)
    protected={str(p.relative_to(root)):digest(p) for folder in ['data/splits','outputs/anomaly_state_smoothed','outputs/integrated_demo'] for p in (root/folder).rglob('*') if p.is_file()}
    write_json=lambda path,value:Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    model,entries,coverage,hours=audit_and_fit(root,source,out)
    rows=score_rows(root,model,entries);write_csv(out/'token_scores.csv',rows)
    cutoffs,summary=evaluate(rows);write_csv(out/'summary.csv',summary)
    examples=plots(out,model,rows,hours,cutoffs)
    selection=lambda name:next(r for r in summary if r['date']=='2026-10-04' and r['ground']=='False' and r['clock_group']=='all' and r['model']==name)
    old,new=selection('global'),selection('context')
    delta=np.array([r['same_state_night_state_nll']-r['same_state_day_state_nll'] for r in rows if r['source']=='aircraft' and r['date']=='2026-10-04'])
    flips=sum((r['same_state_night_state_nll']>cutoffs['context'])!=(r['same_state_day_state_nll']>cutoffs['context']) for r in rows if r['source']=='aircraft' and r['date']=='2026-10-04')
    counts={context:sum(c['eligible_states'] for c in coverage if c['context']==context) for context in CONTEXTS}
    support_summary=[]
    for minimum in [0,100,1000]:
        items=[r for r in rows if r['source']=='aircraft' and r['date']=='2026-10-04' and min(r['day_local_support'],r['night_local_support'])>=minimum]
        support_summary.append(dict(minimum_pooled_support_both_contexts=minimum,tokens=len(items),
            flag_flips=sum((r['same_state_night_state_nll']>cutoffs['context'])!=(r['same_state_day_state_nll']>cutoffs['context']) for r in items)))
    receipt=dict(stage='context_time_experiment',timezone=TIMEZONE,clock_periods=CONTEXTS,train_scenarios=618,train_states=counts,
                 fit_dates=['2026-09-28','2026-09-29','2026-09-30','2026-10-01','2026-10-02'],eval_tokens=len(rows),cutoffs=cutoffs,
                 cutoff_source='10-03 airborne p95, per model, common to both clock contexts; no10-04 calibration',
                 joint_prior_states=JOINT_PRIOR_STATES,local_prior_states=LOCAL_PRIOR_STATES,
                 exact_global_baseline_reproduced=True,source_timestamp_audit_count=618,
                 counterfactual_10_04=dict(median_night_minus_day=float(np.median(delta)),p10=float(np.percentile(delta,10)),p90=float(np.percentile(delta,90)),flag_flips=flips,total=len(delta),support_strata=support_summary),
                 primary_10_04_airborne=dict(global_model=old,context_model=new),elapsed_s=time.monotonic()-started,
                 model_inference=False,neural_training=False,demo_modified=False,weather_used=False)
    assert protected=={p:digest(root/p) for p in protected};assert lock==verify_lock(root)
    receipt['protected_files_unchanged']=len(protected);write_json(out/'acceptance.json',receipt)
    method=f'''# Clock-conditioned grid anomaly experiment

Condition the existing250m grid/height, speed and course density on Chicago local clock07:00–19:00 versus19:00–07:00 (America/Chicago; DST conversion). These are clock periods, not solar day/night or weather categories. UTC = preserved scenario start + frame/3. Audited all618train metadata/index/processed starts and the first calculated state's timestamp/bracketing source times. Original irregular ADS-B samples are documented in the read-only source dataset. No external weather/time verification or completeness claim.

Fit only existing09-28..10-02train states at valid endpoints2s apart, unchanged bins, spatial pooling and speed/course kernels. Global model reproduces frozen M8b probabilities and all28413token state scores exactly. Conditional joint probabilities use1000global-prior pseudo-states; local speed/course conditionals use100global-prior pseudo-states. Fixed before evaluation; not tuned. Unsupported local speed/course conditionals fall back to global; a wholly absent context reproduces global. Location probability uses a globally normalized prior, not per-cell exact fallback; sparse location support may still produce large differences. Pooled3Hz counts are correlated and are not effective independent sample sizes.

Train context states: {counts}. All618train scenarios are used; start periods388day/230night. Evaluate the same existing M8b token rows, using actual aircraft timestamps. The single synthetic drone is tagged by its scripted source interval (18:00UTC/13:00Chicago), not an observed real-world drone recording. Counterfactual day/night scoring changes only the context of the identical state, not its coordinates or trajectory.

Calibrate separate global/context p95 cutoffs on10-03AIRBORNE tokens only (each common across contexts);10-04is report-only. Existing demo cutoff and scores stay unchanged. Aircraft flag fraction is a reference-data exceedance rate, not verified threat false positives. AUROC compares one scripted drone route against aircraft, not evidence of real-world threat detection; night-subgroup AUROC compares daytime drone positives against night aircraft and is not a matched-context metric. No uncertainty/significance claims; inspected development data, not a pristine final test.

Source dataset is sampled traffic, some cached daily-history subsets; coverage varies by hour and may confound context with flow/operations or selection. Only5train days and one drone route. Conditional distributions describe the provided sample; they cannot establish that nighttime is inherently suspicious, or that weather/time changes intent. No event-rate/traffic-count model or threat-rule modification.

# Results

10-04airborne token AUROC: global {old['auroc_drone_positive']:.5f} → conditional {new['auroc_drone_positive']:.5f}.
Mean state NLL: {old['mean_state_nll']:.4f} → {new['mean_state_nll']:.4f}.
Fraction exceeding their respective10-03p95 cutoffs: {100*old['fraction_above_cutoff']:.2f}% → {100*new['fraction_above_cutoff']:.2f}%.
Drone fraction exceeding cutoff: {100*old['drone_fraction_above_cutoff']:.2f}% → {100*new['drone_fraction_above_cutoff']:.2f}%.
Same10-04aircraft states, night-minus-day NLL p10/median/p90: {np.percentile(delta,[10,50,90]).tolist()}. {flips}/{len(delta)}change binary anomaly flag when context alone changes.

The context effect is learned and bidirectional: whether the score increases or decreases depends on the state and the observed clock-conditioned distribution. Accuracy need not improve. Conditional-location differences can reflect unequal coverage or operating flow as well as time. Support-stratified flip counts are recorded in acceptance.json. This is an isolated experiment; no deployed detector or demo changed.

# Outputs

context_density.png: train hourly coverage and learned day/night location ratio. same_state_context_effect.png: identical-state score differences. paired_examples.png/csv: fixed10th/90th-quantile examples from10-03with>=100pooled states in BOTH contexts, illustrative and explicitly selected, not10-04performance. token_scores.csv: actual-clock/global and both counterfactual component scores. coverage.csv/timestamp_audit.csv/state_density.npz/summary.csv/acceptance.json support reproduction. Run: python -m scripts.anomaly_context_time. No model execution, packages, network, weather input or demo edits.
'''
    (out/'summary.md').write_text(method)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
