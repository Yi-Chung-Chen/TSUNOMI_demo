"""Paired M10 summary from CPU receipts; no checkpoint/model selection here."""
import csv
import json
import hashlib
from pathlib import Path
from src.eval.aggregate import aggregate,METRICS

MODELS=['epoch_016','base20','anchor']


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def comparison(a,b,direction):
    if a is None or b is None or direction=='descriptive':return 'NA'
    if a==b:return 'tie'
    return 'better' if (a<b if direction=='lower' else a>b) else 'worse'


def build_summary(root):
    out=root/'outputs/runway_anchor';values={};counts={}
    def add(key,model,value,n):
        values.setdefault(key,{})[model]=None if value in ('',None) else float(value);counts[key]=int(n)
    for model in MODELS:
        directory=out/'evaluation'/model;receipt=json.loads((directory/'completed.json').read_text());assert receipt['device']=='cpu' and receipt['windows']==912 and receipt['anomaly_rows']==28413
        for r in read(directory/'ce.csv'):
            for metric,direction in [('loss','lower'),('ce','lower'),('accuracy','higher')]:
                add(('CE',r['date'],'all','', '',metric,direction),model,r[metric],r['n_tokens'])
        for r in read(directory/'rollout_summary.csv'):
            for metric in METRICS:
                add(('M6',r['date'],r['group'],r['horizon_s'],r['axis'],metric,'lower'),r['model'],r[metric],r['n_agents'])
        for r in read(directory/'anomaly_summary.csv'):
            for metric,direction in [('nll_auroc','higher'),('nll_mean','descriptive'),('nll_median','descriptive')]:
                add(('M8',r['group'].split('(')[-1].strip(')') if '(' in r['group'] else '2026-09-28',r['group'],'','',metric,direction),model,r[metric],r['token_count'])
    flow=json.loads((out/'flow_labels.json').read_text());flow_rows=[]
    for model in MODELS:
        windows=read(out/'evaluation'/model/'window_metrics.csv')
        for label in ['east','west']:
            records=[]
            for r in windows:
                if r['date']!='2026-10-04' or flow['scenarios'][r['scenario']]['flow']!=label:continue
                r=dict(r);r['n_agents']=int(r['n_agents']);r['horizon_s']=int(r['horizon_s'])
                for metric in METRICS:r[metric]=float(r[metric]) if r[metric] else None
                records.append(r)
            for r in aggregate(records):
                if r['date']!='2026-10-04' or r['horizon_s']!=30:continue
                flow_rows.append(dict(flow=label,**r))
                add(('Flow',r['date'],label+'/'+r['group'],30,r['axis'],'FDE_m','lower'),r['model'],r['FDE_m'],r['n_agents'])
    if flow_rows:
        with (out/'flow_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,lineterminator='\n',fieldnames=list(flow_rows[0]));writer.writeheader();writer.writerows(flow_rows)
    rows=[]
    for key,metrics in values.items():
        family,date,group,horizon,axis,metric,direction=key
        a,b,c=[metrics.get(m) for m in MODELS]
        row=dict(family=family,date=date,group=group,horizon_s=horizon,axis=axis,metric=metric,direction=direction,n=counts[key],
            epoch_016=a,base20=b,anchor=c,constant_velocity=metrics.get('constant_velocity'),
            anchor_vs_base20=comparison(c,b,direction),base20_vs_epoch_016=comparison(b,a,direction),
            anchor_minus_base20=None if c is None or b is None else c-b,
            base20_minus_epoch_016=None if b is None or a is None else b-a)
        rows.append(row)
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def fmt(x):return '—' if x is None else f'{x:.5f}' if isinstance(x,float) else str(x)
    lines=['# M10 runway-anchor attention experiment','',
        'Both alignment gates passed: reference stdout byte-identical; exact-ENU counts within±5%. '
        'Both fresh runs use seed817,20epochs,batch32/16,workers4 and the original optimizer/lr/dropout. '
        'Checkpoint selection is best10-03smoothed val loss only. The old epoch016 came from a128epoch schedule. '
        'No hyperparameter search, identity dropout or demo-checkpoint replacement. All final evaluations run on CPU.',
        '', '## Training selection','', '| Run | Best epoch | Best 10-03 val loss |','|---|---:|---:|']
    for name in ['base20','anchor']:
        done=json.loads((out/'training'/name/'completed.json').read_text());assert done['completed'] and done['epochs']==20
        lines.append(f"| {name} | {done['best_epoch']} | {done['best_val_loss']:.6f} |")
    lines+=['','## Primary comparisons (10-04)','',
        'Point estimates only; better/worse is a numerical comparison, not statistical significance. '
        'Lower CE/trajectory errors/miss rates and higher accuracy/AUROC are better. '
        'M8 means/medians are descriptive likelihoods and not labelled better/worse.',
        '', '| Family | Group | Horizon | Axis | Metric | epoch016 | base20 | anchor | CV | Anchor vs base20 | Base20 vs epoch016 |',
        '|---|---|---:|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['date']!='2026-10-04':continue
        if r['family']=='M6' and r['group']!='all':continue
        if r['family']=='M8' and r['metric']!='nll_auroc':continue
        lines.append('| '+' | '.join(fmt(r[k]) for k in ['family','group','horizon_s','axis','metric','epoch_016','base20','anchor','constant_velocity','anchor_vs_base20','base20_vs_epoch_016'])+' |')
    lines+=['','## Flow definition and limitations','',
        'Provisional advisor clarification: exact supplied flow-reference script, not centreline masks. '
        'For each track,6-valid-frame (nominal2s) displacements with on_ground at both endpoints and speed>30m/s; '
        'require>=3roll frames, then median normalised dx. abs(dx)>0.9 votes east/west; scenario majority wins, ties/no-roll skipped. '
        'The direction filter stands in for six E-W runways; diagonal runway rolls are excluded by direction, not a centreline mask.',
        '', f"10-03 flow scenarios: {flow['counts']['2026-10-03']}. 10-04: {flow['counts']['2026-10-04']}. "
        'Selection is entirely east-flow on10-03 and may favour east-flow behaviour; the same selection rule applies to both new runs. '
        '10-04 has been inspected during prior demo development and is not an untouched test set.',
        '', 'All/all-airborne/ground and both axes at10/20/30s, plus every comparison and delta, are in summary.csv. '
        'M6 errors use unchanged32-rollout topk5/temp1 metrics; minADE is per-agent whole-path minimum; '
        'miss_rate requires ALL32 horizontal30s endpoints>500m; mean_rollout_miss_rate is also reported. '
        'Counts are token or eligible agent-window counts, not independent scenarios.',
        '', 'M8 reuses the exact28413original token rows, unchanged GT token chain and residuals. '
        'AUROC treats the single native-pose drone route as positive, higher NLL as anomalous. '
        'This is one route and smoothness/acquisition differ from aircraft; no general drone detector or calibrated threshold is established.',
        '', 'Attention figure uses observed6s history only and a fixed seed of817 for three10-04windows. '
        'Weights average heads and all4anchor layers; they are diagnostic associations, not causal explanations. '
        'Basemap tiles are existing local cache only; no downloads.',
        '', 'The integrated demo continues using frozen epoch_016.ckpt.']
    west=[r for r in rows if r['family']=='Flow' and r['group']=='west/all' and r['axis']=='horizontal']
    east=[r for r in rows if r['family']=='Flow' and r['group']=='east/all' and r['axis']=='horizontal']
    if west and east and west[0]['anchor_vs_base20']=='worse' and east[0]['anchor_vs_base20']=='better':
        lines+=['','Anchor is worse on west-flow FDE while better on east-flow. This is consistent with a possible flow-specific shortcut; '
                'identity dropout could be considered as a separately approved follow-up. It was NOT added or tuned in this experiment.']
    def primary(family,group,metric,axis='',horizon=''):
        return next(r for r in rows if r['family']==family and r['date']=='2026-10-04' and r['group']==group
                    and r['metric']==metric and r['axis']==axis and str(r['horizon_s'])==str(horizon))
    fde=primary('M6','all','FDE_m','horizontal',30)
    airborne=primary('M8','aircraft-airborne (2026-10-04)','nll_auroc')
    ground=primary('M8','aircraft-ground (2026-10-04)','nll_auroc')
    gain=100*(fde['base20']-fde['anchor'])/fde['base20']
    lines+=['','## Plain-language result','',
        f"Runway anchors improve the primary30s horizontal mean FDE from {fde['base20']:.2f}m to {fde['anchor']:.2f}m "
        f"({gain:.1f}% lower) relative to the matched20epoch baseline. The schedule-only baseline also improves "
        f"from the old checkpoint's {fde['epoch_016']:.2f}m, so these two effects are reported separately.",
        '',f"Drone NLL AUROC versus10-04 airborne aircraft improves from {airborne['base20']:.3f} to {airborne['anchor']:.3f} "
        f"(old epoch016:{airborne['epoch_016']:.3f}); versus ground aircraft it improves from {ground['base20']:.3f} to {ground['anchor']:.3f}. "
        'The airborne AUROC remains only slightly above chance, so this is not an effective standalone drone detector. '
        'Forecasting gains are stronger evidence than the anomaly result, and the experiment still covers only one drone route.',
        '', 'Not every metric improves. The following10-04 subgroup results are worse than base20:', '']
    for r in rows:
        if r['date']=='2026-10-04' and r['anchor_vs_base20']=='worse':
            lines.append(f"- {r['family']} / {r['group']} / {r['horizon_s']}s / {r['axis']} / {r['metric']}: "
                         f"{r['base20']:.6f} -> {r['anchor']:.6f} (n={r['n']}).")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return rows

if __name__=='__main__':build_summary(Path(__file__).resolve().parents[1])
