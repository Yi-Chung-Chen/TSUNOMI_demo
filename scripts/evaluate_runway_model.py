"""CPU M10 evaluation reusing unchanged M6 metrics and M8 scoring functions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
import torch
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.metrics.cross_entropy import token_cross_entropy
from src.rollout.closed_loop import rollout
from src.rollout.baselines import baselines
from src.rollout.rollout_io import save_rollout_window
from src.eval.load_rollout import load_rollout
from src.eval.metrics.horizon_ade import compute
from src.eval.aggregate import aggregate
from src.eval.anomaly_score import iter_scores,load_drone,summarize,write_csv
from scripts.run_demo import window_seed


def read_checkpoint(root,name):
    if name=='epoch_016':return root/'outputs/air_pre_bc/checkpoints/epoch_016.ckpt'
    done=json.loads((root/'outputs/runway_anchor/training'/name/'completed.json').read_text())
    return root/done['best_checkpoint']


def m8_key(row):
    return tuple(str(row[k]) for k in ['source','date','scenario','agent_id','window_start_frame','token_start_frame'])


def verify_m8_rows(records,original):
    lookup={m8_key(r):r for r in original}
    assert len(records)==len(original)==len(lookup)==28413
    for row in records:
        old=lookup[m8_key(row)]
        for key in ['chain_token','residual_token']:assert row[key]==int(old[key])
        for key in ['residual_horizontal_m','residual_vertical_m']:
            assert abs(row[key]-float(old[key]))<1e-9


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',required=True,choices=['epoch_016','base20','anchor'])
    args=p.parse_args();root=Path(__file__).resolve().parents[1];output=root/'outputs/runway_anchor/evaluation'/args.model
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    checkpoint=read_checkpoint(root,args.model);sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);model=AirSMART(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'],strict=True);model.eval().requires_grad_(False)
    assert next(model.parameters()).device.type=='cpu'
    dataset=AirDataset(root/'data/processed',root/'data/splits','val');assert len(dataset)==912
    records=[];window_records=[];ce=defaultdict(lambda:dict(count=0,loss=0.,ce=0.,accuracy=0.));started=time.monotonic()
    for index in range(len(dataset)):
        graph=dataset[index];name=f"{graph['scenario']}_f{graph['start_frame']:04d}.npz";date=graph['date']
        with torch.inference_mode():
            logits,tokens=model(graph);scores=token_cross_entropy(logits,tokens,model.current_node,model.hparams.label_smoothing)
            count=scores['count'];ce[date]['count']+=count
            for source,target in [('loss','loss'),('unsmoothed_ce','ce'),('accuracy','accuracy')]:ce[date][target]+=float(scores[source])*count
            seed=window_seed(817,graph['scenario'],graph['start_frame'])
            path=output/'rollouts'/name
            if path.exists():
                cached=load_rollout(path);assert cached['meta']['checkpoint_sha256']==sha and cached['meta']['seed']==seed
            else:
                prediction=rollout(model,graph,n_rollouts=32,num_k=5,temp=1.,seed=seed)
                save_rollout_window(graph,prediction,path,'bc',seed,sha)
            measured=compute(load_rollout(path))
            for row in measured:
                record=dict(model=args.model,**row);records.append(record)
                window_records.append(dict(scenario=graph['scenario'],start_frame=graph['start_frame'],**record))
            if args.model=='epoch_016':
                cv_path=output/'constant_velocity'/name
                save_rollout_window(graph,baselines(graph)['constant_velocity'],cv_path,'constant_velocity',seed)
                for row in compute(load_rollout(cv_path)):
                    record=dict(model='constant_velocity',**row);records.append(record)
                    window_records.append(dict(scenario=graph['scenario'],start_frame=graph['start_frame'],**record))
        if (index+1)%50==0 or index+1==len(dataset):
            print(json.dumps(dict(model=args.model,windows=index+1,total=len(dataset),elapsed_s=round(time.monotonic()-started,1))),flush=True)
    ce_rows=[]
    for date,values in sorted(ce.items()):
        n=values['count'];ce_rows.append(dict(model=args.model,date=date,n_tokens=n,**{k:values[k]/n for k in ['loss','ce','accuracy']}))
    write_csv(output/'ce.csv',ce_rows);write_csv(output/'rollout_summary.csv',aggregate(records));write_csv(output/'window_metrics.csv',window_records)
    vocab=json.loads((root/'data/splits/type_vocab.json').read_text());drone,drone_info=load_drone()
    anomaly=list(iter_scores(model,drone,'ord_drone_2026-09-28_1800z_3hz','drone',vocab,64,'cpu'))
    scenarios=(root/'data/splits/val_manifest.txt').read_text().splitlines()
    for i,scenario in enumerate(scenarios):
        with np.load(root/'data/processed'/f'{scenario}.npz') as arrays:data={k:arrays[k] for k in arrays.files}
        anomaly.extend(iter_scores(model,data,scenario,'aircraft',vocab,64,'cpu'))
        if (i+1)%50==0:print(json.dumps(dict(model=args.model,anomaly_scenarios=i+1,rows=len(anomaly))),flush=True)
    with (root/'outputs/anomaly/token_scores.csv').open() as stream:original=list(csv.DictReader(stream))
    verify_m8_rows(anomaly,original)
    reference_nll_error=None
    if args.model=='epoch_016':
        old={m8_key(r):float(r['nll']) for r in original}
        reference_nll_error=max(abs(r['nll']-old[m8_key(r)]) for r in anomaly)
        assert reference_nll_error<1e-4, f'Legacy M8 NLL reproduction mismatch: {reference_nll_error}'
    write_csv(output/'anomaly_token_scores.csv',anomaly);write_csv(output/'anomaly_summary.csv',summarize(anomaly))
    receipt=dict(model=args.model,checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=sha,
        checkpoint_epoch=ckpt['epoch'],device='cpu',windows=912,dates={k:sum(r['date']==k and r['group']=='all' and r['horizon_s']==30 and r['axis']=='horizontal' and r['model']==args.model for r in window_records) for k in ce},
        seed=817,n_rollouts=32,topk=5,temp=1.,anomaly_rows=len(anomaly),m8_rows_tokens_residuals_identical=True,
        elapsed_s=time.monotonic()-started,drone_info=drone_info,m8_batch_size=64,legacy_m8_max_nll_error=reference_nll_error)
    (output/'completed.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
