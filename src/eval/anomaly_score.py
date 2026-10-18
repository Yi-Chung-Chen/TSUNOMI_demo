"""M8: isolated-agent, teacher-forced first-future-token anomaly scores."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData

from src.common.units import FRAME_HZ, GROUND_ALTITUDE_M, to_internal, to_metres
from src.model.air_smart import AirSMART
from scripts.merge_demo_val import protected_hashes

DRONE_PATH = Path('/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_drone_2026-09-28_1800z_3hz/simulation_poses_native.jsonl')
CURRENT, SHIFT, TOTAL = 18, 6, 109


def interpolate(times, values, query):
    """Piecewise linear interpolation on native timestamps; no extrapolation."""
    times, values, query = np.asarray(times), np.asarray(values), np.asarray(query)
    if len(times) < 2 or np.any(np.diff(times) < 0) or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError('Expected finite values at increasing timestamps')
    unique,index,inverse=np.unique(times,return_index=True,return_inverse=True)
    # Native reads195/196 repeat the exact same time and pose. Collapse exact
    # duplicates only; conflicting values cannot define a unique interpolant.
    if not np.array_equal(values,values[index][inverse]) or len(unique)<2:
        raise ValueError('Conflicting duplicate timestamps')
    times,values=unique,values[index]
    if values.ndim == 1:
        return np.interp(query, times, values, left=np.nan, right=np.nan)
    return np.stack([np.interp(query, times, values[:, j], left=np.nan, right=np.nan)
                     for j in range(values.shape[1])], axis=-1)


def load_drone(path=DRONE_PATH):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if {r['source_object_id'] for r in rows} != {'drone01'}:
        raise ValueError('Expected only drone01 native poses')
    times = np.array([r['elapsed_s'] for r in rows], dtype=float)
    position = np.array([r['xyz_enu_m'] for r in rows], dtype=float)
    # Raw drone ENU uses the dataset's 2250 m altitude origin. The local-plane
    # approximation differs from aircraft geodetic altitude by ~2m at 5.6km.
    position[:, 2] += 2250. - GROUND_ALTITUDE_M
    heading = np.unwrap([r['heading_rad'] for r in rows])
    increasing=np.diff(times)>0
    velocity=np.diff(position,axis=0)[increasing]/np.diff(times)[increasing,None]
    moving=np.linalg.norm(velocity[:,:2],axis=-1)>5.
    delta=heading[1:][increasing]-np.arctan2(velocity[:,1],velocity[:,0])
    heading_error=float(np.median(np.abs(np.arctan2(np.sin(delta[moving]),np.cos(delta[moving])))))
    # Clearly incompatible compass/sign conventions require an explicit decision.
    if heading_error > np.pi/4:
        raise ValueError(f'Drone heading convention mismatch ({heading_error} radians); consult advisor')
    grid = np.arange(901)/FRAME_HZ
    xyz, h = interpolate(times, position, grid), interpolate(times, heading, grid)
    valid = np.isfinite(xyz).all(-1) & np.isfinite(h)
    data = dict(pos=np.nan_to_num(xyz)[None], heading=np.nan_to_num(np.arctan2(np.sin(h),np.cos(h)))[None],
                valid=valid[None], on_ground=np.zeros((1,901),bool), agent_id=np.array(['drone01']),
                agent_type=np.array(['UNK']), date=np.asarray('2026-09-28'))
    info = dict(native_reads=len(rows),unique_native_timestamps=len(np.unique(times)),
                identical_duplicate_reads=len(rows)-len(np.unique(times)),native_time_range_s=[float(times[0]),float(times[-1])],
                grid_frames=901, grid_valid_frames=int(valid.sum()), invalid_grid_frames=np.flatnonzero(~valid).tolist(),
                heading_median_absolute_velocity_difference_rad=heading_error,
                native_knot_interpolation_max_error_m=float(np.abs(interpolate(times,position,times)-position).max()))
    return data, info


def auroc(positive, negative):
    """P(positive > negative) + .5 P(tie), from average ranks; no sklearn."""
    positive, negative = np.asarray(positive,dtype=float), np.asarray(negative,dtype=float)
    if not len(positive) or not len(negative):
        raise ValueError('AUROC needs both classes')
    values = np.concatenate((positive,negative))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    order = np.argsort(values,kind='stable')
    sorted_values=values[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_values)!=0)+1]
    ends=np.r_[starts[1:],len(values)]
    ranks=np.empty(len(values),dtype=float)
    ranks[order]=np.repeat((starts+1+ends)/2,ends-starts)
    n,m=len(positive),len(negative)
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def solo_window(data, agent, start, vocab, pos_scale=10.):
    sl=slice(start,min(start+TOTAL,data['valid'].shape[1]))
    valid=data['valid'][agent,start:start+CURRENT+SHIFT+1]
    if start<0 or len(valid)!=CURRENT+SHIFT+1 or not valid.all():
        raise ValueError('Scoring requires full history and scored segment')
    length=sl.stop-start
    def padded(key):
        values=data[key][agent:agent+1,sl].copy()
        return np.pad(values,[(0,0),(0,TOTAL-length)]+([(0,0)] if values.ndim==3 else []))
    graph=HeteroData(); a=graph['agent']; a.num_nodes=1
    a['position']=torch.from_numpy(to_internal(padded('pos').astype(np.float64),pos_scale))
    a['heading']=torch.from_numpy(padded('heading').astype(np.float64))
    a['valid_mask']=torch.from_numpy(padded('valid'))
    a['on_ground']=torch.from_numpy(padded('on_ground'))
    a['type']=torch.tensor([vocab.get(str(data['agent_type'][agent]),0)],dtype=torch.long)
    a['radius']=torch.full((1,),2.)
    # M8 eligibility is history + first2s; do not require M5's additional10s.
    a['train_mask']=torch.ones(1,dtype=torch.bool)
    return graph


@torch.inference_mode()
def score_batch(model, graphs, device='cpu'):
    """Each graph is a separate one-agent scene, including when batched."""
    if any(g['agent'].num_nodes != 1 for g in graphs):
        raise ValueError('M8 scores agents alone')
    batch=Batch.from_data_list(graphs).to(device)
    logits,tokens=model(batch)
    c=model.current_node; p=model.token_processor; a=batch['agent']
    if (c,p.shift) != (3,SHIFT) or p.num_k != 1:
        raise ValueError('M8 requires default deterministic 2s tokenizer / 6s history')
    nll=F.cross_entropy(logits[:,c],tokens['gt_idx'][:,c+1],reduction='none')
    # Reconstruction diagnostic is independent of accumulated tokenizer drift:
    # match one segment from its raw anchor using the unchanged matching metric.
    raw=a['position'].double(); h=a['heading'].double(); v=a['valid_mask']; ground=a['on_ground']
    end=CURRENT+SHIFT
    recon=p.reconstruct(raw,h,v,ground,start_frame=CURRENT,horizon_frames=SHIFT,one_shot=True)
    difference=to_metres(recon['dense_position']-raw[:,CURRENT+1:end+1],p.meta['pos_scale'])
    return dict(nll=nll.cpu().numpy(), chain_token=tokens['gt_idx'][:,c+1].cpu().numpy(),
                residual_token=recon['idx'][:,1].cpu().numpy(),
                residual_horizontal_m=difference[...,:2].norm(dim=-1).mean(-1).cpu().numpy(),
                residual_vertical_m=difference[...,2].abs().mean(-1).cpu().numpy())


def iter_scores(model, data, scenario, source, vocab, batch_size, device):
    graphs,metadata=[],[]
    for start in range(0,data['valid'].shape[1]-CURRENT-SHIFT,SHIFT):
        eligible=data['valid'][:,start:start+CURRENT+SHIFT+1].all(-1)
        for a in np.flatnonzero(eligible):
            graphs.append(solo_window(data,int(a),start,vocab,model.token_processor.meta['pos_scale']))
            metadata.append(dict(source=source,date=str(data['date']),scenario=scenario,
                agent_id=str(data['agent_id'][a]),window_start_frame=start,token_start_frame=start+CURRENT,
                token_start_s=(start+CURRENT)/FRAME_HZ,on_ground=bool(data['on_ground'][a,start+CURRENT])))
            if len(graphs)==batch_size:
                scores=score_batch(model,graphs,device)
                yield from [dict(row,**{k:values[i].item() for k,values in scores.items()}) for i,row in enumerate(metadata)]
                graphs,metadata=[],[]
    if graphs:
        scores=score_batch(model,graphs,device)
        yield from [dict(row,**{k:values[i].item() for k,values in scores.items()}) for i,row in enumerate(metadata)]


def summarize(records):
    drone=[r for r in records if r['source']=='drone']
    groups=[('drone',drone)]
    for date in ['2026-10-04','2026-10-03']:
        for ground in [False,True]:
            label=f'aircraft-{"ground" if ground else "airborne"} ({date})'
            groups.append((label,[r for r in records if r['source']=='aircraft' and r['date']==date and r['on_ground']==ground]))
    result=[]
    for name,rows in groups:
        if not rows:
            raise ValueError(f'Empty group: {name}')
        row=dict(group=name,token_count=len(rows))
        for key in ['nll','residual_horizontal_m','residual_vertical_m']:
            values=np.array([r[key] for r in rows])
            row.update({key+'_mean':float(values.mean()),key+'_median':float(np.median(values)),
                        key+'_p95':float(np.percentile(values,95)),
                        key+'_auroc':None if name=='drone' else auroc([r[key] for r in drone],values)})
        result.append(row)
    return result


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_outputs(records, output, receipt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    rows=summarize(records)
    write_csv(output/'token_scores.csv',records);write_csv(output/'summary.csv',rows)
    drone,air,ground=rows[:3]
    track_scores={}
    for r in records:
        if r['source']=='aircraft' and r['date']=='2026-10-04' and not r['on_ground']:
            track_scores.setdefault((r['scenario'],r['agent_id']),[]).append(r['nll'])
    track_means=np.array([np.mean(v) for v in track_scores.values()])
    drone_mean=drone['nll_mean']
    track_percentile=float(100*np.mean((track_means<drone_mean)+.5*(track_means==drone_mean)))
    receipt['drone_track_mean_nll_percentile_vs_10_04_airborne']=track_percentile
    receipt['normal_airborne_track_segments']=len(track_means)
    groups=[('Drone','#de6b48',[r for r in records if r['source']=='drone']),
            ('Aircraft airborne · 10-04','#287ca4',[r for r in records if r['source']=='aircraft' and r['date']=='2026-10-04' and not r['on_ground']]),
            ('Aircraft ground · 10-04','#43895b',[r for r in records if r['source']=='aircraft' and r['date']=='2026-10-04' and r['on_ground']])]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    bins=np.linspace(0,max(r['nll'] for _,_,g in groups for r in g),45)
    for name,color,g in groups:
        ax.hist([r['nll'] for r in g],bins=bins,density=True,histtype='step',lw=2,color=color,label=f'{name} (n={len(g):,})')
    ax.set(xlabel='First future token NLL (nats / 2 s token; higher = less likely)',ylabel='Density',title='Single-agent teacher-forced transition likelihood')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'nll_hist.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    dr=groups[0][2]
    ax.plot([r['token_start_s'] for r in dr],[r['nll'] for r in dr],color=groups[0][1],lw=1.6,label='Drone')
    for stat,style in [('median','--'),('p95',':')]:
        ax.axhline(air['nll_'+stat],color=groups[1][1],ls=style,label=f'10-04 aircraft airborne {stat}')
    ax.set(xlabel='Time from drone scenario start (s; scored token start)',ylabel='NLL (nats / 2 s token)',title='Drone transition score over time')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'drone_timeline.png',dpi=160);plt.close(fig)
    columns=['group','token_count','nll_mean','nll_median','nll_p95','residual_horizontal_m_mean',
             'residual_horizontal_m_p95','residual_vertical_m_mean','residual_vertical_m_p95',
             'nll_auroc','residual_horizontal_m_auroc','residual_vertical_m_auroc']
    def cell(v):
        return '—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
    lines=['# Anomaly scoring MVP','', 'Higher NLL/residual = more anomalous; AUROC treats drone tokens as positives.',
           '', '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(cell(r[k]) for k in columns)+' |' for r in rows]
    # Quantitative wording avoids assuming that an OOD state must have high NLL.
    lines += ['', '## Plain-language result','',
        f'The drone has median NLL {drone["nll_median"]:.3f}, versus {air["nll_median"]:.3f} for 10-04 airborne aircraft. '
        f'NLL AUROC is {air["nll_auroc"]:.3f} (0.5 is chance; 1 is perfect separation with higher drone scores). '
        + ('The drone is generally assigned higher likelihood than airborne aircraft, so treating high NLL as an anomaly does not identify this drone reliably.' if air['nll_auroc']<.5 else
           'The transition NLL does not clearly separate the drone from normal airborne aircraft.' if air['nll_auroc']<.8 else
           'The transition NLL separates this drone from airborne aircraft in this sample; this is not evidence about other drones.'),
        '',f'The drone track-mean NLL is at the {track_percentile:.1f}th percentile of {len(track_means)} '
        '10-04 airborne aircraft track-segment means (each scenario/agent pair is one segment). This is a descriptive rank, not an extra score.',
        '',f'The drone reconstructs with mean horizontal / vertical residual {drone["residual_horizontal_m_mean"]:.3f} / '
        f'{drone["residual_vertical_m_mean"]:.3f} m, versus {air["residual_horizontal_m_mean"]:.3f} / {air["residual_vertical_m_mean"]:.3f} m '
        f'for airborne aircraft. Residual AUROC is {air["residual_horizontal_m_auroc"]:.3f} horizontally and '
        f'{air["residual_vertical_m_auroc"]:.3f} vertically. '
        + ('The drone reconstructs at least as well as aircraft on average in both axes; the aircraft dictionary does not reject this route.'
           if drone['residual_horizontal_m_mean']<=air['residual_horizontal_m_mean'] and drone['residual_vertical_m_mean']<=air['residual_vertical_m_mean']
           else 'Reconstruction quality differs by axis; compare the horizontal and vertical errors separately.')
        +' These are reconstruction errors, not generated-future forecast errors.',
        '', '## Method and limits','',
        '- Frozen epoch_016.ckpt and existing K2048/shift6 dictionary; eval mode; no training or model changes.',
        '- Each aircraft/drone is an isolated one-agent scene. Batched graphs have distinct scene IDs, so there is no cross-agent attention.',
        '- Re-index 109-frame windows at stride6 over each listed validation scenario, once per (scenario, agent, start). Frames past the scenario end are padded invalid, allowing the last fully observed scored tokens. Existing stride90 demo windows are not reused or duplicated. Only the first future token node3→4 is scored; history0..18 and scored segment18..24 must be valid. No ten-second future eligibility requirement is imposed for this two-second diagnostic.',
        '- NLL uses the default autoregressive GT tokenizer chain and unsmoothed log-softmax, in natural-log units. The decoder is causal: raw future data beyond the scored token is not input to its prediction.',
        '- Residual uses a separate nearest-token match from the raw anchor at frame18, with the existing endpoint contour + vertical matching metric and causal chord headings. Report mean pointwise xy distance / absolute z error over frames19..24, in metres. Its token ID can differ from the chain label; accumulated chain drift is excluded.',
        '- Drone native poses are linearly interpolated at 0..900 / 3 s; heading unwraps before interpolation. Unsupported boundary times are invalid, never extrapolated. Native-knot reproduction tests the interpolant at its original knots, not a lossy grid→native round-trip.',
        '- The native file has902 reads at901 unique timestamps: reads195/196 have identical time65.49s, xyz and heading. They are collapsed for interpolation without changing any raw file; conflicting duplicates raise an error.',
        '- Drone height is native ENU z +2250−166 m; all drone samples are airborne, type UNK. No camera tracks.',
        f'- Native drone heading vs movement median absolute difference is {np.degrees(receipt["drone_interpolation"]["heading_median_absolute_velocity_difference_rad"]):.3f} degrees; its CCW-from-east convention is consistent and unchanged. Two-sided interpolation can use a bracketing read about0.33s later, as aircraft interpolation also uses both sides.',
        '- 10-04 is the primary normal comparison; 10-03 selected the checkpoint/dictionary settings and is reference only.',
        '- Tokens overlap in history, and all drone positives come from one simulated route. Counts are tokens, not independent flights; AUROC is descriptive and no operating threshold or general anomaly-detection claim is established.',
        '- Drone poses are smooth native Unreal reads near3Hz, whereas aircraft positions are piecewise-linear interpolation of ADS-B near1Hz. This acquisition/interpolation difference can shift residual and NLL independently of vehicle identity.',
        '- Type/state/domain differences remain even with identical single-agent context. An autoregressive transition score can reward steady motion despite an unfamiliar state. No state-plausibility or fallback score was added.',
        '', 'Run receipt is in acceptance.json; individual scores are in token_scores.csv.', '']
    (output/'summary.md').write_text('\n'.join(lines))
    receipt.update(summary=rows)
    (output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('outputs/air_pre_bc/checkpoints/epoch_016.ckpt'))
    parser.add_argument('--motion-dict',type=Path,default=Path('assets/motion_dict_K2048_s6.pkl'))
    parser.add_argument('--drone',type=Path,default=DRONE_PATH)
    parser.add_argument('--processed-dir',type=Path,default=Path('data/processed'))
    parser.add_argument('--split-dir',type=Path,default=Path('data/splits'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/anomaly'))
    parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if args.batch_size<1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(args.threads)
    started=time.monotonic()
    protected_before=protected_hashes()
    input_paths=[args.checkpoint,args.motion_dict,args.drone]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==16
    hparams=dict(checkpoint['hyper_parameters']);hparams['motion_dict']=str(args.motion_dict.resolve())
    model=AirSMART(**hparams);model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval().requires_grad_(False).to(args.device)
    # The checkpoint also contains tokenizer buffers: assert the requested file
    # really is the dictionary used for inference, rather than silently override it.
    from src.tokens.token_processor import AirTokenProcessor
    expected=AirTokenProcessor(args.motion_dict)
    for key in ['traj','dz','cluster_centers']:
        assert torch.equal(getattr(model.token_processor,key).cpu(),getattr(expected,key))
    vocab=json.loads((args.split_dir/'type_vocab.json').read_text())
    drone,info=load_drone(args.drone)
    records=list(iter_scores(model,drone,'ord_drone_2026-09-28_1800z_3hz','drone',vocab,args.batch_size,args.device))
    scenarios=(args.split_dir/'val_manifest.txt').read_text().splitlines()
    date_counts={}
    for i,name in enumerate(scenarios):
        with np.load(args.processed_dir/f'{name}.npz',allow_pickle=False) as arrays:
            data={key:arrays[key] for key in arrays.files}
        date=str(data['date'])
        if date not in ['2026-10-03','2026-10-04']:
            raise ValueError(f'Non-validation date: {date}')
        records.extend(iter_scores(model,data,name,'aircraft',vocab,args.batch_size,args.device))
        date_counts[date]=date_counts.get(date,0)+1
        if (i+1)%10==0 or i+1==len(scenarios):
            print(json.dumps(dict(scenarios=i+1,total=len(scenarios),tokens=len(records),elapsed_s=round(time.monotonic()-started,1))),flush=True)
    assert protected_hashes()==protected_before
    assert hashes=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    assert date_counts=={'2026-10-03':91,'2026-10-04':91}
    receipt=dict(milestone='M8',device=args.device,checkpoint_epoch=16,sha256=hashes,drone_interpolation=info,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 scenarios_by_date=date_counts,scored_tokens=len(records),window_frames=TOTAL,stride_frames=SHIFT,
                 history_s=6,future_token_s=2,nll_node=3,gt_label_node=4,single_agent=True,training=False,
                 protected_sha256=protected_before,protected_artifacts_unchanged=True,
                 elapsed_s=time.monotonic()-started)
    rows=write_outputs(records,args.output_dir,receipt)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
