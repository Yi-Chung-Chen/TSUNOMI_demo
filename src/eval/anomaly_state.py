"""M8b: fixed train-only location-conditioned histogram state density."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.eval import demo_gif as visual
from src.eval.anomaly_score import auroc, load_drone, DRONE_PATH, write_csv
from scripts.merge_demo_val import protected_hashes

CELL_M = 250.
ALPHA = .5
HEIGHT_EDGES = np.array([15.,100.,300.,600.])
MERGED_HEIGHT_EDGES = np.array([100.,300.,600.])
SPEED_EDGES = np.array([5.,10.,*range(20,121,10)])
METRICS = ['transition_nll','location_height_nll','speed_nll','course_nll','state_nll','combined_score']
HEIGHT_LABELS = ['<15 m','15–100 m','100–300 m','300–600 m','≥600 m']


def smooth_bins(counts,circular):
    """Fixed mass-preserving binomial kernel; no fitted bandwidth or centres."""
    values=np.asarray(counts,dtype=np.float64)
    if circular:
        left,right=np.roll(values,1,axis=-1),np.roll(values,-1,axis=-1)
    else:
        left=np.concatenate((values[...,:1],values[...,:-1]),axis=-1)
        right=np.concatenate((values[...,1:],values[...,-1:]),axis=-1)
    return .25*left+.5*values+.25*right


def enu_bounds(bounds):
    """Same local tangent plane as the M7 basemap; no extra z conversion."""
    lat0,lon0 = visual.ORIGIN
    x = np.radians(np.array([bounds['west'],bounds['east']])-lon0)*visual.RADIUS*np.cos(np.radians(lat0))
    y = np.radians(np.array([bounds['south'],bounds['north']])-lat0)*visual.RADIUS
    return np.stack((x,y),axis=-1)


def states_at(data, agents, frames):
    """Positions/heights already metres, z is height above166m reference."""
    agents,frames = np.asarray(agents,dtype=int),np.asarray(frames,dtype=int)
    if np.any(frames<6) or np.any(frames>=data['valid'].shape[1]):
        raise ValueError('State requires two observed endpoints separated by2s')
    if not (data['valid'][agents,frames] & data['valid'][agents,frames-6]).all():
        raise ValueError('Invalid state endpoint')
    position = data['pos'][agents,frames].astype(np.float64)
    delta = position[:,:2]-data['pos'][agents,frames-6,:2].astype(np.float64)
    velocity = delta/2.
    return dict(position=position,speed=np.linalg.norm(velocity,axis=-1),course=np.arctan2(delta[:,1],delta[:,0]))


class StateDensity:
    def __init__(self,bounds,bin_smoothing=True,height_edges=None):
        self.bounds = np.asarray(bounds,dtype=float)
        self.bin_smoothing=bin_smoothing
        self.height_edges=np.array(HEIGHT_EDGES if height_edges is None else height_edges,dtype=float)
        if np.any(np.diff(self.height_edges)<=0):
            raise ValueError('Height edges must increase')
        self.n_height=len(self.height_edges)+1
        self.nx,self.ny = np.ceil((self.bounds[1]-self.bounds[0])/CELL_M).astype(int)
        self.n_cells = self.nx*self.ny+1
        self.joint = np.zeros((self.n_cells,self.n_height),dtype=np.int64)
        self.speed = np.zeros((self.n_cells,self.n_height,14),dtype=np.int64)
        self.course = np.zeros((self.n_cells,self.n_height,12),dtype=np.int64)
        self.eligible_count = 0

    def bins(self,states):
        xy = states['position'][:,:2]
        inside = ((xy>=self.bounds[0]) & (xy<=self.bounds[1])).all(-1)
        indices = np.floor((xy-self.bounds[0])/CELL_M).astype(int)
        x = np.clip(indices[:,0],0,self.nx-1);y = np.clip(indices[:,1],0,self.ny-1)
        cell = np.where(inside,y*self.nx+x,self.n_cells-1)
        height = np.searchsorted(self.height_edges,states['position'][:,2],side='right')
        speed = np.searchsorted(SPEED_EDGES,states['speed'],side='right')
        course = np.floor((states['course']+np.pi)%(2*np.pi)/(2*np.pi)*12).astype(int)
        return cell,height,speed,np.minimum(course,11)

    def add(self,states):
        cell,height,speed,course = self.bins(states)
        key = cell*self.n_height+height
        self.joint += np.bincount(key,minlength=self.joint.size).reshape(self.joint.shape)
        self.speed += np.bincount(key*14+speed,minlength=self.speed.size).reshape(self.speed.shape)
        self.course += np.bincount(key*12+course,minlength=self.course.size).reshape(self.course.shape)
        self.eligible_count += len(cell)

    def pool(self,counts):
        grid = counts[:-1].reshape(self.ny,self.nx,*counts.shape[1:])
        padded = np.pad(grid,[(1,1),(1,1)]+[(0,0)]*(grid.ndim-2))
        pooled = sum(padded[y:y+self.ny,x:x+self.nx] for y in range(3) for x in range(3))
        # The outside cell is a separate bucket, never a neighbour of every edge.
        return np.concatenate((pooled.reshape(-1,*counts.shape[1:]),counts[-1:]),axis=0)

    def finish(self):
        assert self.joint.sum()==self.speed.sum()==self.course.sum()==self.eligible_count
        assert np.array_equal(self.speed.sum(-1),self.joint)
        assert np.array_equal(self.course.sum(-1),self.joint)
        self.pooled_joint = self.pool(self.joint)
        self.pooled_speed,self.pooled_course = self.pool(self.speed),self.pool(self.course)
        assert np.array_equal(self.pooled_speed.sum(-1),self.pooled_joint)
        assert np.array_equal(self.pooled_course.sum(-1),self.pooled_joint)
        self.smoothed_speed=smooth_bins(self.pooled_speed,False) if self.bin_smoothing else self.pooled_speed.astype(float)
        self.smoothed_course=smooth_bins(self.pooled_course,True) if self.bin_smoothing else self.pooled_course.astype(float)
        assert np.allclose(self.smoothed_speed.sum(-1),self.pooled_joint)
        assert np.allclose(self.smoothed_course.sum(-1),self.pooled_joint)
        self.p_joint = (self.pooled_joint+ALPHA)/(self.pooled_joint.sum()+ALPHA*self.joint.size)
        self.p_speed = (self.smoothed_speed+ALPHA)/(self.pooled_joint[...,None]+ALPHA*14)
        self.p_course = (self.smoothed_course+ALPHA)/(self.pooled_joint[...,None]+ALPHA*12)
        assert np.isclose(self.p_joint.sum(),1.)
        assert np.allclose(self.p_speed.sum(-1),1.) and np.allclose(self.p_course.sum(-1),1.)
        return self

    def score(self,states):
        cell,height,speed,course = self.bins(states)
        location_nll = -np.log(self.p_joint[cell,height])
        speed_nll = -np.log(self.p_speed[cell,height,speed])
        course_nll = -np.log(self.p_course[cell,height,course])
        return dict(cell=cell,height_bin=height,speed_bin=speed,course_bin=course,
            location_height_nll=location_nll,speed_nll=speed_nll,course_nll=course_nll,
            state_nll=location_nll+speed_nll+course_nll)


def load_processed(path):
    with np.load(path,allow_pickle=False) as arrays:
        return {key:arrays[key] for key in arrays.files}


def fit_density(root,bin_smoothing=True,height_edges=None):
    names = (root/'data/splits/train_manifest.txt').read_text().splitlines()
    model,expected_bounds = None,None
    receipt = dict(scenarios=len(names),dates={},eligible_by_scenario={},raw_eligible_states=0)
    for i,name in enumerate(names):
        data = load_processed(root/'data/processed'/f'{name}.npz')
        date = str(data['date'])
        if date not in ['2026-09-28','2026-09-29','2026-09-30','2026-10-01','2026-10-02']:
            raise ValueError(f'Non-train date in density fit: {date}')
        bounds = json.loads(str(data['bounds_json']))
        if model is None:
            expected_bounds = bounds;model = StateDensity(enu_bounds(bounds),bin_smoothing,height_edges)
        assert bounds==expected_bounds
        # Plan says endpoints, not every intervening frame: all valid3Hz states.
        agents,frames = np.nonzero(data['valid'][:,6:] & data['valid'][:,:-6])
        states = states_at(data,agents,frames+6)
        model.add(states)
        receipt['dates'][date] = receipt['dates'].get(date,0)+1
        receipt['eligible_by_scenario'][name] = len(frames)
        receipt['raw_eligible_states'] += len(frames)
        if (i+1)%100==0 or i+1==len(names):
            print(json.dumps(dict(train_scenarios=i+1,total=len(names),eligible_states=receipt['raw_eligible_states'])),flush=True)
    assert model.eligible_count==receipt['raw_eligible_states']
    receipt.update(bounds_latlon=expected_bounds,bounds_enu_m=model.bounds.tolist(),grid=[int(model.nx),int(model.ny)],
        cells_including_outside=int(model.n_cells),cell_size_m=CELL_M,alpha=ALPHA,
        height_edges_m=model.height_edges.tolist(),height_bins=model.n_height,speed_edges_mps=SPEED_EDGES.tolist(),course_bins=12,
        train_manifest_sha256=hashlib.sha256((root/'data/splits/train_manifest.txt').read_bytes()).hexdigest(),
        bin_smoothing=dict(enabled=bin_smoothing,kernel=[.25,.5,.25],course='circular',speed='reflecting',
            coordinate='adjacent bin index; no physical bandwidth',order='spatial pooling, bin kernel, alpha, normalize'))
    return model.finish(),receipt


def score_m8_rows(root,model):
    with (root/'outputs/anomaly/token_scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        for key in ['window_start_frame','token_start_frame','chain_token','residual_token']:
            row[key] = int(row[key])
        for key in ['nll','token_start_s','residual_horizontal_m','residual_vertical_m']:
            row[key] = float(row[key])
        row['on_ground'] = row['on_ground']=='True'
        grouped.setdefault((row['source'],row['scenario']),[]).append(row)
    drone,drone_info = load_drone()
    for (source,scenario),items in grouped.items():
        data = drone if source=='drone' else load_processed(root/'data/processed'/f'{scenario}.npz')
        ids = {str(agent):i for i,agent in enumerate(data['agent_id'])}
        agents = np.array([ids[row['agent_id']] for row in items])
        frames = np.array([row['token_start_frame'] for row in items])
        states = states_at(data,agents,frames)
        result = model.score(states)
        for i,row in enumerate(items):
            row.update({key:values[i].item() for key,values in result.items()})
            row.update(x_m=float(states['position'][i,0]),y_m=float(states['position'][i,1]),
                height_m=float(states['position'][i,2]),speed_mps=float(states['speed'][i]),
                course_rad=float(states['course'][i]),transition_nll=row['nll'])
    return rows,drone,drone_info


def standardize(rows):
    """Only10-03 aircraft, all ground states, determine each fixed median/IQR."""
    reference = [row for row in rows if row['source']=='aircraft' and row['date']=='2026-10-03']
    if not reference:
        raise ValueError('No10-03 aircraft reference tokens')
    result = dict(reference_date='2026-10-03',source='aircraft',ground='both',count=len(reference),scores={})
    for metric,key in [('transition_nll','z_transition'),('state_nll','z_state')]:
        values = np.array([row[metric] for row in reference])
        q25,median,q75 = np.percentile(values,[25,50,75]);iqr=q75-q25
        if not np.isfinite(iqr) or iqr<=0:
            raise ValueError(f'Zero/invalid reference IQR for {metric}; no fallback is specified')
        result['scores'][metric] = dict(median=float(median),q25=float(q25),q75=float(q75),iqr=float(iqr))
        for row in rows:
            row[key] = float((row[metric]-median)/iqr)
    for row in rows:
        row['combined_score'] = max(row['z_transition'],row['z_state'])
    return result


def grouped_rows(rows):
    result = [('drone',[r for r in rows if r['source']=='drone'])]
    for date in ['2026-10-04','2026-10-03']:
        for ground in [False,True]:
            result.append((f'aircraft-{"ground" if ground else "airborne"} ({date})',
                [r for r in rows if r['source']=='aircraft' and r['date']==date and r['on_ground']==ground]))
    return result


def summarize(rows):
    groups = grouped_rows(rows);drone=groups[0][1];air=groups[1][1]
    result = []
    for metric in METRICS:
        positives = np.array([r[metric] for r in drone])
        air_p95 = float(np.percentile([r[metric] for r in air],95))
        fraction = float(np.mean(positives>air_p95))
        for group,items in groups:
            values = np.array([r[metric] for r in items])
            result.append(dict(score=metric,group=group,token_count=len(values),mean=float(values.mean()),
                median=float(np.median(values)),p95=float(np.percentile(values,95)),
                auroc_drone_positive=None if group=='drone' else auroc(positives,values),
                airborne_10_04_p95=air_p95,drone_fraction_above_airborne_10_04_p95=fraction))
    return result


def plot_density(model,drone,cache,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    # The final row/column can be partial at the metadata boundary.
    x = np.minimum(model.bounds[0,0]+np.arange(model.nx+1)*CELL_M,model.bounds[1,0])
    y = np.minimum(model.bounds[0,1]+np.arange(model.ny+1)*CELL_M,model.bounds[1,1])
    xx,yy = np.meshgrid(x,y);xy=np.stack((xx,yy),axis=-1)
    image,zoom,pb = visual.basemap(model.bounds,cache,'light')
    pixels=visual.enu_to_pixel(xy,zoom)
    counts=model.joint[:-1].reshape(model.ny,model.nx,model.n_height)
    vmax=float(np.log1p(counts.max()))
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    gt=drone['pos'][0];valid=drone['valid'][0]
    hbin=np.searchsorted(model.height_edges,gt[:,2],side='right')
    height_labels=[f'<{model.height_edges[0]:g} m']+[f'{a:g}–{b:g} m' for a,b in zip(model.height_edges[:-1],model.height_edges[1:])]+[f'≥{model.height_edges[-1]:g} m']
    for height,ax in enumerate(axes.flat[:model.n_height]):
        ax.imshow(image,extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]))
        values=np.ma.masked_where(counts[:,:,height]==0,np.log1p(counts[:,:,height]))
        heat=ax.pcolormesh(pixels[:,:,0],pixels[:,:,1],values,cmap='magma',norm=Normalize(0,vmax),alpha=.72,shading='flat',rasterized=True)
        path=visual.enu_to_pixel(gt[:,:2],zoom).copy()
        path[~(valid & (hbin==height))]=np.nan
        if np.isfinite(path).any():
            ax.plot(*path.T,color='#00a4b8',lw=2.3,label='Drone GT path')
            ax.legend(loc='upper right',fontsize=9)
        ax.set(xlim=(pb[0,0],pb[1,0]),ylim=(pb[1,1],pb[0,1]),aspect='equal',title=f'Height {height_labels[height]}')
        ax.axis('off')
        ax.text(.01,.015,'© OpenStreetMap contributors',transform=ax.transAxes,fontsize=6,color='#4b5563')
    for ax in axes.flat[model.n_height:]:
        ax.axis('off')
    axes.flat[model.n_height].text(.06,.8,'Train aircraft only\n09-28 … 10-02\n250 m cells\nRaw eligible 3 Hz frame counts\n\nZero-count cells show the basemap.\nCyan: drone path in its height bin.\nOutside bucket is not mapped.',va='top',fontsize=13)
    fig.colorbar(heat,ax=list(axes.flat[:model.n_height]),shrink=.7,label='log(1 + eligible train aircraft frames)')
    fig.suptitle('Normal traffic by location and height · no drone counts used',fontsize=17)
    fig.savefig(destination,dpi=150);plt.close(fig)


def plot_timeline(rows,summary,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    drone=grouped_rows(rows)[0][1]
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    for ax,metric,color in zip(axes,['state_nll','transition_nll','combined_score'],['#c76025','#287ca4','#833fa0']):
        ref=next(r for r in summary if r['score']==metric and r['group']=='aircraft-airborne (2026-10-04)')
        ax.plot([r['token_start_s'] for r in drone],[r[metric] for r in drone],color=color,lw=1.5,label='Drone')
        ax.axhline(ref['p95'],color='#555',ls='--',label=f'10-04 airborne p95 = {ref["p95"]:.2f}')
        ax.set_ylabel(metric+(' (z units)' if metric=='combined_score' else ' (nats)'))
        ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Scenario time at token start (s)')
    fig.suptitle('State at token start; transition observes the following2s · fixed10-03 standardisation')
    fig.savefig(destination,dpi=150);plt.close(fig)


def write_report(summary,reference,fit,output):
    columns=list(summary[0]);lines=['# M8b: location-conditioned state density','',
        '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    def cell(value):
        return '—' if value is None else f'{value:.4f}' if isinstance(value,float) else str(value)
    lines += ['| '+' | '.join(cell(row[k]) for k in columns)+' |' for row in summary]
    main={row['score']:row for row in summary if row['group']=='aircraft-airborne (2026-10-04)'}
    components=['location_height_nll','speed_nll','course_nll']
    # Describe the largest mean contribution to the drone-vs-airborne excess.
    drone={row['score']:row for row in summary if row['group']=='drone'}
    differences={key:drone[key]['mean']-main[key]['mean'] for key in components}
    driver=max(differences,key=differences.get)
    lines += ['', '## Plain-language conclusion','',
        f'State NLL AUROC is {main["state_nll"]["auroc_drone_positive"]:.5f}, compared with transition NLL '
        f'{main["transition_nll"]["auroc_drone_positive"]:.5f}; combined AUROC is {main["combined_score"]["auroc_drone_positive"]:.5f}. '
        f'{100*main["state_nll"]["drone_fraction_above_airborne_10_04_p95"]:.1f}% of drone tokens exceed the10-04 airborne state-NLL p95, '
        f'versus {100*main["transition_nll"]["drone_fraction_above_airborne_10_04_p95"]:.1f}% for transition NLL and '
        f'{100*main["combined_score"]["drone_fraction_above_airborne_10_04_p95"]:.1f}% for combined.',
        '',('Location-conditioned state density flags this route much more consistently than transition NLL, which misses most of its steady flight.'
            if main['state_nll']['drone_fraction_above_airborne_10_04_p95']>main['transition_nll']['drone_fraction_above_airborne_10_04_p95']
            else 'Location-conditioned state density does not flag more of this route than transition NLL at the stated reference levels.'),
        '',('Combined does not improve the AUROC over state density alone in this sample.'
            if abs(main['combined_score']['auroc_drone_positive']-main['state_nll']['auroc_drone_positive'])<1e-5
            else 'Compare state and combined AUROC above; combining scores is not guaranteed to improve ranking.'),
        '',f'The largest mean component excess over airborne aircraft is {driver}: {differences[driver]:.3f} nats. '
        'Component mean differences (drone minus10-04 airborne): '+', '.join(f'{key} {value:+.3f}' for key,value in differences.items())+'.',
        '', '**This is one drone and one route: a demonstration, not a validated detector.** '
        'Neighbouring tokens are correlated and source interpolation differs between simulated drone poses and ADS-B aircraft. '
        '10-04 p95 is a descriptive evaluation reference, not an independently validated deployment threshold.',
        '', '## Fixed method and provenance','',
        f'- Histograms fit only {fit["raw_eligible_states"]:,} eligible3Hz states from {fit["scenarios"]} train scenarios (09-28..10-02). '
        'Each state uses valid endpoints t−6 and t; intervening frames are not required by the plan. No validation/drone counts enter the density.',
        '- Processed aircraft z and the existing interpolated drone z are already height above166m; no second altitude conversion. Speed/course use past2s displacement only. atan2(0,0)=0 for stationary course.',
        '- Metadata geographic bounds use the M7 fixed-origin tangent-plane conversion.250m grid anchored at west/south ENU bounds; final cells are partial. Exact upper bounds are included in the final cells; all other out-of-bounds states use one separate outside bucket.',
        '- Pool raw joint and conditional counts over3×3 spatial neighbours at fixed height with zero padding; outside pools only itself. Add alpha0.5 to every bin. Joint normalises over all(cell,height) including outside; speed/course normalise conditionally.',
        ('- User-authorized adjacent-bin smoothing: fixed[1,2,1]/4 on pooled histograms before alpha, circular for course and reflecting for speed. Raw/joint/height counts are unchanged. This is by bin index, not an equal physical m/s bandwidth; no125m/s tail representative is introduced. Hard bin lookup remains, so residual boundary steps can still occur.'
         if fit.get('bin_smoothing',{}).get('enabled') else '- No adjacent speed/course smoothing in this legacy run.'),
        '- State NLL = location_height_nll + speed_nll + course_nll. Height boundaries'+ '/'.join(f'{x:g}' for x in fit['height_edges_m'])+'m; speed boundaries5/10/20/…/120m/s (14bins);12course bins begin at−pi.',
        f'- Standardisation fits only {reference["count"]:,}10-03 aircraft tokens, airborne+ground. Each score uses its own median/IQR; combined=max(z_transition,z_state). Parameters are in standardization.json. A zero IQR raises an error; no fallback/tuning.',
        '- Scores join the exact M8 token rows at token_start_frame. State is causal at the start; transition uses the following2s GT and is available at token end. Combined is not an online score at token start.',
        '- AUROC treats drone as positive and higher scores as anomalous, with average ranks for ties. Exceedance is strict > the separately computed10-04-airborne p95 of each metric.',
        '- Optional combined-score GIFs use that same10-04 airborne combined p95 for the red outline; the labels show combined z-units, not NLL nats. State is known at token start, but the combined label appears only once its transition token completes.',
        '- Original bins/spatial smoothing/combination follow the plan. The adjacent-bin correction was requested by Yi-Chung and fixed provisionally by the advisor before recomputing results; see QUESTIONS.md. No NN training/model changes, package installs or drone tuning.',
        ('- Height100 revision merges only the first two raw height bins before smoothing/pseudocounts.10-04 and the shown aircraft cases informed this revision, so10-04 is inspected development data and must not be presented as a pristine holdout. A future independent dataset is needed for validation.' if fit['height_bins']==4 else '- Five original height bins retained.'),
        '', 'Detailed counts/probabilities: state_density.npz; all token joins: token_scores.csv; run checks: acceptance.json.', '']
    (output/'summary.md').write_text('\n'.join(lines))


def write_combined_gifs(root,rows,summary,output):
    """Reuse the GT renderer; threshold is the specified evaluation p95."""
    from PIL import Image
    from src.eval.anomaly_gif import render_scene
    from src.eval.load_rollout import load_rollout
    destination=output/'gifs';destination.mkdir(parents=True,exist_ok=True)
    threshold=next(r['p95'] for r in summary if r['score']=='combined_score' and r['group']=='aircraft-airborne (2026-10-04)')
    tracks={}
    for row in rows:
        key=(row['source'],row['scenario'],row['agent_id'])
        start=int(row['token_start_frame'])
        # The shared renderer's internal scalar key is nll; the displayed label
        # and manifest explicitly identify the standardised combined score.
        tracks.setdefault(key,[]).append(dict(start=start,end=start+6,nll=float(row['combined_score'])))
    for values in tracks.values():
        values.sort(key=lambda r:r['end'])
    drone,_=load_drone()
    scenes=[dict(source='drone',scenario='ord_drone_2026-09-28_1800z_3hz',date=str(drone['date']),start_frame=0,
        gt_position=drone['pos'],gt_heading=drone['heading'],valid_mask=drone['valid'],agent_id=drone['agent_id'])]
    for path in visual.select_windows(root):
        data=load_rollout(path);data['source']='aircraft';scenes.append(data)
    records,thumbs=[],[]
    for data in scenes:
        frames,record=render_scene(data,tracks,root/'data/basemap_cache',threshold,score_label='combined')
        name='drone01_full' if data['source']=='drone' else f'{data["scenario"]}_f{int(data["start_frame"]):04d}'
        path=destination/f'{name}.gif'
        visual.save_gif(frames,path,1);skip=1
        if path.stat().st_size>8_000_000:
            frames=frames[::2];skip=2;visual.save_gif(frames,path,skip)
        assert path.stat().st_size<=8_000_000
        with Image.open(path) as decoded:
            decoded.seek(decoded.n_frames-1);thumb=decoded.convert('RGB');thumb.thumbnail((480,360));thumbs.append(thumb)
        record.update(output=path.name,bytes=path.stat().st_size,encoded_frames=len(frames),skip=skip)
        records.append(record);print(json.dumps(dict(combined_gif=path.name,bytes=record['bytes'])),flush=True)
    sheet=Image.new('RGB',(1920,1080),visual.THEMES['dark']['background'])
    for i,thumb in enumerate(thumbs):
        sheet.paste(thumb,((i%4)*480,(i//4)*360))
    sheet.save(destination/'contact_sheet.png')
    result=dict(score='max(z_transition,z_state)',threshold=threshold,threshold_source='10-04 airborne combined p95',
        temporal_display='state at token start; combined appears only when the transition token completes',gifs=records)
    (destination/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def write_smoothing_comparison(root,rows,summary,reference,output,height100=False):
    baseline=root/('outputs/anomaly_state_smoothed' if height100 else 'outputs/anomaly_state')
    prefix='height_comparison' if height100 else 'smoothing_comparison'
    with (baseline/'token_scores.csv').open() as stream:
        old=list(csv.DictReader(stream))
    old_reference=json.loads((baseline/'standardization.json').read_text())
    def key(row):
        return row['source'],row['scenario'],row['agent_id'],int(row['token_start_frame'])
    assert [key(r) for r in old]==[key(r) for r in rows]
    for row in old:
        row['on_ground']=row['on_ground']=='True'
        for metric in METRICS:
            row[metric]=float(row[metric])
    old_lookup={key(r):r for r in old}
    groups,old_groups=grouped_rows(rows),dict(grouped_rows(old))
    air_old=old_groups['aircraft-airborne (2026-10-04)']
    def frozen_combined(row):
        z=[]
        for metric in ['transition_nll','state_nll']:
            params=old_reference['scores'][metric]
            z.append((row[metric]-params['median'])/params['iqr'])
        return max(z)
    results=[]
    for metric in METRICS:
        before_cut=float(np.percentile([r[metric] for r in air_old],95))
        after_cut=next(r['p95'] for r in summary if r['score']==metric and r['group']=='aircraft-airborne (2026-10-04)')
        for name,items in groups:
            before=np.array([old_lookup[key(r)][metric] for r in items])
            after=np.array([r[metric] for r in items])
            frozen=np.array([frozen_combined(r) for r in items]) if metric=='combined_score' else after
            results.append(dict(score=metric,group=name,token_count=len(items),old_cutoff=before_cut,new_cutoff=after_cut,
                old_flagged=int((before>before_cut).sum()),old_fraction=float(np.mean(before>before_cut)),
                after_frozen_calibration_flagged=int((frozen>before_cut).sum()),after_frozen_calibration_fraction=float(np.mean(frozen>before_cut)),
                after_recalibration_flagged=int((after>after_cut).sum()),after_recalibration_fraction=float(np.mean(after>after_cut))))
    write_csv(output/f'{prefix}.csv',results)
    case=[]
    old_cut=next(r['old_cutoff'] for r in results if r['score']=='combined_score')
    new_cut=next(r['new_cutoff'] for r in results if r['score']=='combined_score')
    for row in rows:
        if row['scenario']!='scenario_000536_aircraft' or row['agent_id']!='aa0300' or not 24<=row['token_start_s']<=66:
            continue
        previous=old_lookup[key(row)]
        case.append(dict(token_start_s=row['token_start_s'],display_at_s=row['token_start_s']+2,
            course_deg=float(np.degrees(row['course_rad'])),speed_mps=row['speed_mps'],
            old_course_nll=previous['course_nll'],new_course_nll=row['course_nll'],
            old_speed_nll=previous['speed_nll'],new_speed_nll=row['speed_nll'],
            old_state_nll=previous['state_nll'],new_state_nll=row['state_nll'],
            old_combined=previous['combined_score'],new_combined_old_reference=frozen_combined(row),
            new_combined=row['combined_score'],old_flag=previous['combined_score']>old_cut,
            new_frozen_flag=frozen_combined(row)>old_cut,new_flag=row['combined_score']>new_cut))
    write_csv(output/'aa0300_comparison.csv',case)
    target=next(r for r in case if r['token_start_s']==28.)
    compare=dict(baseline=str(baseline.relative_to(root)),old_reference=old_reference,new_reference=reference,
        combined_old_cutoff=old_cut,combined_new_cutoff=new_cut,aa0300_at_28s=target,groups=results)
    (output/f'{prefix}.json').write_text(json.dumps(compare,indent=2)+'\n')
    with (output/'summary.md').open('a') as stream:
        stream.write('\n## Height <100m merge: before / after\n\n' if height100 else '\n## Smoothing correction: before / after\n\n')
        stream.write(('First two raw height bins merged before pooling and alpha. Stationary course remains atan2(0,0)=0; those counts now enter the <100m layer. ' if height100 else '')+'Fixed adjacent-bin[1,2,1]/4 kernel: circular course, reflecting speed, then alpha. '
            'No physical-distance bandwidth or drone-dependent parameter. '
            'Original outputs remain untouched. Hard bin lookup still permits residual steps.\n\n')
        stream.write(f'aa0300 token28–30s: combined {target["old_combined"]:.4f} → {target["new_combined"]:.4f}; '
            f'old cutoff {old_cut:.4f}, recalibrated cutoff {new_cut:.4f}; '
            f'flagged before/after = {target["old_flag"]}/{target["new_flag"]}. '
            f'Course NLL {target["old_course_nll"]:.4f} → {target["new_course_nll"]:.4f}; '
            f'speed NLL {target["old_speed_nll"]:.4f} → {target["new_speed_nll"]:.4f}.\n\n')
        stream.write('| Group | Before combined flags | After, frozen old calibration/cutoff | After, recalibrated |\n|---|---:|---:|---:|\n')
        for result in results:
            if result['score']=='combined_score':
                stream.write(f'| {result["group"]} | {result["old_flagged"]}/{result["token_count"]} ({result["old_fraction"]:.2%}) '
                    f'| {result["after_frozen_calibration_flagged"]} ({result["after_frozen_calibration_fraction"]:.2%}) '
                    f'| {result["after_recalibration_flagged"]} ({result["after_recalibration_fraction"]:.2%}) |\n')
        stream.write('\nRecalibrated10-04 airborne exceedance is approximately5% by construction (strict ties can make it slightly lower); that is not evidence of an improved false-positive rate. '
            'Frozen combined comparisons use both the old10-03 medians/IQRs and old cutoff, rather than mixing calibration scales. '
            'Normal-aircraft flags are false alerts under the clean-aircraft assumption. The corrected example is not a claim that all false alerts disappear. '
            f'Full per-component comparisons and case timeline are in {prefix}.csv and aa0300_comparison.csv.\n')
    if height100:
        write_height_diagnostics(rows,old,old_lookup,key,frozen_combined,old_cut,new_cut,output)
    return compare


def alert_runs(rows,flags):
    """Consecutive flagged 2s tokens; missing/unflagged observations break runs."""
    tracks={}
    for row,flag in zip(rows,flags):
        tracks.setdefault((row['scenario'],row['agent_id']),[]).append((int(row['token_start_frame']),bool(flag)))
    durations=[];long_tracks=0
    for values in tracks.values():
        current=0;previous=None;lengths=[]
        for frame,flag in sorted(values):
            if not flag or previous is None or frame-previous!=6:
                if current: lengths.append(current)
                current=0
            if flag: current+=1
            previous=frame
        if current: lengths.append(current)
        durations.extend(2*n for n in lengths)
        long_tracks+=any(n>=3 for n in lengths)
    return dict(track_count=len(tracks),flagged_tokens=int(sum(flags)),run_count=len(durations),
        max_run_s=max(durations,default=0),median_run_s=float(np.median(durations)) if durations else 0.,
        tracks_with_run_ge6s=long_tracks,fraction_tracks_run_ge6s=long_tracks/len(tracks) if tracks else 0.)


def write_height_diagnostics(rows,old,old_lookup,key,frozen,old_cut,new_cut,output):
    runs=[]
    groups=grouped_rows(rows)+[(f'aircraft-all ({date})',[r for r in rows if r['source']=='aircraft' and r['date']==date]) for date in ['2026-10-03','2026-10-04']]
    for name,items in groups:
        for variant,flags in [
            ('before',[old_lookup[key(r)]['combined_score']>old_cut for r in items]),
            ('after_frozen',[frozen(r)>old_cut for r in items]),
            ('after_recalibrated',[r['combined_score']>new_cut for r in items])]:
            runs.append(dict(group=name,variant=variant,token_count=len(items),**alert_runs(items,flags)))
    write_csv(output/'alert_runs.csv',runs)
    cases=[]
    for row in rows:
        if (row['scenario'],row['agent_id']) not in [('scenario_000667_aircraft','aabc11'),('scenario_000536_aircraft','aa0300')]: continue
        previous=old_lookup[key(row)]
        cases.append(dict(scenario=row['scenario'],agent_id=row['agent_id'],token_start_s=row['token_start_s'],
            height_m=row['height_m'],speed_mps=row['speed_mps'],on_ground=row['on_ground'],
            old_height_bin=previous['height_bin'],new_height_bin=row['height_bin'],
            old_state_nll=previous['state_nll'],new_state_nll=row['state_nll'],
            old_combined=previous['combined_score'],new_combined_frozen=frozen(row),new_combined=row['combined_score'],
            old_flag=previous['combined_score']>old_cut,frozen_flag=frozen(row)>old_cut,new_flag=row['combined_score']>new_cut))
    write_csv(output/'aircraft_cases.csv',cases)
    drone=[]
    for label,lo,hi in [('<100',-np.inf,100),('100–300',100,300),('300–600',300,600),('>=600',600,np.inf)]:
        items=[r for r in rows if r['source']=='drone' and lo<=r['height_m']<hi]
        drone.append(dict(height_bin=label,token_count=len(items),
            old_median_state_nll=float(np.median([old_lookup[key(r)]['state_nll'] for r in items])) if items else None,
            new_median_state_nll=float(np.median([r['state_nll'] for r in items])) if items else None,
            new_combined_flagged=sum(r['combined_score']>new_cut for r in items)))
    write_csv(output/'drone_by_height.csv',drone)
    with (output/'summary.md').open('a') as stream:
        stream.write('\n## Consecutive alerts\n\nRuns group by (scenario, agent); missing/unflagged tokens break runs, and starts must differ by exactly 6 frames. Duration = 2s × token count. Ground/airborne subgroup runs break at phase changes; aircraft-all includes them. These are scenario-agent tracks, not independent flights.\n\n')
        stream.write('| Group | Variant | Flags | Max run s | Median run s | Tracks with ≥6s run |\n|---|---|---:|---:|---:|---:|\n')
        for item in runs:
            stream.write(f'| {item["group"]} | {item["variant"]} | {item["flagged_tokens"]}/{item["token_count"]} | {item["max_run_s"]} | {item["median_run_s"]:g} | {item["tracks_with_run_ge6s"]}/{item["track_count"]} ({item["fraction_tracks_run_ge6s"]:.2%}) |\n')
        stream.write('\nDrone height breakdown (same physical height groups before/after):\n\n| Height m | Tokens | Before median state NLL | After median state NLL | After combined flags |\n|---|---:|---:|---:|---:|\n')
        for item in drone:
            stream.write(f'| {item["height_bin"]} | {item["token_count"]} | {item["old_median_state_nll"]} | {item["new_median_state_nll"]} | {item["new_combined_flagged"]} |\n')
        stream.write('\nThe displayed aabc11 (scenario000667) and aa0300 (scenario000536) cases are dated10-03.10-04 aggregate results have also been inspected during revision, so they are development evidence. Full case timelines: aircraft_cases.csv; full group/run comparisons: alert_runs.csv and height_comparison.csv.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--legacy-bins',action='store_true',help='Reproduce original spatial/alpha-only density without adjacent-bin smoothing')
    parser.add_argument('--keep-height-15',action='store_true',help='Retain five height bins for the previous smoothed baseline')
    parser.add_argument('--gifs',action='store_true',help='Also render the11 existing GT examples with combined-score outlines')
    args=parser.parse_args();root=args.root.resolve()
    height100=not (args.legacy_bins or args.keep_height_15)
    default='anomaly_state_height100' if height100 else ('anomaly_state' if args.legacy_bins else 'anomaly_state_smoothed')
    output=root/(args.output_dir or Path('outputs')/default)
    if height100 and output.resolve() in [(root/'outputs'/d).resolve() for d in ['anomaly_state','anomaly_state_smoothed']]:
        parser.error('Preserve both existing baselines; choose a separate height100 directory')
    if not args.legacy_bins and output.resolve()==(root/'outputs/anomaly_state').resolve():
        parser.error('Preserve original outputs/anomaly_state; choose a separate smoothed output directory')
    output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    protected=protected_hashes(root)
    source=root/'outputs/anomaly/token_scores.csv';score_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    model,fit=fit_density(root,bin_smoothing=not args.legacy_bins,height_edges=MERGED_HEIGHT_EDGES if height100 else HEIGHT_EDGES)
    rows,drone,drone_info=score_m8_rows(root,model)
    reference=standardize(rows);summary=summarize(rows)
    write_csv(output/'token_scores.csv',rows);write_csv(output/'summary.csv',summary)
    (output/'standardization.json').write_text(json.dumps(reference,indent=2)+'\n')
    np.savez_compressed(output/'state_density.npz',bounds=model.bounds,height_edges=model.height_edges,joint=model.joint,speed=model.speed,course=model.course,
        pooled_joint=model.pooled_joint,pooled_speed=model.pooled_speed,pooled_course=model.pooled_course,
        smoothed_speed=model.smoothed_speed,smoothed_course=model.smoothed_course,
        p_joint=model.p_joint,p_speed=model.p_speed,p_course=model.p_course)
    plot_density(model,drone,root/'data/basemap_cache',output/'traffic_density.png')
    plot_timeline(rows,summary,output/'drone_timeline_state.png')
    write_report(summary,reference,fit,output)
    comparison=write_smoothing_comparison(root,rows,summary,reference,output,height100=height100) if not args.legacy_bins else None
    gif_receipt=write_combined_gifs(root,rows,summary,output) if args.gifs else None
    assert protected_hashes(root)==protected
    assert score_hash==hashlib.sha256(source.read_bytes()).hexdigest()
    receipt=dict(milestone='M8b',density_fit=fit,standardization=reference,token_rows=len(rows),
        m8_scores_sha256=score_hash,protected_sha256=protected,protected_artifacts_unchanged=True,
        source_scores_unchanged=True,neural_training=False,model_execution=False,device='cpu',
        drone_interpolation=drone_info,elapsed_s=time.monotonic()-started,summary=summary,combined_gifs=gif_receipt,
        smoothing_comparison=comparison)
    (output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(eligible_train_states=fit['raw_eligible_states'],token_rows=len(rows),reference=reference,
        elapsed_s=receipt['elapsed_s'],primary=[r for r in summary if r['group']=='aircraft-airborne (2026-10-04)']),indent=2),flush=True)


if __name__=='__main__':
    main()
