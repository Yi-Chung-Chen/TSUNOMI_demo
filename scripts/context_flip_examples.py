"""Selected clock-only flip illustrations using the frozen context density."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon,FancyBboxPatch
from matplotlib.transforms import Affine2D
from src.eval import demo_gif as visual
from src.eval.anomaly_gif import DRONE
from src.integrated.presentation import cached_basemap
from src.eval.context_density import ContextDensity
from src.eval.anomaly_state import states_at,load_processed

BG='#0b1622';TEXT='#e8eef5';MUTED='#8fa3b8';CYAN='#4cc9f0';GREEN='#80ed99';ORANGE='#ff9f43'
FACTORS=['location_height_nll','speed_nll','course_nll']
FACTOR_NAMES=['Location / height','Speed at this location','Direction at this location']


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def frozen_hashes(root):
    return {str(p.relative_to(root)):digest(p) for name in ['outputs/anomaly_context_time','outputs/integrated_demo','data/splits']
            for p in (root/name).rglob('*') if p.is_file()}


def find_drone(model,cutoff):
    """Deterministic illustrative grid search, NOT an evaluation benchmark.

Limit height to30/75/150m, speed2/7.5/15/25m/s,12course-bin centres, and >=100
pooled train states in BOTH contexts. Maximize the smaller cutoff margin.
"""
    g=model.global_model;values=[]
    for z in [30.,75.,150.]:
        h=np.searchsorted(g.height_edges,z,side='right')
        cells=np.flatnonzero(np.minimum(model.models[0].pooled_joint[:-1,h],model.models[1].pooled_joint[:-1,h])>=100)
        for cell in cells:
            xy=g.bounds[0]+np.array([cell%g.nx+.5,cell//g.nx+.5])*250
            for speed in [2.,7.5,15.,25.]:
                for angle in -np.pi+(np.arange(12)+.5)*np.pi/6:values.append([*xy,z,speed,angle])
    a=np.array(values);s=dict(position=a[:,:3],speed=a[:,3],course=a[:,4])
    day=model.score_context(s,0)['state_nll'];night=model.score_context(s,1)['state_nll']
    eligible=np.flatnonzero((day<=cutoff)&(night>cutoff))
    if not len(eligible):raise RuntimeError('No supported synthetic flip; do not adjust cutoff/model')
    best=max(eligible,key=lambda i:min(cutoff-day[i],night[i]-cutoff))
    xyz,speed,course=a[best,:3],a[best,3],a[best,4]
    velocity=speed*np.array([np.cos(course),np.sin(course),0])
    # Score the last10s, with2s of prior samples for the original state definition.
    times=np.arange(-6,31)/3;position=xyz+(times[:,None]-5)*velocity
    data=dict(pos=position[None],valid=np.ones((1,len(times)),bool))
    frames=np.arange(6,len(times));states=states_at(data,np.zeros(len(frames),dtype=int),frames)
    assert np.all(model.score_context(states,0)['state_nll']<=cutoff)
    assert np.all(model.score_context(states,1)['state_nll']>cutoff)
    return dict(id='synthetic_drone',title='SYNTHETIC DRONE 01',kind='synthetic_drone',
                times=times[6:],path=position[6:],states=states,
                source='Constructed constant-velocity track; not an observed drone recording',
                selection=dict(searched_states=len(a),qualifying_states=len(eligible),
                               rule='maximum minimum cutoff margin; showcase selection, not test performance'))


def reference_examples(root,model,cutoff):
    with (root/'outputs/anomaly_context_time/token_scores.csv').open() as f:rows=list(csv.DictReader(f))
    candidates=[r for r in rows if r['source']=='aircraft' and r['date']=='2026-10-04'
                and float(r['same_state_day_state_nll'])<=cutoff<float(r['same_state_night_state_nll'])
                and min(int(r['day_local_support']),int(r['night_local_support']))>=1000]
    candidates.sort(key=lambda r:min(cutoff-float(r['same_state_day_state_nll']),float(r['same_state_night_state_nll'])-cutoff),reverse=True)
    cases=[];cells=set()
    for row in candidates:
        d=load_processed(root/'data/processed'/f"{row['scenario']}.npz")
        agent=list(d['agent_id']).index(row['agent_id']);frame=int(row['token_start_frame'])
        if frame<30 or not d['valid'][agent,frame-30:frame+1].all():continue
        states=states_at(d,[agent],[frame]);cell=int(model.global_model.bins(states)[0][0])
        if cell in cells:continue
        cells.add(cell)
        for label,key in [(0,'same_state_day_state_nll'),(1,'same_state_night_state_nll')]:
            np.testing.assert_allclose(model.score_context(states,label)['state_nll'],float(row[key]),atol=1e-10,rtol=0)
        cases.append(dict(id=f'reference_aircraft_{len(cases)+1:02d}',title=f'AIRCRAFT REFERENCE {len(cases)+1:02d}',kind='reference_aircraft',
                          times=np.arange(-30,1)/3,path=d['pos'][agent,frame-30:frame+1].astype(float),states=states,
                          source=f"{row['scenario']} / {row['agent_id']} / frame {frame}",
                          selection=dict(qualifying_token_rows=len(candidates),minimum_support_each=1000,
                                         rule='largest minimum margin, distinct cells, valid10s reference history; illustrative selection')))
        if len(cases)==2:break
    if len(cases)<2:raise RuntimeError('Fewer than two supported reference examples; report without tuning')
    return cases


def render_pair(case,model,cutoff,out,animation_callback=None):
    state={k:v[-1:] for k,v in case['states'].items()}
    day=model.score_context(state,0);night=model.score_context(state,1)
    assert day['state_nll'][0]<=cutoff<night['state_nll'][0]
    cell=int(day['cell'][0]);height=int(day['height_bin'][0]);g=model.global_model
    support=[int(m.pooled_joint[cell,height]) for m in model.models]
    differences=np.array([night[k][0]-day[k][0] for k in FACTORS]);driver=int(np.argmax(differences))
    xy=case['path'][...,:2];center=xy.mean(0)
    span=max(1200.,float(np.ptp(xy,axis=0).max())*1.7)
    bounds=np.array([center-[span/2,span*.34],center+[span/2,span*.34]])
    image,zoom,pb=cached_basemap(bounds,out.parents[1]/'data/basemap_cache')
    projected=visual.enu_to_pixel(xy,zoom)
    cxy=g.bounds[0]+np.array([cell%g.nx,cell//g.nx])*250
    polygon=visual.enu_to_pixel(cxy+np.array([[0,0],[250,0],[250,250],[0,250]]),zoom)
    fig=plt.figure(figsize=(20,10),facecolor=BG);animation_artists=[]
    fig.text(.035,.957,'SYGARD / CONTEXTUAL ANOMALY',color=TEXT,fontsize=21,fontweight='bold')
    fig.text(.035,.918,f"{case['title']}  ·  SAME POSITION, HEIGHT, SPEED AND DIRECTION",color=MUTED,fontsize=12)
    for label,(left,scores,title,clock) in enumerate([(0.035,day,'DAYTIME','12:00 local'),(.525,night,'NIGHTTIME','22:00 local')]):
        colour=GREEN if label==0 else ORANGE
        fig.text(left,.862,title,color=colour,fontsize=20,fontweight='bold')
        fig.text(left+.30,.865,clock,color=MUTED,fontsize=13)
        ax=fig.add_axes([left,.34,.44,.48]);ax.axis('off')
        ax.imshow(image,extent=(pb[0,0],pb[1,0],pb[1,1],pb[0,1]))
        ax.set(xlim=(pb[0,0],pb[1,0]),ylim=(pb[1,1],pb[0,1]),aspect='equal')
        ax.add_patch(Polygon(polygon,closed=True,fill=False,ec='#c77dff',ls=':',lw=1.6,alpha=.8))
        segments=np.stack([projected[:-1],projected[1:]],axis=1)
        colours=np.tile(matplotlib.colors.to_rgba(CYAN),(len(segments),1));colours[:,3]=np.linspace(.15,.95,len(segments))
        trail=LineCollection(segments,colors=colours,linewidths=3);ax.add_collection(trail)
        course=float(state['course'][0]);marker=(DRONE if case['kind']=='synthetic_drone' else visual.PLANE).transformed(Affine2D().rotate(course))
        icon,=ax.plot(*projected[-1],marker=marker,ms=30,mfc=CYAN,mec=ORANGE if label else BG,mew=.8 if label else 1,ls='',zorder=4)
        name=ax.annotate('Drone 01' if case['kind']=='synthetic_drone' else 'Aircraft',projected[-1],xytext=(18,24),textcoords='offset points',color=colour,fontsize=13,
                    bbox=dict(fc=BG,ec='none',pad=3))
        animation_artists.append((trail,icon,name))
        ax.text(.02,.025,'Same 10 s track · dotted outline = 250 m grid cell',transform=ax.transAxes,color=TEXT,fontsize=9,bbox=dict(fc=BG,ec='none',pad=4))
        fig.add_artist(FancyBboxPatch((left,.15),.44,.145,boxstyle='round,pad=.012',fc='#132434',ec='#294052',transform=fig.transFigure))
        fig.text(left+.012,.254,'NORMAL' if label==0 else 'ANOMALY',color=colour,fontsize=21,fontweight='bold')
        fig.text(left+.012,.20,f"Anomaly score  {scores['state_nll'][0]:.2f}",color=TEXT,fontsize=21)
        fig.text(left+.30,.208,f'Cutoff {cutoff:.2f}',color=MUTED,fontsize=12)
        fig.text(left+.012,.165,'Within reference range' if label==0 else f"Main increase: {FACTOR_NAMES[driver]}",color=MUTED,fontsize=11)
    pos=state['position'][0]
    fig.text(.035,.105,f"Height {pos[2]:.0f} m   ·   Speed {state['speed'][0]:.1f} m/s   ·   Train support: {support[0]:,} day / {support[1]:,} night pooled states",color=TEXT,fontsize=12)
    fig.text(.035,.066,'Selected clock-only counterfactual · aircraft reference density · normal does not identify object type or establish safety',color=MUTED,fontsize=10)
    fig.text(.035,.034,'Synthetic drone track' if case['kind']=='synthetic_drone' else 'Reference aircraft track; only clock period is counterfactually changed',color=MUTED,fontsize=10)
    fig.text(.965,.034,'© OpenStreetMap contributors',ha='right',color=MUTED,fontsize=9)
    target=out/f"{case['id']}_day_vs_night.png";fig.savefig(target,dpi=160,facecolor=BG)
    if animation_callback is not None:animation_callback(fig,animation_artists,image,zoom)
    plt.close(fig)
    np.savez_compressed(out/f"{case['id']}_track.npz",time_s=case['times'],position=case['path'])
    record=dict(id=case['id'],kind=case['kind'],source=case['source'],selection=case['selection'],
                position_enu_m=pos.tolist(),speed_mps=float(state['speed'][0]),course_rad=float(state['course'][0]),
                day_score=float(day['state_nll'][0]),night_score=float(night['state_nll'][0]),cutoff=cutoff,
                support_day=support[0],support_night=support[1],largest_increase=FACTOR_NAMES[driver],
                component_increase=dict(zip(FACTOR_NAMES,differences.tolist())),
                components_day={k:float(day[k][0]) for k in FACTORS},components_night={k:float(night[k][0]) for k in FACTORS},
                same_track_for_both_contexts=True,image=target.name)
    return record


def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/context_flip_examples';out.mkdir(exist_ok=True)
    before=frozen_hashes(root)
    model=ContextDensity.load(root/'outputs/anomaly_context_time/state_density.npz')
    cutoff=json.loads((root/'outputs/anomaly_context_time/acceptance.json').read_text())['cutoffs']['context']
    cases=[find_drone(model,cutoff),*reference_examples(root,model,cutoff)]
    records=[render_pair(case,model,cutoff,out) for case in cases]
    # Component accounting: no bonus or score weighting added to make a flip.
    for r in records:
        assert np.isclose(sum(r['components_day'].values()),r['day_score'])
        assert np.isclose(sum(r['components_night'].values()),r['night_score'])
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,r in zip(axes,records):
        bottom=np.zeros(2)
        for key,label,colour in zip(FACTORS,FACTOR_NAMES,['#4c78a8','#f58518','#54a24b']):
            values=[r['components_day'][key],r['components_night'][key]]
            ax.bar(['Day','Night'],values,bottom=bottom,color=colour,label=label);bottom+=values
        ax.axhline(cutoff,color='#b22222',ls='--');ax.set(title=r['id'].replace('_',' ').title(),ylabel='State anomaly score (NLL)',ylim=(0,max(18,float(bottom.max())+1)))
        for i,value in enumerate(bottom):ax.text(i,value+.2,f'{value:.2f}',ha='center',fontweight='bold')
    axes[-1].legend(fontsize=8,loc='lower right');fig.suptitle('Selected examples: identical states and common cutoff; only clock context changes')
    fig.tight_layout();fig.savefig(out/'component_comparison.png',dpi=180);plt.close(fig)
    assert before==frozen_hashes(root)
    receipt=dict(cutoff=cutoff,model_sha256=digest(root/'outputs/anomaly_context_time/state_density.npz'),
                 frozen_files_unchanged=len(before),refit=False,model_rollout=False,examples=records)
    (out/'examples.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# Selected clock-context anomaly flips','',
           'Showcase illustrations selected for crossing the existing cutoff. These are not randomly sampled examples or new performance measurements. No model, cutoff, score formula, context bins or earlier demo was changed. Map/track geometry is identical between each pair; lighting is not simulated. Clock labels mean Chicago local periods07–19/19–07, not actual solar illumination.','',
           'The new synthetic drone is a10s straight constant-speed trajectory constructed to illustrate the frozen aircraft-reference density. Height30m/speed25m/s were selected from a declared finite candidate grid. The original native drone route was not modified. A green NORMAL label means this kinematic state is within the aircraft-reference range; object identity, permission, safety and threat are not inferred.','',
           '| Example | Day score | Night score | Cutoff | Main contribution to increase |','|---|---:|---:|---:|---|']
    lines += [f"| {r['id']} | {r['day_score']:.2f} | {r['night_score']:.2f} | {cutoff:.2f} | {r['largest_increase']} |" for r in records]
    lines += ['', 'Train support counts are spatially pooled and serially correlated. Synthetic current-state support is nonzero in both contexts; the context differences reflect the supplied sample and may include runway flow/selection effects. Two additional cases retain actual10-04reference aircraft histories. Source IDs, selection rules, scores, factors and support are in examples.json. PNGs are3200×1600. Raw tracks are saved as NPZ. No weather, neural inference, training, packages, network or threat changes.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
