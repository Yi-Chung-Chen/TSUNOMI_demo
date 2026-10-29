"""Reference/noise adapters; model inputs contain observed states only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.common.units import to_internal
from src.data.camera_trajectory import camera_features


def verify_lock(root):
    root=Path(root)
    lock=json.loads((root/'configs/integrated_demo/v1.lock.json').read_text())
    for name,expected in lock.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Frozen artifact changed: {name}')
    return lock


def stable_seed(seed,*keys):
    return int.from_bytes(hashlib.sha256('/'.join(map(str,(seed,*keys))).encode()).digest()[:8],'little')%(2**63-1)


def noisy_observations(reference,object_ids,sigma,seed,source):
    """One seeded full-stream realization; extending the future preserves prefix."""
    reference=np.asarray(reference,dtype=float)
    if reference.ndim!=3 or reference.shape[-1]!=3 or len(reference)!=len(object_ids):
        raise ValueError('Expected agent,time,xyz reference')
    if sigma<0:raise ValueError('Negative noise standard deviation')
    result=reference.copy()
    for i,object_id in enumerate(object_ids):
        rng=np.random.default_rng(stable_seed(seed,source,object_id))
        result[i]+=rng.normal(0,sigma,size=reference.shape[1:])
    return result


def observation_features(position,valid):
    """Causal velocity/heading/ground estimates; no clean annotations."""
    n,t,_=position.shape
    velocity=np.zeros_like(position,dtype=float)
    for frame in range(1,t):
        past=max(0,frame-6);dt=(frame-past)/3
        usable=valid[:,past:frame+1].all(1)
        velocity[usable,frame]=(position[usable,frame]-position[usable,past])/dt
    heading=np.zeros((n,t));ground=np.zeros((n,t),bool)
    for a in range(n):
        heading[a],ground[a]=camera_features(dict(position=position[a],velocity=velocity[a],valid=valid[a]))
    return velocity,heading,ground


def observation_graph(observed,valid,current_frame,pos_scale=10.):
    if current_frame<18:raise ValueError('Need6s history')
    past=slice(current_frame-18,current_frame+1)
    if not valid[:,past].all():raise ValueError('Incomplete observed history')
    # Feature estimation can carry past filter state, but never reads future rows.
    velocity,heading,ground=observation_features(observed[:,:current_frame+1],valid[:,:current_frame+1])
    graph=HeteroData();a=graph['agent'];n=len(observed);a.num_nodes=n
    a['position']=torch.from_numpy(to_internal(observed[:,past].astype(float),pos_scale))
    a['heading']=torch.from_numpy(heading[:,past]);a['on_ground']=torch.from_numpy(ground[:,past])
    a['valid_mask']=torch.from_numpy(valid[:,past].copy())
    a['alt_fallback']=torch.zeros((n,19),dtype=torch.bool)
    a['type']=torch.zeros(n,dtype=torch.long);a['radius']=torch.full((n,),2.)
    a['train_mask']=torch.ones(n,dtype=torch.bool)
    return graph


def scripted_route(times,scene):
    center=np.array(scene['center_m'],dtype=float);r=scene['radius_m'];omega=scene['angular_speed_rad_s']
    phase=times*omega
    result=np.tile(center,(len(times),1));result[:,0]+=r*np.cos(phase);result[:,1]+=r*np.sin(phase)
    if scene['route']=='circle_then_accelerate_west':
        change=scene['motion_change_s'];end=scene['acceleration_end_s'];duration=end-change
        origin=center+np.array([r*np.cos(change*omega),r*np.sin(change*omega),0])
        dt=np.maximum(times-change,0);ramp=np.minimum(dt,duration)
        v0=r*omega;v1=scene['westward_speed_mps'];acc=(v1-v0)/duration
        distance=v0*ramp+.5*acc*ramp**2+v1*np.maximum(dt-duration,0)
        mask=times>=change;result[mask]=origin;result[mask,0]-=distance[mask]
    return result


def scenario_stream(root,config,scene):
    times=np.arange(round(config['duration_s']*3)+1)/3
    drone=scripted_route(times,scene)
    bg=config['background']
    with np.load(Path(root)/bg['path'],allow_pickle=False) as arrays:
        agent=list(arrays['agent_id']).index(bg['agent_id']);start=bg['start_frame']
        background=arrays['pos'][agent,start:start+len(times)].astype(float)
        valid=arrays['valid'][agent,start:start+len(times)].copy()
    if len(background)!=len(times) or not valid.all():raise ValueError('Background segment not fully available')
    reference=np.stack([drone,background]);valid=np.ones(reference.shape[:2],bool)
    ids=['scripted_drone',str(bg['agent_id'])]
    observed=noisy_observations(reference,ids,scene['scripted_noise_sigma_m_per_axis'],config['seed'],scene['id'])
    return dict(times=times,reference=reference,observed=observed,valid=valid,object_ids=np.asarray(ids))
