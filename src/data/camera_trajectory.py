"""Camera-only, causal input adapter. GT associations never enter model features."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.common.units import FRAME_HZ, GROUND_ALTITUDE_M, to_internal


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_camera_tracks(path,origin_altitude=2250.):
    """Final sequential camera event at a timestamp is the posterior state.

    Ignore matched_source_object_id, confirmed_moving and all GT metadata.
    Record whether ANY event at each timestamp incorporated a measurement.
    """
    groups=defaultdict(list)
    for row in read_jsonl(path):groups[int(row['tracker_id'])].append(row)
    result={}
    for track_id,rows in sorted(groups.items()):
        by_time=defaultdict(list)
        for row in rows:by_time[float(row['elapsed_s'])].append(row)
        times=np.array(sorted(by_time),dtype=float)
        final=[max(by_time[t],key=lambda r:int(r['event_index'])) for t in times]
        pos=np.asarray([r['xyz_enu_m'] for r in final],dtype=float)
        pos[:,2]+=origin_altitude-GROUND_ALTITUDE_M
        velocity=np.asarray([r['velocity_enu_mps'] for r in final],dtype=float)
        std=np.asarray([r['position_std_m'] for r in final],dtype=float)
        if not (np.isfinite(pos).all() and np.isfinite(velocity).all() and np.isfinite(std).all()):
            raise ValueError('Nonfinite camera state')
        result[track_id]=dict(times=times,position=pos,velocity=velocity,std=std,
            measured=np.array([any(r['status']=='measured' for r in by_time[t]) for t in times]),
            final_measured=np.array([r['status']=='measured' for r in final]),
            event_index=np.array([r['event_index'] for r in final]),native_rows=len(rows))
        t=result[track_id]
        t['initialized']=(np.cumsum(t['measured'])>=2)|(np.linalg.norm(velocity,axis=1)>0.)
    return result


def sample_camera(track,query,max_age=1.):
    """Causal tracker CV projection from the latest known state, age≤1s."""
    query=np.asarray(query,dtype=float)
    index=np.searchsorted(track['times'],query,side='right')-1
    take=np.maximum(index,0)
    age=query-track['times'][take]
    valid=(index>=0)&(age<=max_age)&(age>=0)
    valid &= track.get('initialized',np.ones(len(track['times']),bool))[take]
    # Bounded velocity projection is causal even if no later event has arrived.
    # Display/export may separately stop at the recorded track lifetime.
    return dict(position=track['position'][take]+track['velocity'][take]*age[:,None],velocity=track['velocity'][take].copy(),
        std=track['std'][take].copy(),valid=valid,age_s=age,
        native_index=index,source_time_s=track['times'][take],
        final_measured=track['final_measured'][take])


def camera_features(sample,ground_enter=10.,ground_exit=20.):
    """Unvalidated fixed demo ground proxy; all estimates are camera-only.

    Raw heading uses current Kalman velocity, then the previous usable heading.
    The existing tokenizer handles causal2s chord headings unchanged.
    """
    if ground_enter>=ground_exit:raise ValueError('Ground hysteresis thresholds must increase')
    pos,valid=sample['position'],sample['valid']
    ground=np.zeros(len(pos),dtype=bool);heading=np.zeros(len(pos))
    previous_ground=False;previous_heading=0.
    for i in range(len(pos)):
        if not valid[i]:continue
        if i==0 or not valid[:i].any():previous_ground=bool(pos[i,2]<ground_enter)
        elif pos[i,2]<ground_enter:previous_ground=True
        elif pos[i,2]>ground_exit:previous_ground=False
        ground[i]=previous_ground
        vector=sample['velocity'][i,:2]
        if np.linalg.norm(vector)>1.:previous_heading=float(np.arctan2(vector[1],vector[0]))
        heading[i]=previous_heading
    return heading,ground


def first_camera_origin(track,max_age=1.,min_measurements=3):
    """First eligible3Hz forecast start, chosen without any GT or future error."""
    first=int(np.ceil(track['times'][0]*FRAME_HZ))+18
    last=int(np.floor(track['times'][-1]*FRAME_HZ))
    for frame in range(first,last+1):
        query=np.arange(frame-18,frame+1)/FRAME_HZ
        sample=sample_camera(track,query,max_age)
        measured=track['times'][track['measured']]
        count=np.count_nonzero((measured>=query[0])&(measured<=query[-1]))
        recent=measured[measured<=query[-1]]
        fresh=len(recent)>0 and query[-1]-recent[-1]<=1.
        if sample['valid'].all() and count>=min_measurements and fresh:return frame
    return None


def make_camera_graph(track,current_frame,max_age=1.,pos_scale=10.,ground_enter=10.,ground_exit=20.):
    """Exactly19 historical samples; no GT and no future arrays accepted."""
    query=np.arange(current_frame-18,current_frame+1)/FRAME_HZ
    sample=sample_camera(track,query,max_age)
    if not sample['valid'][-1]:raise ValueError('Camera history must have19 valid samples including current')
    if not sample['valid'].all():raise ValueError('Camera history must have19 valid samples')
    heading,ground=camera_features(sample,ground_enter,ground_exit)
    graph=HeteroData();a=graph['agent'];a.num_nodes=1
    a['position']=torch.from_numpy(to_internal(sample['position'][None],pos_scale))
    a['heading']=torch.from_numpy(heading[None]);a['on_ground']=torch.from_numpy(ground[None])
    a['valid_mask']=torch.from_numpy(sample['valid'][None])
    a['alt_fallback']=torch.zeros((1,19),dtype=torch.bool)
    a['type']=torch.zeros(1,dtype=torch.long)  # UNK, never GT subtype.
    a['radius']=torch.full((1,),2.);a['train_mask']=torch.ones(1,dtype=torch.bool)
    return graph,sample


def load_reference_tracks(path,origin_altitude=2250.):
    """Offline overlay only; never passed into make_camera_graph."""
    groups=defaultdict(list)
    for row in read_jsonl(path):groups[str(row['source_object_id'])].append(row)
    result={}
    for object_id,rows in groups.items():
        rows=sorted(rows,key=lambda r:r['elapsed_s'])
        times=np.asarray([r['elapsed_s'] for r in rows],dtype=float)
        position=np.asarray([r['xyz_enu_m'] for r in rows],dtype=float)
        unique,index,inverse=np.unique(times,return_index=True,return_inverse=True)
        if not np.array_equal(position,position[index][inverse]):raise ValueError('Conflicting duplicate GT time')
        position=position[index];position[:,2]+=origin_altitude-GROUND_ALTITUDE_M
        result[object_id]=dict(times=unique,position=position)
    return result


def sample_reference(track,query,max_gap=1.):
    """Bounded GT interpolation for visualization/evaluation, no extrapolation."""
    times=track['times'];query=np.asarray(query)
    right=np.searchsorted(times,query,side='left');clipped=np.minimum(right,len(times)-1)
    exact=(right<len(times))&np.isclose(times[clipped],query,atol=1e-9,rtol=0)
    left=np.maximum(right-1,0)
    valid=exact|((right>0)&(right<len(times))&(times[clipped]-times[left]<=max_gap))
    position=np.stack([np.interp(query,times,track['position'][:,j]) for j in range(3)],axis=-1)
    position[~valid]=np.nan
    return position,valid
