"""Compress sampled futures for display; weights are sample shares, not logits.

Separate entering/non-entering paths before deterministic 3D trajectory medoids,
so the colour of a displayed group describes every member, not only its medoid.
"""
import numpy as np
from src.integrated.geometry import inside_zone


def _medoid_groups(distance, indices, limit):
    indices=np.asarray(indices,dtype=int)
    d=distance[np.ix_(indices,indices)]
    medoids=[int(np.argmin(d.sum(axis=1)))]
    while len(medoids)<min(limit,len(indices)):
        nearest=d[:,medoids].min(axis=1)
        candidate=int(np.argmax(nearest))
        if nearest[candidate]<=1e-9:break # do not invent distinct paths
        medoids.append(candidate)
    for _ in range(32):
        labels=np.argmin(d[:,medoids],axis=1)
        updated=[]
        for group,medoid in enumerate(medoids):
            members=np.flatnonzero(labels==group)
            if not len(members):continue
            updated.append(int(members[np.argmin(d[np.ix_(members,members)].sum(axis=1))]))
        if updated==medoids:break
        medoids=updated
    labels=np.argmin(d[:,medoids],axis=1)
    return [(int(indices[medoid]),indices[labels==group].tolist())
            for group,medoid in enumerate(medoids) if np.any(labels==group)]


def group_futures(paths,zone,max_groups=4):
    """paths[R,T,3]; uses only predicted states, including intermediate crossings.

Allocate up to four representatives proportionally across entry/non-entry
strata, with at least one for each present stratum. Group within each stratum by
mean 3D pointwise distance. A group is display compression, not learned intent.
"""
    paths=np.asarray(paths,dtype=float)
    if paths.ndim!=3 or paths.shape[-1]!=3 or not paths.shape[0] or not paths.shape[1]:
        raise ValueError('Expected nonempty [samples,time,3] trajectories')
    if not np.isfinite(paths).all():raise ValueError('Nonfinite forecast')
    if max_groups<2:raise ValueError('Need at least two slots for entry strata')
    enters=inside_zone(paths,zone).any(axis=1)
    distance=np.linalg.norm(paths[:,None]-paths[None,:],axis=-1).mean(axis=-1)
    strata=[np.flatnonzero(enters==value) for value in [True,False]]
    nonempty=[i for i,s in enumerate(strata) if len(s)]
    allocation={i:1 for i in nonempty}
    for _ in range(min(max_groups,len(paths))-len(nonempty)):
        available=[i for i in nonempty if allocation[i]<len(strata[i])]
        if not available:break
        i=max(available,key=lambda i:len(strata[i])/allocation[i]);allocation[i]+=1
    groups=[]
    for stratum in nonempty:
        for representative,members in _medoid_groups(distance,strata[stratum],allocation[stratum]):
            weight=len(members)/len(paths)
            percentage=f'{100*weight:.3f}'.rstrip('0').rstrip('.')+'%'
            groups.append(dict(representative_index=representative,members=members,count=len(members),
                total_samples=len(paths),share=weight,percentage=percentage,enters_zone=bool(enters[representative])))
    groups.sort(key=lambda group:(-group['count'],group['representative_index']))
    for i,group in enumerate(groups):group['label']=chr(65+i)
    return groups


def replay_groups(timeline,forecasts,zone):
    """Foreground-only grouping of each frame's remaining forecast, not past entry."""
    result={}
    for frame,row in enumerate(timeline):
        origin=row['objects'][0]['forecast_origin_frame']
        if origin is not None:
            paths=forecasts[origin]['position'][:,0,frame-origin:]
            result[frame]=group_futures(paths,zone)
    return result


def replay_object_groups(timeline,forecasts,zone):
    """Independent 32-sample grouping for every currently anomalous object."""
    result={}
    for frame,row in enumerate(timeline):
        per_object={}
        for agent,obj in enumerate(row['objects']):
            origin=obj['forecast_origin_frame']
            if obj['anomalous'] and origin is not None:
                paths=forecasts[origin]['position'][:,agent,frame-origin:]
                per_object[agent]=group_futures(paths,zone)
        if per_object:result[frame]=per_object
    return result
