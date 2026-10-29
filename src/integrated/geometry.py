"""Configured demo volumes and sampled forecast intent; never inferred GT intent."""
import numpy as np


def inside_zone(position,zone):
    p=np.asarray(position,dtype=float);xy=p[...,:2];polygon=np.asarray(zone['polygon_enu_m'],float)
    inside=np.zeros(p.shape[:-1],bool);boundary=np.zeros_like(inside)
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a;cross=(xy[...,0]-a[0])*edge[1]-(xy[...,1]-a[1])*edge[0]
        on=(np.abs(cross)<1e-8)&((xy>=np.minimum(a,b)-1e-8)&(xy<=np.maximum(a,b)+1e-8)).all(-1)
        boundary |= on
        if b[1]!=a[1]:
            intercept=(b[0]-a[0])*(xy[...,1]-a[1])/(b[1]-a[1])+a[0]
            inside ^= ((a[1]>xy[...,1])!=(b[1]>xy[...,1]))&(xy[...,0]<intercept)
    lo,hi=zone['height_range_m']
    return (inside|boundary)&(p[...,2]>=lo)&(p[...,2]<=hi)&np.isfinite(p).all(-1)


def distance_to_zone(position,zone):
    p=np.asarray(position,dtype=float);polygon=np.asarray(zone['polygon_enu_m'],float)
    distances=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a;fraction=np.clip(((p[...,:2]-a)*edge).sum(-1)/(edge@edge),0,1)
        distances.append(np.linalg.norm(p[...,:2]-(a+fraction[...,None]*edge),axis=-1))
    horizontal=np.min(distances,axis=0)
    footprint=dict(zone,height_range_m=[-np.inf,np.inf]);horizontal=np.where(inside_zone(p,footprint),0,horizontal)
    lo,hi=zone['height_range_m'];vertical=np.maximum(np.maximum(lo-p[...,2],p[...,2]-hi),0)
    return np.hypot(horizontal,vertical)


def forecast_intent(predictions,zone,min_samples=8):
    """32 joint samples; entry evaluated on full3Hz predictions, no GT."""
    inside=inside_zone(predictions,zone);enters=inside.any(-1)
    result=[]
    for agent in range(predictions.shape[1]):
        times=[float((np.flatnonzero(inside[r,agent])[0]+1)/3) for r in range(len(predictions)) if enters[r,agent]]
        count=int(enters[:,agent].sum())
        result.append(dict(entry_samples=count,total_samples=len(predictions),entry_fraction=count/len(predictions),
            earliest_entry_s=min(times) if times else None,median_entry_s=float(np.median(times)) if times else None,
            relation='Forecast enters demo zone' if count>=min_samples else ('Possible entry in sampled forecasts' if count else 'Remaining outside demo zone')))
    return result
