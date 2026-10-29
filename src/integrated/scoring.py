"""Read-only frozen state scoring for observations and sampled futures."""
import csv
import json
from pathlib import Path
import numpy as np
from src.eval.anomaly_state import StateDensity

COMPONENTS=['location_height_nll','speed_nll','course_nll','state_nll']


class FrozenStateScorer:
    def __init__(self,directory):
        directory=Path(directory)
        with np.load(directory/'state_density.npz') as arrays:
            self.model=StateDensity(arrays['bounds'],bin_smoothing=True)
            for key in ['p_joint','p_speed','p_course']:setattr(self.model,key,arrays[key].copy())
        if self.model.p_joint.shape!=self.model.joint.shape:raise ValueError('Requires frozen five-height density')
        reference=json.loads((directory/'standardization.json').read_text())['scores']['state_nll']
        self.median,self.iqr=reference['median'],reference['iqr']
        with (directory/'summary.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.cutoff=float(next(r['p95'] for r in rows if r['score']=='state_nll' and r['group']=='aircraft-airborne (2026-10-04)'))
        self.z_cutoff=(self.cutoff-self.median)/self.iqr

    def states(self,position,previous):
        """Arbitrary leading shape, endpoints exactly2s apart; same old scorer."""
        position,previous=np.asarray(position),np.asarray(previous)
        shape=position.shape[:-1];delta=(position-previous)[...,:2]/2
        states=dict(position=position.reshape(-1,3),speed=np.linalg.norm(delta,axis=-1).reshape(-1),
            course=np.arctan2(delta[...,1],delta[...,0]).reshape(-1))
        result={key:value.reshape(shape) for key,value in self.model.score(states).items()}
        result['z_state']=(result['state_nll']-self.median)/self.iqr
        return result

    def observation_trace(self,observed,valid):
        n,t,_=observed.shape
        result={key:np.full((n,t),np.nan) for key in COMPONENTS+['z_state']}
        for frame in range(6,t):
            eligible=valid[:,frame]&valid[:,frame-6]
            if not eligible.any():continue
            scores=self.states(observed[eligible,frame],observed[eligible,frame-6])
            for key in result:result[key][eligible,frame]=scores[key]
        return result

    def futures(self,predictions,current):
        """+2..+30s; first displacement anchored at current observation, no GT."""
        sampled=predictions[:,:,5::6,:]
        anchor=np.broadcast_to(current[None,:,None,:],(*predictions.shape[:2],1,3))
        previous=np.concatenate([anchor,sampled[:,:,:-1]],axis=2)
        return self.states(sampled,previous)


def future_anomaly_evidence(scores,cutoff,min_samples=8):
    """Input[R,N,15]; expose per-horizon sample counts, never calibrated odds."""
    count=np.count_nonzero(scores>cutoff,axis=0)
    earliest=[]
    for row in count:
        hit=np.flatnonzero(row>=min_samples)
        earliest.append(float((hit[0]+1)*2) if len(hit) else None)
    return dict(exceedance_counts=count.tolist(),earliest_horizon_s=earliest,
        median_state_nll=np.median(scores,axis=0).tolist(),p90_state_nll=np.percentile(scores,90,axis=0).tolist())
