"""Clock-period-conditioned M8b density with fixed global-prior shrinkage."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import numpy as np
from src.eval.anomaly_state import StateDensity

CONTEXTS=('day_07_19','night_19_07')
TIMEZONE='America/Chicago'
JOINT_PRIOR_STATES=1000.
LOCAL_PRIOR_STATES=100.


def context_from_utc(utc_s):
    """07:00 <= Chicago local clock <19:00; not astronomical daylight.

Conversion uses the timezone database (including DST). Quantizing to minutes
is exact for these hour boundaries, including a forecast crossing a boundary.
"""
    values=np.asarray(utc_s,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Missing/nonfinite UTC context')
    minutes=np.floor(values/60).astype(np.int64)
    unique,inverse=np.unique(minutes,return_inverse=True)
    tz=ZoneInfo(TIMEZONE)
    labels=np.array([0 if 7<=datetime.fromtimestamp(int(m)*60,timezone.utc).astimezone(tz).hour<19 else 1 for m in unique])
    return labels[inverse].reshape(values.shape)


def subset(states,mask):return {key:np.asarray(value)[mask] for key,value in states.items()}


class ContextDensity:
    """P(cell,height|clock) P(speed|cell,height,clock) P(course|cell,height,clock).

Fit the exact existing global model. Context joint counts receive 1000 pooled
pseudo-states distributed according to its global joint probability. Local
speed/course distributions receive 100 pseudo-states from the matching global
conditional. Values are fixed before evaluation, not tuned to examples.
"""
    def __init__(self,bounds):
        self.global_model=StateDensity(bounds)
        self.models=[StateDensity(bounds) for _ in CONTEXTS]

    def add(self,states,utc_s):
        labels=np.broadcast_to(context_from_utc(utc_s),(len(states['position']),))
        self.global_model.add(states)
        for label,model in enumerate(self.models):model.add(subset(states,labels==label))

    def finish(self):
        g=self.global_model.finish()
        for model in self.models:
            model.finish()
            model.p_joint=(model.pooled_joint+JOINT_PRIOR_STATES*g.p_joint)/(model.pooled_joint.sum()+JOINT_PRIOR_STATES)
            model.p_speed=(model.smoothed_speed+LOCAL_PRIOR_STATES*g.p_speed)/(model.pooled_joint[...,None]+LOCAL_PRIOR_STATES)
            model.p_course=(model.smoothed_course+LOCAL_PRIOR_STATES*g.p_course)/(model.pooled_joint[...,None]+LOCAL_PRIOR_STATES)
            assert np.isclose(model.p_joint.sum(),1.)
            assert np.allclose(model.p_speed.sum(-1),1.) and np.allclose(model.p_course.sum(-1),1.)
        assert sum(m.eligible_count for m in self.models)==g.eligible_count
        return self

    def score_context(self,states,labels):
        labels=np.broadcast_to(np.asarray(labels),(len(states['position']),))
        if not np.isin(labels,[0,1]).all():raise ValueError('Unknown clock context')
        result={}
        for label,model in enumerate(self.models):
            mask=labels==label
            if not mask.any():continue
            scores=model.score(subset(states,mask))
            for key,value in scores.items():
                if key not in result:result[key]=np.empty(len(labels),dtype=value.dtype)
                result[key][mask]=value
        return result

    def score(self,states,utc_s):return self.score_context(states,context_from_utc(utc_s))

    def save(self,path):
        values=dict(bounds=self.global_model.bounds,contexts=np.array(CONTEXTS),timezone=np.array(TIMEZONE),
                    joint_prior_states=JOINT_PRIOR_STATES,local_prior_states=LOCAL_PRIOR_STATES)
        for label,model in [('global',self.global_model),*zip(CONTEXTS,self.models)]:
            for key in ['joint','speed','course','p_joint','p_speed','p_course']:
                values[f'{label}_{key}']=getattr(model,key)
        np.savez_compressed(path,**values)

    @classmethod
    def load(cls,path):
        """Rebuild fixed probabilities from saved counts and verify provenance."""
        with np.load(path,allow_pickle=False) as values:
            assert tuple(values['contexts'].tolist())==CONTEXTS
            assert str(values['timezone'])==TIMEZONE
            assert float(values['joint_prior_states'])==JOINT_PRIOR_STATES
            assert float(values['local_prior_states'])==LOCAL_PRIOR_STATES
            result=cls(values['bounds'])
            for label,model in [('global',result.global_model),*zip(CONTEXTS,result.models)]:
                for key in ['joint','speed','course']:setattr(model,key,values[f'{label}_{key}'].copy())
                model.eligible_count=int(model.joint.sum())
            result.finish()
            for label,model in [('global',result.global_model),*zip(CONTEXTS,result.models)]:
                for key in ['p_joint','p_speed','p_course']:
                    np.testing.assert_array_equal(getattr(model,key),values[f'{label}_{key}'])
        return result
