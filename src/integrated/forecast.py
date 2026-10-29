"""Frozen inference, joint medoid and numeric-only constant-velocity reference."""
import numpy as np
import torch
from src.model.air_smart import AirSMART
from src.rollout.closed_loop import rollout


def load_model(root,checkpoint):
    saved=torch.load(root/checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==16
    model=AirSMART(**saved['hyper_parameters']);model.load_state_dict(saved['state_dict'],strict=True)
    return model.eval().requires_grad_(False)


def medoid_index(predictions):
    """One coherent joint sample; mean3D L2 over agents/frames, never GT."""
    p=np.asarray(predictions,dtype=float)
    distances=np.linalg.norm(p[:,None]-p[None,:],axis=-1).mean(axis=(-1,-2))
    return int(np.argmin(distances.sum(1)))


def constant_velocity(observed,current_frame):
    velocity=(observed[:,current_frame]-observed[:,current_frame-6])/2
    return observed[:,current_frame,None]+velocity[:,None]*np.arange(1,91)[None,:,None]/3


def infer(model,graph,seed):
    prediction=rollout(model,graph,n_rollouts=32,num_k=5,temp=1.,seed=seed)
    return {key:value.cpu().numpy() for key,value in prediction.items()}
