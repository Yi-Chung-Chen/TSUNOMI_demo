"""3D constant velocity and exact GT replay, with the Demo rollout layout."""
import numpy as np
from src.common.units import to_metres, FRAME_HZ
from src.rollout.rollout_io import as_numpy


def baselines(graph, n_rollouts=32, current=18, future_frames=90):
    a = graph['agent']
    position = as_numpy(to_metres(a['position'], float(graph['pos_scale']))).astype(np.float32)
    valid, heading = as_numpy(a['valid_mask']), as_numpy(a['heading'])
    velocity = np.zeros((len(position), 3), np.float64)
    for i in range(len(position)):
        observed = np.flatnonzero(valid[i, :current + 1])
        if len(observed) >= 2:
            t0, t1 = observed[-2:]
            velocity[i] = (position[i, t1].astype(np.float64) - position[i, t0]) / ((t1 - t0) / FRAME_HZ)
    seconds = np.arange(1, future_frames + 1) / FRAME_HZ
    cv = position[:, current:current + 1].astype(np.float64) + seconds[None, :, None] * velocity[:, None]
    cv_heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    cv_heading = np.where(np.linalg.norm(velocity[:, :2], axis=-1) > 1e-6, cv_heading, heading[:, current])
    future = position[:, current + 1:current + future_frames + 1]
    return dict(constant_velocity=dict(position=np.repeat(cv[None], n_rollouts, 0),
                                       heading=np.broadcast_to(cv_heading[None, :, None], (n_rollouts, len(position), future_frames))),
                gt_replay=dict(position=np.repeat(future[None], n_rollouts, 0),
                               heading=np.repeat(heading[None, :, current + 1:current + future_frames + 1], n_rollouts, 0)))
