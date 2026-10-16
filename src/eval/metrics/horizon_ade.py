"""Agent-weighted horizon errors; adapted from SC horizon/minADE metrics.

ADE/FDE average over rollouts; minADE takes the best whole-trajectory ADE.
Horizontal and vertical minima are selected independently. Only targets with
all GT frames valid from current through the horizon are evaluated.
"""
import numpy as np


def compute(data, horizons=(10, 20, 30), miss_threshold_m=500.):
    prediction = data['position'].astype(np.float64)
    gt, valid = data['gt_position'].astype(np.float64), data['valid_mask']
    target, ground = data['target_mask'], data['on_ground'][:, 18]
    rows = []
    for horizon in horizons:
        frames = horizon * 3
        eligible = target & valid[:, 18:19 + frames].all(-1)
        error = prediction[:, :, :frames] - gt[None, :, 19:19 + frames]
        horizontal = np.linalg.norm(error[..., :2], axis=-1)
        vertical = np.abs(error[..., 2])
        for group in ['all', 'airborne', 'ground']:
            take = eligible if group == 'all' else eligible & (ground == (group == 'ground'))
            n = int(take.sum())
            for axis, distances in [('horizontal', horizontal), ('vertical', vertical)]:
                d = distances[:, take]
                row = dict(date=str(data['date']), group=group, horizon_s=horizon, axis=axis, n_agents=n,
                           minADE_m=float(d.mean(-1).min(0).mean()) if n else None,
                           ADE_m=float(d.mean()) if n else None,
                           FDE_m=float(d[:, :, -1].mean()) if n else None,
                           miss_rate=None, mean_rollout_miss_rate=None)
                if axis == 'horizontal' and horizon == 30 and n:
                    row['miss_rate'] = float((d[:, :, -1].min(0) > miss_threshold_m).mean())
                    row['mean_rollout_miss_rate'] = float((d[:, :, -1] > miss_threshold_m).mean())
                rows.append(row)
    return rows
