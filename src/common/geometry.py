"""Circular contour and planar transforms shared by dictionary and tokenizer.

Adapted from starcraft-motion-private/build_motion_dictionary.py.
Point order is front, left, back, right, matching CAT-K.
"""
import numpy as np


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def contours(traj, radius=2.0):
    angle = traj[..., 2, None] + np.asarray([0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=traj.dtype)
    return traj[..., None, :2] + radius * np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def rotate(xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([xy[..., 0] * c - xy[..., 1] * s, xy[..., 0] * s + xy[..., 1] * c], -1)


def causal_headings(position, heading, valid, on_ground, shift=6, frame_hz=3.0):
    """Chord heading uses only the previous shift frames; data fallback otherwise."""
    out = heading.copy()
    for t in range(shift, position.shape[1]):
        delta = position[:, t, :2] - position[:, t - shift, :2]
        eligible = valid[:, t - shift:t + 1].all(-1) & ~on_ground[:, t]
        eligible &= np.linalg.norm(delta, axis=-1) * frame_hz / shift > 10.0
        out[eligible, t] = np.arctan2(delta[eligible, 1], delta[eligible, 0])
    return out


def match_endpoints(position, heading, previous_position, previous_heading, dictionary, batch_size=128, endpoint_idx=-1):
    """Nearest joint xy-contour/dz endpoint in each reconstructed local frame."""
    local_xy = rotate(position[:, :2] - previous_position[:, :2], -previous_heading)
    target = contours(np.concatenate([local_xy, wrap_angle(heading - previous_heading)[:, None]], -1), dictionary['meta']['radius'])
    target_dz = position[:, 2] - previous_position[:, 2]
    vocab = dictionary['cluster_centers'][:, endpoint_idx]
    vocab_dz = dictionary['dz'][:, endpoint_idx]
    ids = []
    for start in range(0, len(target), batch_size):
        diff = target[start:start + batch_size, None] - vocab[None]
        dist = np.sqrt((diff * diff).sum(-1)).mean(-1)
        dist += dictionary['meta']['lambda_z'] * np.abs(target_dz[start:start + batch_size, None] - vocab_dz)
        ids.append(dist.argmin(-1))
    return np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)


def decode_tokens(ids, previous_position, previous_heading, dictionary):
    traj = dictionary['traj'][ids]
    xy = rotate(traj[..., :2], previous_heading[:, None]) + previous_position[:, None, :2]
    z = dictionary['dz'][ids] + previous_position[:, None, 2]
    head = wrap_angle(traj[..., 2] + previous_heading[:, None])
    return np.concatenate([xy, z[..., None]], -1), head
