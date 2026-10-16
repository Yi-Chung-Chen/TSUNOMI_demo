"""Version2 Demo NPZ: predicted position[R,N,90,3], all lengths metres."""
import json
from pathlib import Path
import numpy as np
import torch
from src.common.units import to_metres, FRAME_HZ, GROUND_ALTITUDE_M


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def save_rollout_window(graph, prediction, path, model_name, seed, checkpoint_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = graph['agent']
    arrays = dict(position=as_numpy(prediction['position']).astype(np.float32),
                  heading=as_numpy(prediction['heading']).astype(np.float32),
                  gt_position=as_numpy(to_metres(a['position'], float(graph['pos_scale']))).astype(np.float32),
                  gt_heading=as_numpy(a['heading']), valid_mask=as_numpy(a['valid_mask']),
                  target_mask=as_numpy(a['train_mask']), on_ground=as_numpy(a['on_ground']),
                  alt_fallback=as_numpy(a['alt_fallback']), agent_id=np.asarray(a['id']),
                  date=np.asarray(graph['date']), scenario=np.asarray(graph['scenario']),
                  start_frame=np.asarray(graph['start_frame']))
    arrays['meta_json'] = np.asarray(json.dumps(dict(schema_version=2, layout='R,N,T,3', length_unit='metres',
                                  frame_hz=FRAME_HZ, current_frame_idx=18, height_reference_altitude_m=GROUND_ALTITUDE_M,
                                  model=model_name, seed=seed, n_rollouts=arrays['position'].shape[0],
                                  num_k=5 if model_name == 'bc' else None, temp=1.0 if model_name == 'bc' else None,
                                  checkpoint_sha256=checkpoint_sha256, bounds=json.loads(graph['bounds_json']))))
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path
