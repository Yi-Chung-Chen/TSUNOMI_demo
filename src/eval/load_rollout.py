"""Strict loader for the metre-valued BC Demo rollout schema."""
import json
from pathlib import Path
import numpy as np


def load_rollout(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    meta = json.loads(str(data['meta_json']))
    assert meta['schema_version'] == 2 and meta['layout'] == 'R,N,T,3'
    assert meta['length_unit'] == 'metres' and meta['frame_hz'] == 3.
    pred, gt = data['position'], data['gt_position']
    assert pred.shape == (32, len(gt), 90, 3)
    assert gt.shape == (len(gt), 109, 3)
    assert np.isfinite(pred).all()
    assert data['valid_mask'].shape == (len(gt), 109)
    data['meta'] = meta
    return data
