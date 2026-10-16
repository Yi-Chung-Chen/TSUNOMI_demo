"""Window views over scenario NPZ files; no materialized chunk copies."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import HeteroData
from src.common.units import to_internal
from src.preprocess.windows import target_mask


class AirDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir, split_dir, split, pos_scale=10., current_frame_idx=18,
                 future_frames=90, min_future_frames=30, radius=2., cache_size=32):
        self.processed_dir = Path(processed_dir)
        self.split_dir = Path(split_dir)
        self.pos_scale, self.current_frame_idx = pos_scale, current_frame_idx
        self.future_frames, self.min_future_frames, self.radius = future_frames, min_future_frames, radius
        self.total_frames = current_frame_idx + future_frames + 1
        with (self.split_dir / f'{split}_windows.csv').open() as stream:
            self.windows = list(csv.DictReader(stream))
        self.type_vocab = json.loads((self.split_dir / 'type_vocab.json').read_text())
        self._load = lru_cache(maxsize=cache_size)(self._read)

    def _read(self, name):
        with np.load(self.processed_dir / f'{name}.npz', allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row = self.windows[index]
        d = self._load(row['scenario'])
        start = int(row['start_frame'])
        sl = slice(start, start + self.total_frames)
        # Only aircraft already present at prediction time; never reveal future entrants.
        take = d['valid'][:, start + self.current_frame_idx]
        valid = d['valid'][take, sl].copy()
        graph = HeteroData()
        agent = graph['agent']
        agent.num_nodes = int(take.sum())
        agent['valid_mask'] = torch.from_numpy(valid)
        agent['position'] = torch.from_numpy(to_internal(d['pos'][take, sl].astype(np.float64), self.pos_scale))
        agent['heading'] = torch.from_numpy(d['heading'][take, sl].copy())
        agent['on_ground'] = torch.from_numpy(d['on_ground'][take, sl].copy())
        agent['alt_fallback'] = torch.from_numpy(d['alt_fallback'][take, sl].copy())
        agent['type'] = torch.tensor([self.type_vocab.get(v, 0) for v in d['agent_type'][take]], dtype=torch.long)
        agent['radius'] = torch.full((int(take.sum()),), self.radius)
        agent['id'] = d['agent_id'][take].tolist()
        agent['train_mask'] = torch.from_numpy(target_mask(valid, self.current_frame_idx, self.min_future_frames))
        graph['scenario'] = row['scenario']
        graph['start_frame'] = start
        graph['date'] = str(d['date'])
        graph['pos_scale'] = self.pos_scale
        graph['bounds_json'] = str(d['bounds_json'])
        return graph
