"""M4 acceptance: iterate every train window with CPU loader workers."""
import argparse
import json
import torch
from src.data.datamodule import AirDataModule


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(1)
    dm = AirDataModule('data/processed', 'data/splits', num_workers=args.workers)
    dm.setup('fit')
    windows = agents = targets = batches = 0
    for batch in dm.train_dataloader():
        windows += batch.num_graphs
        agents += batch['agent'].num_nodes
        targets += int(batch['agent']['train_mask'].sum())
        batches += 1
        if batches % 1000 == 0:
            print(f'{batches} batches, {windows} windows', flush=True)
    assert windows == len(dm.train_dataset)
    print(json.dumps(dict(batches=batches, windows=windows, agents=agents, targets=targets, workers=args.workers)))


if __name__ == '__main__':
    main()
