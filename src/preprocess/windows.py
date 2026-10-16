"""Index fixed-length prediction windows with contiguous target eligibility."""
import argparse
import csv
from pathlib import Path
import numpy as np


def target_mask(valid, current_frame_idx=18, min_future_frames=30):
    stop = current_frame_idx + min_future_frames + 1
    if valid.ndim != 2 or valid.shape[1] < stop:
        raise ValueError('Not enough frames for history and minimum future')
    return valid[:, :stop].all(axis=1)


def index_windows(processed_dir, scenarios, stride, current_frame_idx=18,
                  future_frames=90, min_future_frames=30):
    total = current_frame_idx + future_frames + 1
    if stride < 1 or min_future_frames > future_frames:
        raise ValueError('Invalid stride or future length')
    result = []
    for name in scenarios:
        with np.load(Path(processed_dir) / f'{name}.npz', allow_pickle=False) as data:
            valid = data['valid']
            for start in range(0, valid.shape[1] - total + 1, stride):
                n = int(target_mask(valid[:, start:start + total], current_frame_idx, min_future_frames).sum())
                if n:
                    result.append(dict(scenario=name, start_frame=start, n_targets=n))
    return result


def write_windows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scenario', 'start_frame', 'n_targets'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--split-dir', type=Path, default=Path('data/splits'))
    p.add_argument('--current-frame-idx', type=int, default=18)
    p.add_argument('--future-frames', type=int, default=90)
    p.add_argument('--min-future-frames', type=int, default=30)
    p.add_argument('--train-stride', type=int, default=15)
    p.add_argument('--eval-stride', type=int, default=90)
    args = p.parse_args()
    for split in ['train', 'val']:
        names = (args.split_dir / f'{split}_manifest.txt').read_text().splitlines()
        rows = index_windows(args.processed_dir, names,
                             args.train_stride if split == 'train' else args.eval_stride,
                             args.current_frame_idx, args.future_frames, args.min_future_frames)
        write_windows(args.split_dir / f'{split}_windows.csv', rows)
        print(f'{split}: {len(names)} scenarios, {len(rows)} windows, {sum(r["n_targets"] for r in rows)} target instances')


if __name__ == '__main__':
    main()
