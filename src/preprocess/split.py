"""UTC-date split and train-only ICAO vocabulary."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

DEFAULT_DATES = dict(train=['2026-09-28', '2026-09-29', '2026-09-30', '2026-10-01', '2026-10-02'],
                     val=['2026-10-03', '2026-10-04'])


def create_splits(processed_dir, output_dir, dates=None, top_types=31):
    dates = DEFAULT_DATES if dates is None else dates
    flat = [day for days in dates.values() for day in days]
    if len(flat) != len(set(flat)):
        raise ValueError('Dates overlap between splits')
    day_split = {day: split for split, days in dates.items() for day in days}
    names = {s: [] for s in dates}
    types = Counter()
    for path in sorted(Path(processed_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            day = str(data['date'])
            if day not in day_split:
                raise ValueError(f'Date {day} is not assigned')
            split = day_split[day]
            names[split].append(path.stem)
            if split == 'train':
                types.update(t for t in data['agent_type'].tolist() if t)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, scenarios in names.items():
        (out / f'{split}_manifest.txt').write_text(''.join(n + '\n' for n in scenarios))
    ranked = sorted(types, key=lambda t: (-types[t], t))[:top_types]
    vocab = {'UNK': 0, **{name: i + 1 for i, name in enumerate(ranked)}}
    (out / 'type_vocab.json').write_text(json.dumps(vocab, indent=2) + '\n')
    (out / 'split_config.json').write_text(json.dumps(dict(dates=dates, top_types=top_types), indent=2) + '\n')
    return names, vocab


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output-dir', type=Path, default=Path('data/splits'))
    p.add_argument('--dates-json', type=Path)
    p.add_argument('--top-types', type=int, default=31)
    args = p.parse_args()
    dates = json.loads(args.dates_json.read_text()) if args.dates_json else None
    names, vocab = create_splits(args.processed_dir, args.output_dir, dates, args.top_types)
    print(json.dumps(dict(scenarios={s: len(n) for s, n in names.items()}, vocabulary_size=len(vocab))))


if __name__ == '__main__':
    main()
