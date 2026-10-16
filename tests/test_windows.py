import csv
import json
from pathlib import Path
import numpy as np
import pytest
from src.preprocess.windows import target_mask
from src.preprocess.split import create_splits


def test_target_requires_contiguous_history_and_future():
    valid = np.ones((4, 109), bool)
    valid[1, 8] = False
    valid[2, 33] = False
    valid[3, 49:] = False
    np.testing.assert_array_equal(target_mask(valid), [True, False, False, True])


def test_train_only_vocabulary_and_date_split(tmp_path):
    data = tmp_path / 'processed'
    data.mkdir()
    for i, (day, kind) in enumerate([('2026-09-28', 'B738'), ('2026-10-03', 'VAL_ONLY'), ('2026-10-04', 'TEST_ONLY')]):
        np.savez(data / f's{i}.npz', date=day, agent_type=[kind])
    names, vocab = create_splits(data, tmp_path / 'splits')
    assert names == {'train': ['s0'], 'val': ['s1', 's2']}
    assert vocab == {'UNK': 0, 'B738': 1}
    with pytest.raises(ValueError, match='overlap'):
        create_splits(data, tmp_path / 'bad', dict(train=['x'], val=['x']))


def test_all_generated_windows():
    root = Path(__file__).resolve().parents[1]
    split_dir = root / 'data/splits'
    if not split_dir.exists():
        pytest.skip('Run split and windows CLIs for integration acceptance')
    sets = []
    for split in ['train', 'val']:
        scenarios = set((split_dir / f'{split}_manifest.txt').read_text().splitlines())
        sets.append(scenarios)
        cache = {}
        with (split_dir / f'{split}_windows.csv').open() as stream:
            for row in csv.DictReader(stream):
                name, start = row['scenario'], int(row['start_frame'])
                assert name in scenarios
                assert 0 <= start and start + 109 <= 901
                if name not in cache:
                    with np.load(root / 'data/processed' / f'{name}.npz') as d:
                        cache[name] = d['valid']
                n = target_mask(cache[name][:, start:start + 109]).sum()
                assert n >= 1 and n == int(row['n_targets'])
    assert not sets[0] & sets[1]
    assert [len(s) for s in sets] == [618, 182]
    assert len(json.loads((split_dir / 'type_vocab.json').read_text())) == 32
