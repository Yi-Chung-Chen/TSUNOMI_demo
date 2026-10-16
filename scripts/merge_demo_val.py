"""Merge existing date indexes byte-for-byte, without rebuilding trajectories."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

PROTECTED = ['data/splits/train_manifest.txt', 'data/splits/train_windows.csv', 'data/splits/type_vocab.json',
             'assets/motion_dict_K2048_s6.pkl', 'outputs/air_pre_bc/checkpoints/epoch_016.ckpt']


def protected_hashes(root=Path('.')):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in PROTECTED}


def merge(root=Path('.')):
    split = root / 'data/splits'
    receipt = split / 'merge_receipt.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        assert protected_hashes(root) == result['protected_sha256']
        assert hashlib.sha256((split / 'val_windows.csv').read_bytes()).hexdigest() == result['val_windows_sha256']
        return result
    before = protected_hashes(root)
    source = {name: (split / name).read_bytes() for name in ['val_manifest.txt', 'test_manifest.txt', 'val_windows.csv', 'test_windows.csv', 'split_config.json']}
    val_names, test_names = [source[f'{s}_manifest.txt'].decode().splitlines() for s in ['val', 'test']]
    assert len(val_names) == len(test_names) == 91 and not set(val_names) & set(test_names)
    val_rows, test_rows = [list(csv.DictReader(io.StringIO(source[f'{s}_windows.csv'].decode()))) for s in ['val', 'test']]
    assert len(val_rows) == 488 and len(test_rows) == 424
    assert len({(r['scenario'], r['start_frame']) for r in val_rows + test_rows}) == 912
    archive = split / 'legacy_date_splits'
    archive.mkdir(exist_ok=True)
    for name, content in source.items():
        path = archive / name
        if path.exists():
            assert path.read_bytes() == content
        else:
            path.write_bytes(content)
    header, body = source['test_windows.csv'].split(b'\n', 1)
    assert header == source['val_windows.csv'].split(b'\n', 1)[0]
    (split / 'val_manifest.txt').write_bytes(source['val_manifest.txt'] + source['test_manifest.txt'])
    (split / 'val_windows.csv').write_bytes(source['val_windows.csv'] + body)
    config = json.loads(source['split_config.json'])
    config['dates']['val'] = ['2026-10-03', '2026-10-04']
    config['dates'].pop('test')
    (split / 'split_config.json').write_text(json.dumps(config, indent=2) + '\n')
    # Retire active test names; original bytes remain in the ignored archive.
    for name in ['test_manifest.txt', 'test_windows.csv']:
        (split / name).unlink()
    assert protected_hashes(root) == before
    result = dict(val_scenarios=182, val_windows=912, original_windows={'2026-10-03': 488, '2026-10-04': 424},
                  protected_sha256=before, val_windows_sha256=hashlib.sha256((split / 'val_windows.csv').read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256(content).hexdigest() for name, content in source.items()},
                  note='Merged existing rows only; no ingest, window rebuild, dictionary build or training.')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(merge(), indent=2))
