from pathlib import Path
import json
import numpy as np
import torch
import yaml
from src.eval.metrics.horizon_ade import compute
from src.eval.aggregate import aggregate
from src.rollout.baselines import baselines
from src.rollout.rollout_io import save_rollout_window
from src.eval.load_rollout import load_rollout
from src.data.dataset import AirDataset
from scripts.merge_demo_val import merge, protected_hashes

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    data = dict(position=np.zeros((32, 3, 90, 3)), gt_position=np.zeros((3, 109, 3)),
                valid_mask=np.ones((3, 109), bool), target_mask=np.ones(3, bool),
                on_ground=np.zeros((3, 109), bool), date=np.asarray('2026-10-03'))
    data['on_ground'][1] = True
    data['valid_mask'][2, 25] = False  # interior gap, endpoints remain valid
    return data


def test_known_errors_and_contiguous_horizon_masks():
    d = _fixture()
    d['position'][1:, 0, :, 0] = 10.
    d['position'][:, 1, :, 0] = 20.
    d['position'][..., 2] = 5.
    rows = compute(d)
    row = next(r for r in rows if r['horizon_s'] == 10 and r['group'] == 'all' and r['axis'] == 'horizontal')
    assert row['n_agents'] == 2
    assert row['minADE_m'] == 10.
    assert row['ADE_m'] == row['FDE_m'] == (31/32*10 + 20)/2
    assert all(r['n_agents'] == 2 for r in rows if r['group'] == 'all')
    vertical = next(r for r in rows if r['horizon_s'] == 30 and r['group'] == 'all' and r['axis'] == 'vertical')
    assert vertical['minADE_m'] == vertical['ADE_m'] == vertical['FDE_m'] == 5.


def test_miss_rate_is_all_rollouts_miss_and_strict_threshold():
    d = _fixture()
    d['position'][:, :2, :, 0] = 600.
    d['position'][0, 0, :, 0] = 500.
    row = next(r for r in compute(d) if r['horizon_s'] == 30 and r['group'] == 'all' and r['axis'] == 'horizontal')
    assert row['miss_rate'] == .5
    assert row['mean_rollout_miss_rate'] == (31/32 + 1)/2


def test_aggregation_weights_agents_not_windows():
    rows = []
    for date, count, value in [('2026-10-03', 1, 1.), ('2026-10-04', 9, 9.)]:
        rows.append(dict(model='bc', date=date, group='all', horizon_s=10, axis='horizontal', n_agents=count,
                         minADE_m=value, ADE_m=value, FDE_m=value, miss_rate=None, mean_rollout_miss_rate=None))
    row = next(r for r in aggregate(rows) if r['date'] == 'combined')
    assert row['n_agents'] == 10 and row['ADE_m'] == 8.2


def test_cv_linear_3d_motion_and_exact_gt_replay(tmp_path):
    graph = AirDataset(ROOT/'data/processed', ROOT/'data/splits', 'val')[0]
    n = graph['agent'].num_nodes
    metres = torch.arange(109, dtype=torch.float64)[:, None] * torch.tensor([1., -2., 3.])
    graph['agent']['position'] = (metres[None].repeat(n, 1, 1) / 10.)
    graph['agent']['valid_mask'][:] = True
    graph['agent']['train_mask'][:] = True
    for name, pred in baselines(graph).items():
        data = load_rollout(save_rollout_window(graph, pred, tmp_path / f'{name}.npz', name, 0))
        for row in compute(data):
            assert all(row[k] in [0., None] for k in ['minADE_m', 'ADE_m', 'FDE_m', 'miss_rate'])


def test_merge_receipt_protected_artifacts_and_no_rebuild():
    receipt = merge(ROOT)
    assert receipt['val_windows'] == 912 and receipt['val_scenarios'] == 182
    assert receipt['protected_sha256'] == protected_hashes(ROOT)
    split = ROOT/'data/splits'
    archive = split/'legacy_date_splits'
    assert (split/'val_manifest.txt').read_bytes() == (archive/'val_manifest.txt').read_bytes() + (archive/'test_manifest.txt').read_bytes()
    assert (split/'val_windows.csv').read_bytes() == (archive/'val_windows.csv').read_bytes() + (archive/'test_windows.csv').read_bytes().split(b'\n',1)[1]
    assert not (split/'test_manifest.txt').exists()
    assert yaml.safe_load((ROOT/'configs/experiment/air_pre_bc.yaml').read_text())['trainer']['max_epochs'] == 20


def test_plot_selection_is_seeded_and_ignores_prediction_errors():
    from src.eval.demo_plots import choose_windows
    paths = sorted((ROOT/'outputs/demo_smoke/demo_rollouts/bc').glob('*.npz'))
    if not paths:
        import pytest
        pytest.skip('Run demo smoke to validate saved-plot selection')
    first = choose_windows(paths, 0)
    assert first == choose_windows(paths, 0)
    assert len({r['path'] for r in first}) == len(first) == 6


def test_minade_selects_a_whole_trajectory_not_best_frame():
    d = _fixture()
    d['target_mask'][1:] = False
    d['position'][:, 0, :, 0] = 20.
    d['position'][0, 0, :45, 0] = 0.
    d['position'][0, 0, 45:, 0] = 10.
    d['position'][1, 0, :45, 0] = 10.
    d['position'][1, 0, 45:, 0] = 0.
    row = next(r for r in compute(d) if r['horizon_s'] == 30 and r['group'] == 'all' and r['axis'] == 'horizontal')
    assert row['n_agents'] == 1 and row['minADE_m'] == 5.
    assert row['FDE_m'] == (10. + 30 * 20.) / 32
