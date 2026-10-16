from pathlib import Path
import json
import pickle
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader
from src.data.dataset import AirDataset
from src.tokens.token_processor import AirTokenProcessor
from src.common.units import to_internal, to_metres
from src.preprocess.motion_dict_eval import load_validation, evaluate

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'assets/motion_dict_K2048_s6.pkl'


@pytest.fixture(scope='module')
def processor():
    torch.set_num_threads(1)
    return AirTokenProcessor(ASSET)


def test_variable_agent_collation_and_output_shapes(processor):
    dataset = AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')
    indices = [0, len(dataset) - 1]
    items = [dataset[i] for i in indices]
    n = sum(item['agent'].num_nodes for item in items)
    batch = next(iter(DataLoader(items, batch_size=2)))
    assert batch['agent']['position'].shape == (n, 109, 3)
    assert batch['agent']['batch'].unique().tolist() == [0, 1]
    empty, out = processor(batch)
    assert empty == {}
    shapes = dict(gt_idx=(n, 19), gt_pos=(n, 19, 3), gt_heading=(n, 19), gt_z=(n, 19),
                  gt_z_raw=(n, 19), gt_dz=(n, 19), gt_pos_raw=(n, 19, 3),
                  gt_heading_raw=(n, 19), gt_dense=(n, 108, 3), valid_mask=(n, 19),
                  sampled_idx=(n, 19), sampled_pos=(n, 19, 3), sampled_heading=(n, 19),
                  sampled_z=(n, 19), on_ground=(n, 19), type=(n,), train_mask=(n,), batch=(n,))
    assert set(shapes) == set(out)
    for key, shape in shapes.items():
        assert out[key].shape == shape, key
    assert out['train_mask'].sum() >= 2


def test_mid_token_gap_is_invalid(processor):
    p = torch.zeros((1, 13, 3))
    h = torch.zeros((1, 13))
    v = torch.ones((1, 13), dtype=torch.bool)
    v[:, 3] = False
    g = torch.zeros_like(v)
    out = processor.reconstruct(p, h, v, g)
    assert out['valid_mask'].tolist() == [[True, False, True]]


def test_torch_roundtrip_reproduces_m3_all_groups_and_horizons(processor):
    data = load_validation(ROOT / 'data/processed', ROOT / 'data/splits')
    p, h, g, v = data
    with ASSET.open('rb') as stream:
        dictionary = pickle.load(stream)
    for one_shot in [False, True]:
        expected_rows = evaluate(dictionary, data, autoregressive=not one_shot)
        for horizon in [10, 30]:
            end = 18 + horizon * 3
            take = v[:, 18:end + 1].all(-1)
            x = torch.from_numpy(to_internal(p[take].astype(np.float64), 10.))
            out = processor.reconstruct(x, torch.from_numpy(h[take]), torch.from_numpy(v[take]),
                                        torch.from_numpy(g[take]), start_frame=18,
                                        horizon_frames=horizon * 3, one_shot=one_shot)
            # Detokenise independently, anchoring exactly as in the evaluator.
            prev_p, prev_h = x[:, 18], torch.from_numpy(h[take, 18]).double()
            if processor.meta['heading_source'] == 'chord':
                from src.common.torch_geometry import causal_headings
                hh = causal_headings(x, torch.from_numpy(h[take]).double(), torch.from_numpy(v[take]), torch.from_numpy(g[take]), 6, 10.)
                prev_h = hh[:, 18]
            dense = []
            for step in range(1, out['idx'].shape[1]):
                if one_shot:
                    begin = 18 + (step - 1) * 6
                    prev_p, prev_h = x[:, begin], hh[:, begin] if processor.meta['heading_source'] == 'chord' else torch.from_numpy(h[take, begin])
                decoded, headings = processor.decode(out['idx'][:, step], prev_p, prev_h)
                dense.append(decoded[:, 1:])
                prev_p, prev_h = decoded[:, -1], headings[:, -1]
            np.testing.assert_allclose(torch.cat(dense, 1).numpy(), out['dense_position'].numpy(), atol=1e-5)
            error = to_metres(out['dense_position'].numpy(), 10.) - p[take, 19:end + 1]
            horiz, vert = np.linalg.norm(error[..., :2], axis=-1), np.abs(error[..., 2])
            for row in expected_rows:
                if row['horizon_s'] != horizon:
                    continue
                mask = np.ones(take.sum(), bool) if row['group'] == 'all' else g[take, 18] == (row['group'] == 'ground')
                assert row['n_agents'] == int(mask.sum())
                for axis, errors in [('horizontal', horiz), ('vertical', vert)]:
                    assert float(errors[mask].mean()) == pytest.approx(row[f'{axis}_ade_m'], abs=1e-3)
                    assert float(errors[mask, -1].mean()) == pytest.approx(row[f'{axis}_fde_m'], abs=1e-3)


def test_history_tokens_are_invariant_to_future(processor):
    ds = AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')
    data = ds[0]
    first = processor.tokenize_agent(data)
    data['agent']['position'][:, 19:] += 10000
    data['agent']['heading'][:, 19:] += 1.
    second = processor.tokenize_agent(data)
    for key in ['gt_idx', 'gt_pos', 'gt_heading', 'sampled_idx', 'sampled_pos']:
        torch.testing.assert_close(first[key][:, :4], second[key][:, :4])


def test_sampled_path_has_valid_shapes_and_deterministic_gt():
    torch.set_num_threads(1)
    proc = AirTokenProcessor(ASSET, num_k=4, temp=.5)
    ds = AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')
    data = ds[0]
    first = proc.tokenize_agent(data)
    second = proc.tokenize_agent(data)
    torch.testing.assert_close(first['gt_idx'], second['gt_idx'])
    assert first['sampled_pos'].shape == first['gt_pos'].shape
    assert torch.isfinite(first['sampled_pos']).all()


def test_default_chain_ids_decode_to_matched_states(processor):
    """Default frame0 chain roundtrip; raw-current M3 parity is tested separately."""
    ds = AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')
    data = ds[0]
    out = processor.tokenize_agent(data)
    a = data['agent']
    prev_p, prev_h = out['gt_pos'][:, 0], out['gt_heading'][:, 0]
    for step in range(1, 19):
        decoded, headings = processor.decode(out['gt_idx'][:, step], prev_p, prev_h)
        valid = out['valid_mask'][:, step]
        torch.testing.assert_close(decoded[valid, -1], out['gt_pos'][valid, step], atol=1e-5, rtol=1e-7)
        torch.testing.assert_close(headings[valid, -1], out['gt_heading'][valid, step], atol=1e-5, rtol=1e-7)
        # Invalid context tokens are explicitly reset, matching processor semantics.
        prev_p, prev_h = out['gt_pos'][:, step], out['gt_heading'][:, step]
