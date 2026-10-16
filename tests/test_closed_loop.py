from pathlib import Path
import numpy as np
import torch
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.rollout.closed_loop import rollout, history_state
from src.rollout.rollout_io import save_rollout_window
from src.eval.load_rollout import load_rollout

ROOT = Path(__file__).resolve().parents[1]


def _model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return AirSMART(ROOT / 'assets/motion_dict_K2048_s6.pkl', dropout=0.).eval()


def _graph():
    return AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')[0]


def test_rollout_seed_shapes_immutability_and_future_gt_independence(tmp_path):
    model, graph = _model(), _graph()
    before_graph = graph.clone()
    before = rollout(model, graph, n_rollouts=32, seed=13)
    again = rollout(model, graph, n_rollouts=32, seed=13)
    torch.testing.assert_close(before['position'], again['position'], rtol=0, atol=0)
    for key in ['position', 'heading', 'valid_mask', 'on_ground']:
        torch.testing.assert_close(graph['agent'][key], before_graph['agent'][key])
    changed = graph.clone()
    changed['agent']['position'][:, 19:] += 1234
    changed['agent']['heading'][:, 19:] += 2.
    changed['agent']['valid_mask'][:, 19:] = False
    changed['agent']['on_ground'][:, 19:] = ~changed['agent']['on_ground'][:, 19:]
    after = rollout(model, changed, n_rollouts=32, seed=13)
    torch.testing.assert_close(before['position'], after['position'], rtol=0, atol=0)
    assert before['position'].shape == (32, graph['agent'].num_nodes, 90, 3)
    path = save_rollout_window(graph, before, tmp_path / 'window.npz', 'bc', 13)
    saved = load_rollout(path)
    np.testing.assert_allclose(saved['position'], before['position'].numpy(), atol=.002, rtol=1e-6)
    assert str(saved['date']) == '2026-10-03'


def test_joint_rollouts_have_independent_scene_ids():
    model, graph = _model(), _graph()
    tokens = model.token_processor.tokenize_agent(graph)
    state = history_state(model, tokens, 3)
    n = len(tokens['type'])
    assert state['batch'].tolist() == [0] * n + [1] * n + [2] * n
    for r in range(3):
        torch.testing.assert_close(state['gt_pos'][r*n:(r+1)*n], tokens['gt_pos'][:, :4])
