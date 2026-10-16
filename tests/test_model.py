from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src.data.dataset import AirDataset
from src.model.air_smart import AirSMART
from src.metrics.cross_entropy import future_token_mask, token_cross_entropy

ROOT = Path(__file__).resolve().parents[1]


def _model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    return AirSMART(ROOT / 'assets/motion_dict_K2048_s6.pkl', dropout=0.).eval()


def test_causal_logits_and_graph_isolation():
    model = _model()
    ds = AirDataset(ROOT / 'data/processed', ROOT / 'data/splits', 'val')
    a = ds[0]
    with torch.no_grad():
        baseline, _ = model(a)
        batch = next(iter(DataLoader([a, ds[-1]], batch_size=2)))
        batched, _ = model(batch)
        torch.testing.assert_close(baseline, batched[:a['agent'].num_nodes], atol=1e-5, rtol=1e-5)
        a['agent']['position'][:, 19:] += 1000
        a['agent']['on_ground'][:, 19:] = ~a['agent']['on_ground'][:, 19:]
        changed, _ = model(a)
        torch.testing.assert_close(baseline[:, :4], changed[:, :4], atol=1e-5, rtol=1e-5)


def test_future_loss_stops_at_gap_and_ignores_history():
    valid = torch.ones((1, 19), dtype=torch.bool)
    valid[:, 6] = False
    tokens = dict(valid_mask=valid, train_mask=torch.tensor([True]), gt_idx=torch.zeros((1, 19), dtype=torch.long))
    mask = future_token_mask(tokens, 3)
    assert mask.nonzero().tolist() == [[0, 3], [0, 4]]
    logits = torch.zeros((1, 19, 4), requires_grad=True)
    result = token_cross_entropy(logits, tokens, 3, 0.)
    result['loss'].backward()
    assert result['count'] == 2
    assert logits.grad[:, :3].abs().sum() == 0
    assert logits.grad[:, 5:].abs().sum() == 0
    assert logits.grad[:, 3:5].abs().sum() > 0


def test_spatial_encoding_retains_altitude():
    model = _model()
    decoder = model.decoder.agent_decoder
    pos = torch.tensor([[0., 0., 0.], [0., 0., 10.]])
    head = torch.zeros(2)
    features = decoder.relative_features(pos, head, torch.tensor([[0], [1]]), 1, False)
    assert features[0, 2] == -10.
