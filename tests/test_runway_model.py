from pathlib import Path
import math
import torch
from src.model.air_smart import AirSMART
from src.rollout.closed_loop import rollout
from src.common.torch_geometry import rotate
from src.data.datamodule import AirDataModule

ROOT=Path(__file__).resolve().parents[1]


def legacy():
    return torch.load(ROOT/'outputs/runway_anchor/legacy_bitwise_fixture.pt',map_location='cpu',weights_only=False)


def model(anchor=False):
    d=legacy();m=AirSMART(**d['hparams'],runway_anchors=anchor);m.eval();return m,d


def test_disabled_is_bitwise_identical_and_loads_existing_checkpoint_strictly():
    torch.set_num_threads(1);m,d=model(False)
    c=torch.load(ROOT/'outputs/air_pre_bc/checkpoints/epoch_016.ckpt',map_location='cpu',weights_only=False)
    result=m.load_state_dict(c['state_dict'],strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert list(m.state_dict())==d['state_keys']
    with torch.no_grad():logits,_=m(d['graph'])
    assert torch.equal(logits,d['logits'])
    predicted=rollout(m,d['graph'],n_rollouts=2,seed=817)
    for key in predicted:assert torch.equal(predicted[key],d['prediction'][key])


def test_anchor_edges_count_geometry_rotation_and_fixed_buffers():
    m,_=model(True);dec=m.decoder.agent_decoder.double()
    p=torch.tensor([[[10.,20.,3.],[30.,10.,2.]],[[1.,4.,1.],[8.,12.,2.]]],dtype=torch.float64)
    h=torch.tensor([[.1,.3],[.8,1.]],dtype=torch.float64);valid=torch.tensor([[True,False],[True,True]])
    edges,features=dec.anchor_edges(p,h,valid);assert edges.shape==(2,48)
    assert not any('anchor_position' in name or 'anchor_heading' in name for name,_ in dec.named_parameters())
    angle=.71;old=dec.anchor_position.clone()
    p_rot=p.clone();p_rot[...,:2]=rotate(p[...,:2],torch.tensor(angle,dtype=p.dtype))
    dec.anchor_position[:,:2]=rotate(old[:,:2],torch.tensor(angle,dtype=p.dtype));dec.anchor_heading+=angle
    rotated_edges,rotated_features=dec.anchor_edges(p_rot,h+angle,valid)
    assert torch.equal(edges,rotated_edges)
    torch.testing.assert_close(features,rotated_features,atol=1e-9,rtol=1e-9)


def test_rollout_anchor_features_are_recomputed_from_sampled_pose_and_ignore_future_gt():
    torch.set_num_threads(1);m,d=model(True);dec=m.decoder.agent_decoder;original=dec.anchor_edges;captured=[]
    def capture(p,h,v):
        edges,features=original(p,h,v);captured.append((p.clone(),features.clone()));return edges,features
    dec.anchor_edges=capture
    first=rollout(m,d['graph'],n_rollouts=2,seed=817);before=captured[:];captured.clear()
    changed=d['graph'].clone();changed['agent']['position'][:,19:]+=10000;changed['agent']['heading'][:,19:]+=2
    second=rollout(m,changed,n_rollouts=2,seed=817)
    assert len(before)==len(captured)==15
    for (p,a),(q,b) in zip(before,captured):assert torch.equal(p,q) and torch.equal(a,b)
    for key in first:assert torch.equal(first[key],second[key])
    assert not torch.equal(before[0][0][:,-1],before[1][0][:,-1])
    assert not torch.equal(before[0][1][-16:],before[1][1][-16:])


def test_validation_override_uses_existing_1003_rows_only():
    d=AirDataModule(ROOT/'data/processed',ROOT/'data/splits',num_workers=0,
        val_windows_path=ROOT/'outputs/runway_anchor/val_10_03_windows.csv')
    d.setup('fit');assert len(d.train_dataset)==24570 and len(d.val_dataset)==488
    assert {d.val_dataset[i]['date'] for i in [0,200,487]}=={'2026-10-03'}


def test_false_matches_frozen_original_decoder_implementation_directly():
    from tests.fixtures.air_agent_decoder_pre_runway import AirAgentDecoder as Original
    torch.set_num_threads(1);m,d=model(False)
    c=torch.load(ROOT/'outputs/air_pre_bc/checkpoints/epoch_016.ckpt',map_location='cpu',weights_only=False)
    m.load_state_dict(c['state_dict'],strict=True)
    p=m.token_processor;old=Original(K=p.K,token_dim=m.token_features.shape[1]);old.eval()
    old.load_state_dict(m.decoder.agent_decoder.state_dict(),strict=True)
    with torch.no_grad():
        logits,tokens=m(d['graph']);expected=old(tokens,m.token_features)
    assert torch.equal(logits,expected)
