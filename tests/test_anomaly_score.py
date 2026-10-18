import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData, Batch

from src.eval import anomaly_score as score
from src.model.air_smart import AirSMART
from src.data.dataset import AirDataset
from src.metrics.cross_entropy import token_cross_entropy

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model_and_graph():
    torch.set_num_threads(1)
    checkpoint=torch.load(ROOT/'outputs/air_pre_bc/checkpoints/epoch_016.ckpt',map_location='cpu',weights_only=False)
    model=AirSMART(**checkpoint['hyper_parameters']).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    graph=AirDataset(ROOT/'data/processed',ROOT/'data/splits','val')[0]
    chosen=int(torch.where(graph['agent']['train_mask'])[0][0])
    single=HeteroData();single['agent'].num_nodes=1
    for key in ['position','heading','valid_mask','on_ground','type','radius','train_mask']:
        single['agent'][key]=graph['agent'][key][chosen:chosen+1].clone()
    return model,single


def test_first_future_nll_equals_model_unsmoothed_ce(model_and_graph):
    model,graph=model_and_graph
    measured=score.score_batch(model,[graph])['nll'][0]
    with torch.inference_mode():
        logits,tokens=model(Batch.from_data_list([graph]))
        # Isolate the first trained future prediction in the model's CE helper.
        tokens['valid_mask'][:,5:]=False
        result=token_cross_entropy(logits,tokens,model.current_node,label_smoothing=0.)
    assert result['count']==1
    np.testing.assert_allclose(measured,result['unsmoothed_ce'].item(),rtol=1e-6)


def test_future_causality_and_single_agent_batching(model_and_graph):
    model,graph=model_and_graph
    changed=graph.clone()
    changed['agent']['position'][:,25:]+=12345.
    changed['agent']['heading'][:,25:]+=1.
    changed['agent']['on_ground'][:,25:]=~changed['agent']['on_ground'][:,25:]
    baseline=score.score_batch(model,[graph])
    batched=score.score_batch(model,[graph,changed])
    np.testing.assert_allclose(batched['nll'],np.repeat(baseline['nll'],2),atol=1e-5)
    np.testing.assert_array_equal(batched['chain_token'],np.repeat(baseline['chain_token'],2))
    np.testing.assert_allclose(batched['residual_horizontal_m'],np.repeat(baseline['residual_horizontal_m'],2),atol=1e-7)
    with torch.inference_mode():
        _,tokens=model(Batch.from_data_list([graph,changed]))
        _,spatial=model.decoder.agent_decoder.build_edges(tokens['valid_mask'],tokens['batch'])
    assert spatial.shape==(2,0)


def test_native_interpolant_knots_grid_height_and_heading():
    rows=[json.loads(line) for line in score.DRONE_PATH.read_text().splitlines()]
    times=np.array([r['elapsed_s'] for r in rows])
    positions=np.array([r['xyz_enu_m'] for r in rows])
    np.testing.assert_allclose(score.interpolate(times,positions,times),positions,atol=1e-6,rtol=0)
    drone,info=score.load_drone()
    assert info['native_reads']==902 and drone['pos'].shape==(1,901,3)
    assert info['invalid_grid_frames']==[0] and info['grid_valid_frames']==900
    xyz=score.interpolate(times,positions,np.arange(1,901)/3)
    np.testing.assert_allclose(drone['pos'][0,1:,2],xyz[:,2]+2250-166,atol=1e-9)
    assert not drone['on_ground'].any() and str(drone['agent_type'][0])=='UNK'
    assert info['heading_median_absolute_velocity_difference_rad']<.1
    # Branch-cut crossing must interpolate through pi, rather than zero.
    h=score.interpolate([0,1],np.unwrap([3.1,-3.1]),[.5])[0]
    assert abs(h-np.pi)<1e-8
    with pytest.raises(ValueError,match='Conflicting duplicate'):
        score.interpolate([0,0,1],[1,2,3],[.5])


def test_auroc_perfect_ties_and_pairwise_reference():
    assert score.auroc([2,3],[0,1])==1.
    assert score.auroc([1,1],[1,1,1])==.5
    assert score.auroc([0,1],[2,3])==0.
    positive=np.array([3,1,1,4]);negative=np.array([2,1,5])
    expected=np.mean((positive[:,None]>negative)+.5*(positive[:,None]==negative))
    assert score.auroc(positive,negative)==expected


def test_padding_masks_stride_and_unique_drone_tokens(monkeypatch):
    drone,_=score.load_drone()
    vocab={'UNK':0};seen=[]
    def fake_score(model,graphs,device):
        for graph in graphs:
            assert graph['agent']['valid_mask'][:,:25].all()
            assert graph['agent']['type'].item()==0
            seen.append(graph['agent']['valid_mask'].clone())
        return {key:np.zeros(len(graphs)) for key in ['nll','chain_token','residual_token','residual_horizontal_m','residual_vertical_m']}
    monkeypatch.setattr(score,'score_batch',fake_score)
    model=SimpleNamespace(token_processor=SimpleNamespace(meta={'pos_scale':10.}))
    rows=list(score.iter_scores(model,drone,'drone','drone',vocab,32,'cpu'))
    starts=[r['window_start_frame'] for r in rows]
    assert starts==list(range(6,877,6)) and len(starts)==len(set(starts))==146
    assert rows[-1]['token_start_s']==298.
    assert seen[-1][0,:25].all() and not seen[-1][0,25:].any()
    broken={key:values.copy() for key,values in drone.items()}
    broken['valid'][0,30]=False
    with pytest.raises(ValueError):
        score.solo_window(broken,0,6,vocab)
