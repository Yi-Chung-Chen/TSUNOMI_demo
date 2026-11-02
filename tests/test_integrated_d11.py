"""D11 checkpoint loading and invariant boundary regressions."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.integrated_d11 import assert_invariants
from src.integrated.forecast import load_model

ROOT=Path(__file__).resolve().parents[1]


def test_anchor_config_changes_checkpoint_and_version_only():
    old=json.loads((ROOT/'configs/integrated_demo/v1.json').read_text())
    new=json.loads((ROOT/'configs/integrated_demo/v2_anchor.json').read_text())
    assert {k for k in old if old[k]!=new[k]}=={'version','checkpoint'}
    assert new['checkpoint']=='outputs/runway_anchor/training/anchor/checkpoints/epoch_018.ckpt'


def test_loader_uses_checkpoint_hyperparameters_and_strict_state(tmp_path):
    # Both real checkpoints load without epoch-specific construction assumptions.
    for name,anchor in [('outputs/air_pre_bc/checkpoints/epoch_016.ckpt',False),
                        ('outputs/runway_anchor/training/anchor/checkpoints/epoch_018.ckpt',True)]:
        saved=torch.load(ROOT/name,map_location='cpu',weights_only=False)
        model=load_model(ROOT,name)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert ('anchor_position' in ' '.join(model.state_dict())) == anchor
        assert set(model.state_dict())==set(saved['state_dict'])
        for k,v in model.state_dict().items():assert torch.equal(v,saved['state_dict'][k]),k
    # A missing state key must fail instead of silently accepting partial weights.
    saved['state_dict'].pop(next(iter(saved['state_dict'])))
    torch.save(saved,tmp_path/'broken.ckpt')
    with pytest.raises(RuntimeError):load_model(tmp_path,'broken.ckpt')


def data():
    obj=dict(id='drone',current_state_nll=8.,z_state=2.,current_exceeds_cutoff=True,
             anomalous=True,score_components={'speed_nll':3.},explanation=[],anomaly_reason='Unusual speed',entry={'entry_samples':0})
    return dict(stream={'observed':np.ones((1,1,3))},manifest=dict(cutoff_raw=7.,cutoff_z=1.,scene={},zone={}),
                timeline=[dict(frame=0,time_s=0.,objects=[obj],scenario=dict(incident_latched=False,incidents=[],status='Unusual activity'))],
                pb=np.array([[0,0],[1,1]]),zoom=14)


def test_invariant_boundary_allows_forecast_status_changes_only():
    old=data();new=deepcopy(old)
    new['timeline'][0]['objects'][0]['entry']['entry_samples']=20
    new['timeline'][0]['scenario']['status']='Potential intrusion'
    assert_invariants(old,new)
    for mutate in [lambda d:d['stream']['observed'].fill(2),
                   lambda d:d['timeline'][0]['objects'][0].update(anomalous=False),
                   lambda d:d['timeline'][0]['objects'][0].update(current_state_nll=9),
                   lambda d:d['timeline'][0]['scenario'].update(incident_latched=True),
                   lambda d:d['manifest'].update(cutoff_raw=9)]:
        changed=deepcopy(new);mutate(changed)
        with pytest.raises(AssertionError):assert_invariants(old,changed)
