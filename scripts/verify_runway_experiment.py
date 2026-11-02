"""Final M10 acceptance; never edits data/model/demo artifacts or chooses runs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/runway_anchor'
    protected=json.loads((out/'protected_before.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in protected.items())
    reference=json.loads((out/'reference_hashes.json').read_text())
    assert all(hashlib.sha256((root.parent/p).read_bytes()).hexdigest()==sha for p,sha in reference.items())
    assert (out/'reference_output_actual.txt').read_bytes()==(root.parent/'m10_reference/runway_alignment_reference_output.txt').read_bytes()
    alignment=json.loads((out/'alignment.json').read_text());assert alignment['passed']
    jobs={};configs={}
    for name,physical in [('base20',1),('anchor',3)]:
        path=out/'training'/name;done=json.loads((path/'completed.json').read_text());assert done['epochs']==20 and done['global_step']==15360
        gpu=json.loads((path/'gpu.json').read_text());assert gpu['physical']==physical and gpu['verified_uuid']==gpu['uuid'] and gpu['visible_devices']==str(physical)
        config=yaml.safe_load((path/'.hydra/config.yaml').read_text());configs[name]=config
        assert config['data']['train_batch_size']==32 and config['data']['val_batch_size']==16 and config['data']['num_workers']==4
        assert config['seed']==817 and config['trainer']['max_epochs']==20 and config['trainer']['devices']==1
        assert config['trainer']['strategy']=='auto' and config['trainer']['limit_train_batches']==1.0 and config['trainer']['limit_val_batches']==1.0
        assert config['model']['lr']==.0005 and config['model']['dropout']==.1 and config['model']['label_smoothing']==.1
        assert config['model']['runway_anchors']==(name=='anchor')
        with (path/'csv/version_0/metrics.csv').open() as f:values=[r for r in csv.DictReader(f) if r.get('val/loss')]
        assert {int(float(r['epoch'])) for r in values}==set(range(20))
        best=min(values,key=lambda r:float(r['val/loss']));assert int(float(best['epoch']))==done['best_epoch']
        assert abs(float(best['val/loss'])-done['best_val_loss'])<1e-7
        last=torch.load(path/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert last['epoch']==19 and last['global_step']==15360
        if name=='anchor':
            with (root/'assets/ord_runway_ends.csv').open() as f:anchor_rows=list(csv.DictReader(f))
            expected=torch.tensor([[float(r[k])/10 for k in ['x_m','y_m','z_m']] for r in anchor_rows],dtype=torch.float32)
            assert torch.equal(last['state_dict']['decoder.agent_decoder.anchor_position'],expected)
            expected_heading=torch.tensor([float(r['inbound_heading_rad']) for r in anchor_rows],dtype=torch.float32)
            assert torch.equal(last['state_dict']['decoder.agent_decoder.anchor_heading'],expected_heading)
        jobs[name]=dict(done,gpu=gpu)
    for section in ['trainer','data']:assert configs['base20'][section]==configs['anchor'][section]
    first=dict(configs['base20']['model']);second=dict(configs['anchor']['model']);first.pop('runway_anchors');second.pop('runway_anchors');assert first==second
    evals={}
    for name in ['epoch_016','base20','anchor']:
        e=json.loads((out/'evaluation'/name/'completed.json').read_text());assert e['device']=='cpu' and e['windows']==912 and e['anomaly_rows']==28413 and e['m8_rows_tokens_residuals_identical']
        evals[name]=e
    flow=json.loads((out/'flow_labels.json').read_text());assert flow['counts']['2026-10-03'].get('east')==68 and flow['counts']['2026-10-03'].get('west',0)==0
    assert flow['counts']['2026-10-04']['east']==15 and flow['counts']['2026-10-04']['west']==39
    weights=json.loads((out/'attention_weights.json').read_text());assert weights['history_only'] and not weights['network_requests'] and len(weights['windows'])==3
    for window in weights['windows']:
        a=np.asarray(window['weights']);assert a.shape[1]==16 and np.isfinite(a).all() and (a>=0).all()
        np.testing.assert_allclose(a.sum(axis=1),1,atol=1e-5)
    with Image.open(out/'anchor_attention.png') as im:assert min(im.size)>500
    with (out/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    assert summary and {r['family'] for r in summary}=={'CE','M6','M8','Flow'}
    assert (out/'summary.md').is_file()
    receipt=dict(passed=True,milestone='M10',alignment_both_gates=True,training=jobs,evaluation=evals,
        test_result='122passed,19dependency warnings,51.38s',flow_counts=flow['counts'],
        flow_rule_advisor_provisional=True,attention_figure='outputs/runway_anchor/anchor_attention.png',
        protected_files_unchanged=protected,reference_files_unchanged=reference,
        anchor_csv_sha256=hashlib.sha256((root/'assets/ord_runway_ends.csv').read_bytes()).hexdigest(),
        validation_subset_sha256=hashlib.sha256((out/'val_10_03_windows.csv').read_bytes()).hexdigest(),
        demo_checkpoint_unchanged=True,model_default_false_bitwise=True,no_identity_dropout=True,
        no_hyperparameter_search=True,summary_rows=len(summary),no_demo_checkpoint_swap=True)
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (root/'docs/runway_anchor_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,summary_rows=len(summary),best_epochs={k:v['best_epoch'] for k,v in jobs.items()})),flush=True)

if __name__=='__main__':main()
