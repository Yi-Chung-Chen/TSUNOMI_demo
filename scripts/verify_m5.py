"""Accept real CUDA smoke/full BC; distinguish best and final epoch weights."""
import csv
import json
import math
from pathlib import Path
import torch


def completed_epochs(rows, epochs=128):
    validation = {int(float(r['epoch'])): r for r in rows if r.get('val/loss')}
    training = {int(float(r['epoch'])): r for r in rows if r.get('train/loss')}
    assert set(validation) == set(training) == set(range(epochs)), 'Incomplete train/val epoch history'
    final_step = max(int(float(r['step'])) for r in rows if r.get('step')) + 1
    assert final_step == 98304, f'Unexpected completed step count: {final_step}'
    return validation, final_step


def epoch_metrics(row):
    metrics = {name: float(row[name]) for name in ['val/loss', 'val/next_token_accuracy',
                                                'val/open_loop_minADE_30s_m', 'val/open_loop_agents_30s']}
    assert all(math.isfinite(v) for v in metrics.values())
    assert int(metrics['val/open_loop_agents_30s']) == 668
    return metrics


def main():
    out = Path('outputs/m5_execution')
    overfit = json.loads(Path('outputs/m5_overfit/result.json').read_text())
    assert overfit['history'][-1]['ce'] < .03 and overfit['history'][-1]['accuracy'] >= .99
    gpu = json.loads((out / 'gpu.json').read_text())
    full_gpu = json.loads((out / 'gpu_full.json').read_text())
    assert gpu['physical'] in [1, 3] and full_gpu['physical'] in [1, 3]
    assert not gpu['occupied'] and not full_gpu['occupied']
    for name, chosen in [('mapping_smoke.json', gpu), ('mapping_bc.json', full_gpu)]:
        mapping = json.loads((out / name).read_text())
        assert mapping['verified_uuid'] == chosen['uuid']
    for name in ['smoke.log', 'bc.log']:
        assert 'GPU available: True (cuda), used: True' in (out / name).read_text()
    assert '`max_epochs=128` reached' in (out / 'bc.log').read_text()
    smoke = torch.load('outputs/m5_smoke/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    assert smoke['epoch'] >= 0 and smoke['global_step'] >= 16
    csv_paths = sorted(Path('outputs/air_pre_bc/csv').glob('version_*/metrics.csv'))
    assert csv_paths
    with csv_paths[-1].open() as stream:
        validation, final_step = completed_epochs(list(csv.DictReader(stream)))
    best_epoch = min(validation, key=lambda e: float(validation[e]['val/loss']))
    checkpoint_path = Path(f'outputs/air_pre_bc/checkpoints/epoch_{best_epoch:03d}.ckpt')
    bc = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert bc['epoch'] == best_epoch and bc['global_step'] == int(float(validation[best_epoch]['step'])) + 1
    state = next(v for v in bc['callbacks'].values() if v.get('monitor') == 'val/loss')
    assert Path(state['best_model_path']).name == checkpoint_path.name
    assert math.isclose(float(state['best_model_score']), float(validation[best_epoch]['val/loss']), abs_tol=1e-4)
    last_path = Path('outputs/air_pre_bc/checkpoints/last.ckpt')
    final_weights_saved = not last_path.is_symlink() and torch.load(last_path, map_location='cpu', weights_only=False)['epoch'] == 127
    result = dict(smoke_physical_gpu=gpu['physical'], bc_physical_gpu=full_gpu['physical'],
                  cuda_visible_devices='1,3', overfit=overfit['history'][-1],
                  smoke_global_step=smoke['global_step'], completed_epochs=128, completed_global_steps=final_step,
                  retained_checkpoint=dict(path=str(checkpoint_path), epoch=best_epoch,
                                           global_step=bc['global_step'], metrics=epoch_metrics(validation[best_epoch])),
                  final_epoch=dict(epoch=127, weights_saved=final_weights_saved, metrics=epoch_metrics(validation[127])))
    (out / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    with Path('PROGRESS.md').open('a') as stream:
        stream.write('\n## M5 — GPU acceptance passed\n\nActual CUDA smoke and complete 128-epoch / 98304-step BC verified. The retained checkpoint is selected by best validation loss; it is distinct from the final epoch metrics.\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n\nThe original save_last=link left last.ckpt pointing at the best epoch16 checkpoint. Final epoch127 weights were not saved. Future runs use save_last=true. Best val loss at epoch16, followed by worse final validation loss, suggests overfitting. Provisional advisor decision: use best epoch16 for M6; verify full training using complete CSV epoch history and completion log. No rerun required by the plan.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
