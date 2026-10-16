import pytest
from scripts.verify_m5 import completed_epochs


def _history():
    rows = []
    for epoch in range(128):
        step = (epoch + 1) * 768 - 1
        rows += [dict(epoch=str(epoch), step=str(step), **{'val/loss': '1.0'}),
                 dict(epoch=str(epoch), step=str(step), **{'train/loss': '1.0'})]
    return rows


def test_full_epoch_evidence_independent_of_best_checkpoint_epoch():
    validation, steps = completed_epochs(_history())
    assert len(validation) == 128 and steps == 98304


def test_missing_epoch_cannot_pass_full_training_gate():
    with pytest.raises(AssertionError, match='Incomplete'):
        completed_epochs([r for r in _history() if r['epoch'] != '50'])
