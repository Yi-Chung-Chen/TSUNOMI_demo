from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pytest
from src.eval.context_density import ContextDensity,context_from_utc


def utc(local):return datetime.fromisoformat(local).replace(tzinfo=ZoneInfo('America/Chicago')).timestamp()


def test_context_uses_chicago_time_boundaries_and_dst_not_utc_or_purdue():
    ts=[utc(s) for s in ['2026-10-01T06:59:59','2026-10-01T07:00:00','2026-10-01T18:59:59','2026-10-01T19:00:00','2026-12-01T07:00:00']]
    np.testing.assert_array_equal(context_from_utc(ts),[1,0,0,1,0])
    with pytest.raises(ValueError):context_from_utc(np.nan)


def states(n,speed):
    return dict(position=np.tile([100.,100.,30.],(n,1)),speed=np.full(n,speed),course=np.zeros(n))


def test_identical_state_rarity_depends_on_train_context_and_probabilities_normalize():
    m=ContextDensity([[0,0],[500,500]])
    m.add(states(1000,35),utc('2026-10-01T12:00:00'))
    m.add(states(1000,5),utc('2026-10-01T23:00:00'));m.finish()
    x=states(1,35)
    assert m.score_context(x,0)['state_nll'][0]<m.score_context(x,1)['state_nll'][0]
    for model in m.models:
        assert np.isclose(model.p_joint.sum(),1)
        assert np.allclose(model.p_speed.sum(-1),1)
        assert np.allclose(model.p_course.sum(-1),1)
    # No support outside the grid: local conditional distributions are exactly global.
    np.testing.assert_allclose(m.models[0].p_speed[-1],m.global_model.p_speed[-1])
    np.testing.assert_allclose(m.models[1].p_course[-1],m.global_model.p_course[-1])


def test_entirely_missing_context_falls_back_to_global_and_mixed_batch_is_correct():
    m=ContextDensity([[0,0],[500,500]]);m.add(states(20,35),utc('2026-10-01T12:00:00'));m.finish()
    for key in ['p_joint','p_speed','p_course']:
        np.testing.assert_allclose(getattr(m.models[1],key),getattr(m.global_model,key))
    x=states(2,35);score=m.score_context(x,[0,1])
    for i in [0,1]:assert score['state_nll'][i]==m.models[i].score(states(1,35))['state_nll'][0]
    with pytest.raises(ValueError):m.score_context(x,[0,2])


def test_saved_model_round_trip_preserves_context_scores(tmp_path):
    m=ContextDensity([[0,0],[500,500]])
    m.add(states(30,35),utc('2026-10-01T12:00:00'))
    m.add(states(15,5),utc('2026-10-01T23:00:00'));m.finish()
    m.save(tmp_path/'density.npz');loaded=ContextDensity.load(tmp_path/'density.npz')
    for label in [0,1]:
        for key,values in m.score_context(states(5,35),label).items():
            np.testing.assert_array_equal(values,loaded.score_context(states(5,35),label)[key])
