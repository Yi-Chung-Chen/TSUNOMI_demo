from copy import deepcopy
import json

import numpy as np
import pytest

from src.eval.anomaly_state import StateDensity,states_at,standardize,fit_density


def fixture_data():
    p=np.zeros((2,20,3),dtype=float)
    p[:,:,0]=np.arange(20)[None]*8
    p[:,:,1]=300.;p[:,:,2]=150.
    valid=np.ones((2,20),bool);valid[0,7]=False
    return dict(pos=p,valid=valid)


def test_counts_equal_eligible_train_states_and_only_train_files(tmp_path):
    processed=tmp_path/'data/processed';split=tmp_path/'data/splits'
    processed.mkdir(parents=True);split.mkdir(parents=True)
    (split/'train_manifest.txt').write_text('train_scenario\n')
    d=fixture_data();d.update(date=np.asarray('2026-09-28'),
        bounds_json=np.asarray(json.dumps(dict(south=41.94,north=42.015,west=-87.95,east=-87.845))))
    np.savez(processed/'train_scenario.npz',**d)
    # Deliberately broken held-out/drone files must never be read during fit.
    (processed/'held_out.npz').write_text('do not read this')
    model,receipt=fit_density(tmp_path)
    expected=sum(bool(d['valid'][a,t] and d['valid'][a,t-6]) for a in range(2) for t in range(6,20))
    assert model.joint.sum()==model.speed.sum()==model.course.sum()==expected==receipt['raw_eligible_states']
    assert receipt['dates']=={'2026-09-28':1}


def test_all_joint_conditional_distributions_normalize_after_pooling():
    data=fixture_data();a,t=np.nonzero(data['valid'][:,6:] & data['valid'][:,:-6])
    model=StateDensity([[0,0],[1750,1750]])
    model.add(states_at(data,a,t+6));model.finish()
    np.testing.assert_array_equal(model.pooled_speed.sum(-1),model.pooled_joint)
    np.testing.assert_array_equal(model.pooled_course.sum(-1),model.pooled_joint)
    assert np.isclose(model.p_joint.sum(),1.)
    np.testing.assert_allclose(model.p_speed.sum(-1),1.)
    np.testing.assert_allclose(model.p_course.sum(-1),1.)
    # Outside bucket is independent of boundary neighbours, with uniform
    # conditional distributions when there is no out-of-bounds train traffic.
    np.testing.assert_array_equal(model.pooled_joint[-1],model.joint[-1])
    np.testing.assert_allclose(model.p_speed[-1],1/14)
    np.testing.assert_allclose(model.p_course[-1],1/12)


def test_empty_far_corner_scores_higher_than_busy_cell():
    model=StateDensity([[0,0],[1750,1750]])
    train=dict(position=np.tile([800.,800.,150.],(1000,1)),speed=np.full(1000,24.),course=np.zeros(1000))
    model.add(train);model.finish()
    probe=dict(position=np.array([[800.,800.,150.],[20.,20.,150.]]),speed=np.array([24.,24.]),course=np.zeros(2))
    scores=model.score(probe)
    assert scores['location_height_nll'][1]>scores['location_height_nll'][0]
    assert scores['state_nll'][1]>scores['state_nll'][0]


def test_state_speed_course_and_score_are_causal():
    data=fixture_data();states=states_at(data,[1],[12])
    model=StateDensity([[0,0],[1750,1750]]);model.add(states);model.finish()
    changed=deepcopy(data);changed['pos'][:,13:]+=1e5
    after=states_at(changed,[1],[12])
    for key in states:
        np.testing.assert_array_equal(states[key],after[key])
    np.testing.assert_array_equal(model.score(states)['state_nll'],model.score(after)['state_nll'])
    assert states['speed'][0]==24. and states['course'][0]==0.
    changed['valid'][1,6]=False
    with pytest.raises(ValueError,match='Invalid state endpoint'):
        states_at(changed,[1],[12])


def test_standardization_uses_only_10_03_both_ground_groups():
    rows=[dict(source='aircraft',date='2026-10-03',on_ground=bool(i%2),transition_nll=float(v),state_nll=2.*v)
          for i,v in enumerate([1,2,4,8])]
    rows += [dict(source='aircraft',date='2026-10-04',on_ground=False,transition_nll=1e6,state_nll=1e9),
             dict(source='drone',date='2026-09-28',on_ground=False,transition_nll=-100.,state_nll=1e8)]
    ref=standardize(rows)
    altered=deepcopy(rows)
    for row in altered[4:]:
        row['transition_nll']=1e12;row['state_nll']=-1e12
    assert standardize(altered)==ref
    assert ref['count']==4 and ref['scores']['transition_nll']['median']==3.
    assert all(r['combined_score']==max(r['z_transition'],r['z_state']) for r in rows)
    constant=[dict(source='aircraft',date='2026-10-03',transition_nll=1.,state_nll=2.)]*4
    with pytest.raises(ValueError,match='IQR'):
        standardize(constant)


def test_fixed_bin_boundaries_and_outside_cell():
    model=StateDensity([[0,0],[510,510]])
    states=dict(position=np.array([[0,0,14.9],[510,510,15.],[-1,0,100.],[1,1,300.],[1,1,600.]]),
        speed=np.array([0.,5.,10.,120.,130.]),course=np.array([-np.pi,np.pi,0.,0.,0.]))
    cell,height,speed,course=model.bins(states)
    assert list(height)==[0,1,2,3,4]
    assert list(speed)==[0,1,2,13,13]
    assert list(course)==[0,0,6,6,6]
    assert cell[0]==0 and cell[1]==8 and cell[2]==9
