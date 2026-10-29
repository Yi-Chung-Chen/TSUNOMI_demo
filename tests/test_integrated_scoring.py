from pathlib import Path
import numpy as np
from src.integrated.scoring import FrozenStateScorer,future_anomaly_evidence

ROOT=Path(__file__).resolve().parents[1]


def test_future_speed_and_course_use_each_own_path_and_observed_anchor():
    scorer=FrozenStateScorer(ROOT/'outputs/anomaly_state_smoothed')
    current=np.array([[100.,200.,110.]])
    pred=np.tile(current[None,:,None,:],(2,1,90,1));t=np.arange(1,91)/3
    pred[0,0,:,0]+=20*t;pred[1,0,:,1]-=30*t
    result=scorer.futures(pred,current)
    for r,velocity in enumerate([[20,0], [0,-30]]):
        positions=pred[r,0,5::6]
        expected=scorer.model.score(dict(position=positions,speed=np.full(15,np.linalg.norm(velocity)),
            course=np.full(15,np.arctan2(velocity[1],velocity[0]))))
        np.testing.assert_array_equal(result['state_nll'][r,0],expected['state_nll'])
    assert np.isclose(scorer.z_cutoff,(scorer.cutoff-scorer.median)/scorer.iqr)


def test_future_anomaly_threshold_is_strict_and_requires_sample_count():
    scores=np.ones((32,1,15))*5;scores[:7,0,0]=11;scores[:8,0,1]=11
    scores[:,0,2]=10
    result=future_anomaly_evidence(scores,10)
    assert result['earliest_horizon_s']==[4.]
    assert result['exceedance_counts'][0][:3]==[7,8,0]


def test_missing_observed_endpoints_remain_unscored():
    scorer=FrozenStateScorer(ROOT/'outputs/anomaly_state_smoothed')
    positions=np.zeros((1,20,3));valid=np.ones((1,20),bool);valid[0,8]=False
    result=scorer.observation_trace(positions,valid)
    assert np.isnan(result['state_nll'][0,:6]).all()
    assert np.isnan(result['state_nll'][0,8]) and np.isnan(result['state_nll'][0,14])


def test_zone_height_boundaries_and_forecast_intent():
    from src.integrated.geometry import inside_zone,distance_to_zone,forecast_intent
    zone=dict(polygon_enu_m=[[0,0],[10,0],[10,10],[0,10]],height_range_m=[0,20])
    points=np.array([[0,5,10],[5,5,21],[15,5,10],[5,5,10.]])
    assert list(inside_zone(points,zone))==[True,False,False,True]
    np.testing.assert_allclose(distance_to_zone(points,zone),[0,1,5,0])
    pred=np.tile(np.array([15.,5.,10.]),(32,1,90,1));pred[:8,0,29,0]=5
    e=forecast_intent(pred,zone)[0]
    assert e['entry_samples']==8 and e['earliest_entry_s']==10
    assert e['relation']=='Forecast enters demo zone'
