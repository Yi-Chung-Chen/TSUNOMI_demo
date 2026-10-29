import numpy as np
from src.integrated.story import evaluate_story


def timeline(level,zone_points,count,time):
    return dict(time_s=time,objects=[dict(current_exceeds_cutoff=True,anomalous=True,
        future_anomaly_horizon_s=10,threat=dict(level=level,score=zone_points,rule_points={'zone':zone_points}),
        entry=dict(entry_samples=count))])


def fixture():
    zone=dict(polygon_enu_m=[[0,0],[10,0],[10,10],[0,10]],height_range_m=[0,20])
    scene=dict(zone=zone,threat=dict(entry_points=60))
    stream=dict(reference=np.array([[[20,5,10],[15,5,10],[5,5,10]]]),times=np.array([0.,1.,2.]))
    return scene,stream


def test_gate_requires_model_warning_before_actual_entry_not_only_current_inside():
    scene,stream=fixture()
    result=evaluate_story(scene,stream,[timeline('High',80,32,1.),timeline('High',60,8,2.)])
    assert result['first_model_based_preentry_high_s'] is None
    result=evaluate_story(scene,stream,[timeline('High',60,8,1.)])
    assert result['first_model_based_preentry_high_s']==1 and result['model_warning_lead_s']==1
