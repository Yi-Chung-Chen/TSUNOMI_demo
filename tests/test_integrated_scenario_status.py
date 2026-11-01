from copy import deepcopy
import numpy as np
from src.integrated.scenario_status import current_only_timeline,scenario_status

ZONE=dict(polygon_enu_m=[[0,0],[10,0],[10,10],[0,10]],height_range_m=[0,100])


def obj(score=5,count=0):
    return dict(id='test',current_state_nll=score,anomalous=score is not None and score>10,
                entry=None if count is None else dict(entry_samples=count),
                explanation=[dict(factor='Speed')])


def test_current_only_removes_latch_without_mutating_original():
    rows=[dict(objects=[obj(11)]),dict(objects=[obj(9)])]
    rows[1]['objects'][0]['anomalous']=True # old latch
    original=deepcopy(rows)
    result=current_only_timeline(rows,np.array([[[20,20,10],[20,20,10]]]),dict(cutoff_raw=10,zone=ZONE))
    assert [r['objects'][0]['anomalous'] for r in result]==[True,False]
    assert result[1]['scenario']['status']=='Normal traffic'
    assert rows==original


def test_normal_object_entering_does_not_combine_with_other_anomaly():
    result=scenario_status([obj(11),obj(5,32)],[[20,20,10],[5,5,10]],ZONE)
    assert result['status']=='Unusual activity'


def test_entry_threshold_is_inclusive_and_requires_anomaly():
    assert scenario_status([obj(11,7)],[[20,20,10]],ZONE)['status']=='Unusual activity'
    assert scenario_status([obj(11,8)],[[20,20,10]],ZONE)['status']=='Potential intrusion'
    assert scenario_status([obj(10,32)],[[20,20,10]],ZONE)['status']=='Normal traffic'


def test_current_intrusion_needs_no_forecast_and_takes_priority():
    assert scenario_status([obj(11,None)],[[5,5,10]],ZONE)['status']=='Intrusion detected'
    assert scenario_status([obj(11,32)],[[5,5,200]],ZONE)['status']=='Potential intrusion'


def test_background_can_trigger_scenario_and_unknown_is_not_normal():
    result=scenario_status([obj(5),obj(11,8)],[[20,20,10],[20,20,10]],ZONE)
    assert result['status']=='Potential intrusion' and 'Aircraft 01' in result['reasons'][0]
    assert scenario_status([obj(None)],[[20,20,10]],ZONE)['status']=='Collecting observations'


def test_missing_forecast_and_subthreshold_evidence_are_explicit():
    missing=scenario_status([obj(11,None)],[[20,20,10]],ZONE)
    assert 'cannot be assessed' in missing['reasons'][0]
    few=scenario_status([obj(11,7)],[[20,20,10]],ZONE)
    assert 'threshold is not met' in few['reasons'][0]
