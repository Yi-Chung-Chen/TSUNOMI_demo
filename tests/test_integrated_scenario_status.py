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


def incident_fixture():
    rows=[dict(time_s=float(i),objects=[obj(score)]) for i,score in enumerate([11,11,9,None])]
    positions=np.array([[[20,20,10],[5,5,10],[20,20,10],[np.nan,np.nan,np.nan]]])
    return current_only_timeline(rows,positions,dict(cutoff_raw=10,zone=ZONE)),positions


def test_confirmed_intrusion_latches_scene_after_exit_normalisation_and_missing_data():
    from src.integrated.scenario_status import latch_intrusions
    rows,positions=incident_fixture();original=deepcopy(rows)
    latched=latch_intrusions(rows,positions,ZONE)
    assert [r['scenario']['incident_latched'] for r in latched]==[False,True,True,True]
    assert all(r['scenario']['status']=='High threat' for r in latched[1:])
    assert 'has left' in latched[2]['scenario']['reasons'][0]
    assert 'unavailable' in latched[3]['scenario']['reasons'][0]
    assert latched[2]['objects'][0]['anomalous'] is False
    assert latched[3]['objects'][0]['anomalous'] is False
    assert latched[-1]['scenario']['incidents'][0]['first_detected_s']==1
    assert rows==original


def test_forecast_only_and_normal_observed_entry_do_not_latch():
    from src.integrated.scenario_status import latch_intrusions
    positions=np.array([[[20,20,10],[5,5,10]]])
    rows=[dict(time_s=0.,objects=[obj(11,32)]),dict(time_s=1.,objects=[obj(5,32)])]
    rows=current_only_timeline(rows,positions,dict(cutoff_raw=10,zone=ZONE))
    result=latch_intrusions(rows,positions,ZONE)
    assert not any(r['scenario']['incident_latched'] for r in result)
    assert result[0]['scenario']['status']=='Potential intrusion'
    assert result[1]['scenario']['status']=='Normal traffic'


def test_incident_memory_resets_per_scenario_and_prefix_is_causal():
    from src.integrated.scenario_status import latch_intrusions
    rows,positions=incident_fixture();full=latch_intrusions(rows,positions,ZONE)
    prefix=latch_intrusions(rows[:1],positions[:,:1],ZONE)
    assert prefix==full[:1] and prefix[0]['scenario']['incidents']==[]
    reset=latch_intrusions(rows[2:3],positions[:,2:3],ZONE)
    assert reset[0]['scenario']['status']=='Normal traffic'
    assert not reset[0]['scenario']['incident_latched']
