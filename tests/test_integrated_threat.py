import json
from pathlib import Path
import numpy as np
from src.integrated.threat import AnomalyLatch,threat_level,explain_state

C=json.loads((Path(__file__).resolve().parents[1]/'configs/integrated_demo/v1.json').read_text())
ZONE=dict(polygon_enu_m=[[0,0],[10,0],[10,10],[0,10]],height_range_m=[0,20])


def entry(n):return dict(entry_samples=n,total_samples=32,earliest_entry_s=10 if n else None)


def score(anomalous=True,n=0,context='daytime',current=None,previous=None,older=None):
    current=np.array([1000.,1000,10]) if current is None else np.asarray(current,float)
    previous=current.copy() if previous is None else np.asarray(previous,float)
    older=previous.copy() if older is None else np.asarray(older,float)
    return threat_level(anomalous,12,current,previous,older,entry(n),context,ZONE,C['threat'])


def test_observation_latch_strict_cutoff_ten_second_inclusive_expiry():
    latch=AnomalyLatch(10)
    assert not latch.update(0,10)
    assert latch.update(1,10.01)
    assert latch.update(11,9)
    assert not latch.update(11.0001,9)


def test_normal_traffic_no_score_even_inside_or_future_entry_at_night():
    result=score(anomalous=False,n=32,context='nighttime',current=[5,5,10])
    assert result['level']=='Normal traffic' and result['score'] is None and result['rule_points']=={}


def test_anomalous_far_hobbyist_low_and_night_alone_does_not_add_score():
    for context in ['daytime','nighttime']:
        r=score(context=context)
        assert r['level']=='Low' and r['score']==0 and 'night' not in r['rule_points']


def test_zone_base_not_added_twice_and_no_anomaly_modifier():
    r=score(n=32,current=[5,5,10])
    assert r['level']=='High' and r['score']==80
    assert r['rule_points']=={'zone':80}
    assert score(n=8)['score']==60
    assert score(n=7)['score']==30
    assert score(n=8,context='nighttime')['score']==70


def test_observed_acceleration_toward_zone_and_traceable_rules():
    r=score(n=8,context='nighttime',current=[30,5,10],previous=[70,5,10],older=[90,5,10])
    assert r['score']==80 and r['speed_increase_mps']==10 and r['closing_mps']==20
    assert r['rule_points']==dict(zone=60,night=10,acceleration=10)
    assert len(r['reasons'])==len(r['rule_points'])


def test_unknown_not_reported_as_low_and_future_only_never_sets_latch():
    latch=AnomalyLatch(10)
    assert not latch.update(0,9)  # A forecast alert is not an input to latch.
    r=threat_level(True,12,[0,0,0],None,None,None,'nighttime',ZONE,C['threat'])
    assert r['level']=='Insufficient data' and r['score'] is None
    r=threat_level(False,np.nan,[0,0,0],None,None,None,'nighttime',ZONE,C['threat'])
    assert r['level']=='Insufficient data'


def test_explanation_terms_reconstruct_actual_component_excess():
    values=dict(location_height_nll=8,speed_nll=7,course_nll=2)
    baseline=dict(location_height_nll=6,speed_nll=1,course_nll=1)
    parts=explain_state(values,baseline)
    assert parts[0]['factor']=='Speed' and parts[0]['excess']==6
    assert sum(r['nll'] for r in parts)==17
