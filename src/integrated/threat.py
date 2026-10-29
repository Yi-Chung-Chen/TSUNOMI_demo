"""Transparent demo rules; anomaly gate is from current observations only."""
from collections import deque
import numpy as np
from src.integrated.geometry import inside_zone,distance_to_zone


class AnomalyLatch:
    def __init__(self,cutoff,window_s=10):
        self.cutoff=cutoff;self.window_s=window_s;self.exceedances=deque();self.last_time=-np.inf

    def update(self,time_s,current_state_score):
        if time_s<self.last_time:raise ValueError('Latch time must be monotonic')
        self.last_time=time_s
        while self.exceedances and self.exceedances[0]<time_s-self.window_s-1e-9:self.exceedances.popleft()
        if np.isfinite(current_state_score) and current_state_score>self.cutoff:self.exceedances.append(time_s)
        return bool(self.exceedances)


def threat_level(anomalous,current_score,current,previous,older,entry,context,zone,rules):
    """A null score means no threat computed, never a zero disguised as normal."""
    if not np.isfinite(current_score):
        return dict(level='Insufficient data',score=None,reasons=['No valid current state score'],rule_points={})
    if not anomalous:
        return dict(level='Normal traffic',score=None,reasons=['No current anomaly in the last10s'],rule_points={})
    if entry is None or previous is None or older is None:
        return dict(level='Insufficient data',score=None,reasons=['Recent anomaly; forecast/history unavailable'],rule_points={})
    distance=float(distance_to_zone(current,zone));before=float(distance_to_zone(previous,zone))
    closing=(before-distance)/2
    speed=float(np.linalg.norm(np.asarray(current)[:2]-np.asarray(previous)[:2])/2)
    old_speed=float(np.linalg.norm(np.asarray(previous)[:2]-np.asarray(older)[:2])/2)
    increase=speed-old_speed
    points={};reasons=[];count=entry['entry_samples']
    if bool(inside_zone(current,zone)):
        points['zone']=rules['inside_points'];reasons.append('Currently inside demo protected volume')
    elif count>=rules['entry_min_samples']:
        points['zone']=rules['entry_points'];reasons.append(f"{count}/{entry['total_samples']} sampled futures enter demo zone")
    elif count>0:
        points['zone']=rules['possible_entry_points'];reasons.append(f"Possible entry: {count}/{entry['total_samples']} sampled futures")
    elif distance<rules['near_m'] and closing>rules['closing_mps']:
        points['zone']=rules['approach_points'];reasons.append('Nearby and approaching demo zone')
    else:
        points['zone']=0;reasons.append('No forecasted entry or nearby approach')
    if points['zone']>0:
        if context=='nighttime':points['night']=rules['night_points'];reasons.append('Scripted nighttime context')
        if closing>rules['closing_mps'] and increase>rules['speed_increase_mps']:
            points['acceleration']=rules['acceleration_points'];reasons.append('Observed speed increasing toward zone')
    score=int(np.clip(sum(points.values()),0,100))
    level='High' if score>=rules['high_min'] else ('Medium' if score>=rules['medium_min'] else 'Low')
    return dict(level=level,score=score,reasons=reasons,rule_points=points,
        distance_to_zone_m=distance,closing_mps=closing,speed_increase_mps=increase)


def explain_state(scores,aircraft_medians):
    """Descriptive excess versus existing aircraft medians, not causal claims."""
    labels={'location_height_nll':'Location / height','speed_nll':'Speed','course_nll':'Travel direction'}
    parts=[dict(factor=labels[key],nll=float(scores[key]),reference_median=float(aircraft_medians[key]),
        excess=float(scores[key]-aircraft_medians[key])) for key in labels]
    return sorted(parts,key=lambda r:r['excess'],reverse=True)
