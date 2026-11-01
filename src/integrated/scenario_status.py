"""Current-only object anomaly and scenario-level intrusion evidence.

Reinterpret saved observations and forecasts without inference or score fitting.
The original D4 timeline remains an immutable v1 acceptance artifact.
"""
from copy import deepcopy
import numpy as np
from src.integrated.geometry import inside_zone

OBJECT_NAMES = ('Drone 01', 'Aircraft 01')
FACTOR_REASONS = {
    'Location / height': 'Unusual location or height',
    'Speed': 'Unusual speed for this location',
    'Travel direction': 'Unusual direction for this location',
}


def object_reason(obj):
    """Describe the largest reference-relative component, not a causal claim."""
    parts = obj.get('explanation', [])
    if not parts:
        return 'State is uncommon in the reference aircraft data'
    return FACTOR_REASONS.get(parts[0]['factor'], 'State is uncommon in the reference aircraft data')


def scenario_status(objects, positions, zone, min_samples=8):
    """Only the same currently anomalous object's entry can trigger an alert."""
    anomalous = [i for i, obj in enumerate(objects) if obj['anomalous']]
    if not anomalous:
        valid = [obj['current_state_nll'] is not None and np.isfinite(obj['current_state_nll']) for obj in objects]
        if not all(valid):
            return dict(status='Collecting observations', reasons=['Waiting for valid current state estimates.'], objects=[])
        return dict(status='Normal traffic', reasons=['No unusual activity detected in the current observations.'], objects=[])
    inside = [i for i in anomalous if bool(inside_zone(positions[i], zone))]
    entering = [i for i in anomalous if objects[i]['entry'] is not None
                and objects[i]['entry']['entry_samples'] >= min_samples]
    selected = inside or entering or anomalous
    status = 'Intrusion detected' if inside else ('Potential intrusion' if entering else 'Unusual activity')
    reasons = []
    for i in selected:
        obj = objects[i];name = OBJECT_NAMES[i] if i < len(OBJECT_NAMES) else obj['id']
        driver = object_reason(obj).lower()
        if i in inside:
            action = 'is currently inside the protected zone'
        elif i in entering:
            action = 'is forecast to enter the protected zone'
        elif obj['entry'] is None:
            action = 'has no forecast available yet; zone entry cannot be assessed'
        elif obj['entry']['entry_samples']:
            action = 'has limited forecast support for zone entry; the warning threshold is not met'
        else:
            action = 'has no predicted entry into the protected zone'
        reasons.append(f'{name} shows {driver} and {action}.')
    unknown = [i for i, obj in enumerate(objects)
               if obj['current_state_nll'] is None or not np.isfinite(obj['current_state_nll'])]
    if unknown:
        reasons.append('Some objects have no valid current state estimate yet.')
    return dict(status=status, reasons=reasons, objects=selected)


def current_only_timeline(timeline, observed, manifest, min_samples=8):
    """Create revised replay records; retain no legacy threat score or intent field."""
    rows = deepcopy(timeline);cutoff = manifest['cutoff_raw']
    for frame, row in enumerate(rows):
        for obj in row['objects']:
            score = obj['current_state_nll']
            obj['anomalous'] = bool(score is not None and np.isfinite(score) and score > cutoff)
            obj['current_exceeds_cutoff'] = obj['anomalous']
            obj['anomaly_reason'] = object_reason(obj) if obj['anomalous'] else None
            for field in ['threat', 'intent', 'future_anomaly_horizon_s']:
                obj.pop(field, None)
        row['scenario'] = scenario_status(row['objects'], observed[:, frame], manifest['zone'], min_samples)
    return rows


def latch_intrusions(timeline,observed,zone):
    """Scenario incident memory; object anomalies remain strictly current-only.

Trigger only on a currently anomalous object's observed position inside the3D
zone. Forecast entry alone and normal traffic cannot latch an incident. Memory
lasts until the end of this scenario, even after exit, normalisation or missing
observations. A new call/scenario starts with no incidents; no future GT is read.
"""
    rows=deepcopy(timeline);incidents={};previous_time=-np.inf
    for frame,row in enumerate(rows):
        time_s=float(row['time_s'])
        if not np.isfinite(time_s) or time_s<previous_time:
            raise ValueError('Incident timeline must have finite monotonic times')
        previous_time=time_s
        for agent,obj in enumerate(row['objects']):
            key=str(obj['id'])
            if obj['anomalous'] and bool(inside_zone(observed[agent,frame],zone)) and key not in incidents:
                incidents[key]=dict(object_id=key,object_index=agent,first_detected_s=time_s)
        scenario=row['scenario'];scenario['incident_latched']=bool(incidents)
        scenario['incidents']=deepcopy(list(incidents.values()))
        if not incidents:continue
        reasons=[]
        for incident in incidents.values():
            agent=incident['object_index'];name=OBJECT_NAMES[agent] if agent<len(OBJECT_NAMES) else incident['object_id']
            point=observed[agent,frame]
            if not np.isfinite(point).all():current='Current position is unavailable; the high alert remains active.'
            elif bool(inside_zone(point,zone)):current='It is still inside; the high alert remains active.'
            else:current='It has left; the high alert remains active.'
            reasons.append(f"{name} entered the protected zone at {incident['first_detected_s']:.1f} s. {current}")
        row['scenario']=dict(status='High threat',reasons=reasons,objects=[i['object_index'] for i in incidents.values()],
            incident_latched=True,incidents=deepcopy(list(incidents.values())),current_status=scenario['status'])
    return rows
