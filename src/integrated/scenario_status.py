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
