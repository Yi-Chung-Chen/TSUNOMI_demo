from copy import deepcopy
import numpy as np
import pytest
from src.integrated.visual_cleanup import integer_shares,place_label,intersects,cleanup_texts,CleanupRenderer
from src.integrated.presentation import ORANGE,BG
from tests.test_integrated_presentation import fixture
from src.integrated.future_groups import group_futures


def test_largest_remainder_is_display_only_and_sums_to100():
    groups=[dict(count=c,total_samples=32,share=c/32) for c in [14,11,6,1]]
    before=deepcopy(groups);assert integer_shares(groups)==[44,34,19,3]
    assert groups==before and sum(integer_shares(groups))==100
    assert integer_shares([dict(count=1,total_samples=3) for _ in range(3)])==[34,33,33]
    assert integer_shares([dict(count=1,total_samples=1000),dict(count=999,total_samples=1000)])==[0,100]


def test_name_marker_and_every_group_label_use_cleanup_style_without_data_changes():
    d=fixture();d.update(grouped=True,all_anomalies=True,intrusion_memory=True)
    d['manifest']['zone']['height_range_m']=[0,180]
    for obj in d['timeline'][18]['objects']:
        obj.update(anomalous=True,anomaly_reason='Unusual speed for this location')
    d['object_groups']={18:{a:group_futures(d['forecasts'][18]['position'][:,a],d['manifest']['zone']) for a in range(2)}}
    before=deepcopy(d['object_groups']);r=CleanupRenderer(d);r.frame(18)
    assert r.boxes==[]
    for i,track in enumerate(r.tracks):
        assert track[4].get_text()==('Drone 01' if i==0 else 'Aircraft 01')
        assert track[4].get_bbox_patch() is None
        assert track[2].get_markeredgecolor()==ORANGE
    assert [a[3].get_text() for a in r.group_artists if a[3].get_visible()]==['100%','100%']
    assert r.layout_receipt[18]['labels_do_not_overlap']
    assert d['object_groups']==before
    r.frame(17)
    assert all(track[2].get_markeredgecolor()==BG for track in r.tracks)
    assert not any(a[3].get_visible() for a in r.group_artists)
    r.close()


def test_object_card_has_one_line_per_object_and_no_duplicated_reason():
    d=fixture();row=d['timeline'][18];row['objects'][0].update(anomalous=True,anomaly_reason='Unusual speed for this location')
    texts=cleanup_texts(row)
    assert len(texts['objects'].split('\n\n'))==2
    assert all('\n' not in line for line in texts['objects'].split('\n\n'))
    assert 'Unusual speed' not in texts['objects']
    assert texts['objects'].count('zone entry')==2


def test_dense_labels_avoid_reserved_zone_and_each_other():
    occupied=[(180,150,350,195)];bounds=(0,0,600,400)
    for _ in range(10):
        box=place_label((300,200),(50,16),bounds,occupied)
        assert all(not intersects(box,old) for old in occupied)
        occupied.append(box)


def test_each_current_anomaly_has_a_reason_even_when_not_selected_for_scene_alert():
    d=fixture();row=d['timeline'][18]
    row['objects'][0].update(anomalous=True,anomaly_reason='Unusual direction for this location')
    row['objects'][1].update(anomalous=True,anomaly_reason='Unusual speed for this location')
    row['scenario']=dict(status='Potential intrusion',objects=[0],reasons=[
        'Drone 01 shows unusual direction for this location and is forecast to enter the protected zone.'])
    before=deepcopy(row);text=cleanup_texts(row)['explanation']
    flattened=' '.join(text.split()).lower()
    assert flattened.count('unusual direction for this location')==1
    assert flattened.count('unusual speed for this location')==1
    assert 'Aircraft 01:' in text
    assert row==before
    row['scenario']=dict(status='High threat',objects=[0],reasons=[
        'Drone 01 entered the protected zone at 80.7 s. It has left; the high alert remains active.'])
    text=cleanup_texts(row)['explanation'];flattened=' '.join(text.split()).lower()
    assert '80.7 s' in flattened and 'has left' in flattened
    assert 'unusual direction for this location' in flattened
    assert 'unusual speed for this location' in flattened
