import numpy as np
import pytest
from src.integrated.future_groups import group_futures,replay_groups

ZONE=dict(polygon_enu_m=[[0,0],[10,0],[10,10],[0,10]],height_range_m=[0,100])


def test_counts_cover_every_sample_and_weights_sum_to_one():
    paths=np.zeros((32,8,3));paths[:20]=[20,20,10];paths[20:]=[5,5,10]
    groups=group_futures(paths,ZONE)
    assert sorted(m for group in groups for m in group['members'])==list(range(32))
    assert sum(group['share'] for group in groups)==1
    assert {g['percentage'] for g in groups}=={'62.5%','37.5%'}
    assert sum(g['count'] for g in groups if g['enters_zone'])==12


def test_intermediate_crossing_and_height_are_respected():
    paths=np.full((3,3,3),[20,20,10.]);paths[0,1]=[5,5,10];paths[1,1]=[5,5,200]
    groups=group_futures(paths,ZONE)
    assert [g['members'] for g in groups if g['enters_zone']]==[[0]]


def test_representatives_are_actual_group_medoids_and_deterministic():
    rng=np.random.default_rng(10);paths=rng.normal(size=(32,10,3))*8+[30,30,10]
    groups=group_futures(paths,ZONE)
    assert groups==group_futures(paths,ZONE) and len(groups)<=4
    for group in groups:
        members=group['members'];assert group['representative_index'] in members
        d=np.linalg.norm(paths[members,None]-paths[None,members],axis=-1).mean(axis=-1)
        best=d.sum(axis=1).min()
        assert d.sum(axis=1)[members.index(group['representative_index'])]==pytest.approx(best)


def test_identical_futures_do_not_create_spurious_modes():
    groups=group_futures(np.full((32,90,3),[20,20,10]),ZONE)
    assert len(groups)==1 and groups[0]['percentage']=='100%'


def test_replay_groups_discard_crossings_that_are_now_in_the_past():
    paths=np.full((32,1,3,3),[20,20,10.]);paths[:,:,0]=[5,5,10]
    rows=[dict(objects=[dict(forecast_origin_frame=0)]) for _ in range(3)]
    groups=replay_groups(rows,{0:dict(position=paths)},ZONE)
    assert all(g['enters_zone'] for g in groups[0])
    assert not any(g['enters_zone'] for g in groups[1])
