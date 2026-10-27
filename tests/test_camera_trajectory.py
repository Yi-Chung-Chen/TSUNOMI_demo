import json
import numpy as np
import pytest
import torch

from src.data.camera_trajectory import (load_camera_tracks,sample_camera,camera_features,
    first_camera_origin,make_camera_graph,sample_reference)


def track():
    times=np.arange(0,15.01,.25)
    return dict(times=times,position=np.stack([times*20,times*2,100+times],-1),
        velocity=np.tile([20.,2.,1.],(len(times),1)),std=np.ones((len(times),3)),
        measured=np.ones(len(times),bool),final_measured=np.ones(len(times),bool))


def test_sequential_events_take_last_posterior_and_keep_measurement_evidence(tmp_path):
    rows=[dict(tracker_id=1,elapsed_s=1.,event_index=e,xyz_enu_m=[x,0,-2080],
        velocity_enu_mps=[1,0,0],position_std_m=[1,1,1],status=status,
        matched_source_object_id='must_not_be_used') for e,x,status in [(2,2,'predicted'),(1,1,'measured')]]
    path=tmp_path/'camera.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    t=load_camera_tracks(path)[1]
    assert t['position'][0,0]==2 and t['position'][0,2]==4
    assert t['measured'][0] and not t['final_measured'][0]
    assert 'matched_source_object_id' not in t


def test_camera_sampling_uses_causal_velocity_projection_and_no_future_interpolation():
    t=track();q=np.array([-.1,.1,1.1,16.5])
    result=sample_camera(t,q)
    assert list(result['valid'])==[False,True,True,False]
    assert np.isclose(result['position'][2,0],22.)  # state at1s advanced by its own velocity.
    t['position'][5,0]=1e9
    assert np.isclose(sample_camera(t,[1.1])['position'][0,0],22.)
    assert (result['source_time_s'][result['valid']]<=q[result['valid']]).all()
    # Holding an already-known state cannot depend on a later event existing.
    t['times']=t['times'][:5]
    for key in ['position','velocity','std','measured','final_measured']:t[key]=t[key][:5]
    assert sample_camera(t,[1.1])['valid'][0]


def test_camera_graph_unchanged_if_future_positions_or_identity_annotations_change():
    t=track();g,_=make_camera_graph(t,24)
    changed={k:v.copy() for k,v in t.items()};future=changed['times']>8
    for key in ['position','velocity','std']:changed[key][future]=1e9
    changed['gt_heading']=np.zeros(100);changed['matched_source_object_id']='unrelated'
    after,_=make_camera_graph(changed,24)
    for key in g['agent'].keys():
        before=g['agent'][key];value=after['agent'][key]
        if isinstance(before,torch.Tensor):torch.testing.assert_close(before,value)
    assert g['agent']['position'].shape==(1,19,3)
    assert g['agent']['type'].item()==0
    assert first_camera_origin(t)==18


def test_ground_hysteresis_and_heading_are_causal_camera_only():
    t=track();sample=sample_camera(t,np.arange(19)/3)
    sample['position'][:,2]=[5,12,18,21,15,11,9]+[8]*12
    heading,ground=camera_features(sample)
    assert list(ground[:7])==[True,True,True,False,False,False,True]
    np.testing.assert_allclose(heading,np.arctan2(2,20))
    altered={k:v.copy() for k,v in sample.items()};altered['position'][10:]=1e6
    h2,g2=camera_features(altered)
    np.testing.assert_array_equal(heading[:10],h2[:10]);np.testing.assert_array_equal(ground[:10],g2[:10])


def test_gaps_do_not_become_complete_model_history():
    t=track();keep=(t['times']<2)|(t['times']>4)
    t={k:v[keep] for k,v in t.items()}
    with pytest.raises(ValueError,match='19 valid'):make_camera_graph(t,24)
    assert first_camera_origin(t)>=30


def test_gt_overlay_interpolation_masks_gaps_and_never_extrapolates():
    t=dict(times=np.array([1.,1.5,4.]),position=np.array([[0,0,0],[1,1,1],[10,10,10.]]))
    pos,valid=sample_reference(t,np.array([0.,1.25,2.,4.,5.]))
    assert list(valid)==[False,True,False,True,False]
    np.testing.assert_allclose(pos[1],[.5,.5,.5]);assert np.isnan(pos[~valid]).all()


def test_origin_requires_recent_measurement_not_only_kalman_coasting():
    t=track();t['measured'][:]=False;t['measured'][:16:4]=True
    assert first_camera_origin(t) is None


def test_zero_velocity_initialization_waits_for_second_measurement(tmp_path):
    rows=[dict(tracker_id=1,elapsed_s=float(i),event_index=i,xyz_enu_m=[0,0,-2080],
        velocity_enu_mps=[0,0,0],position_std_m=[1,1,1],status='measured') for i in range(3)]
    path=tmp_path/'camera.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows))
    t=load_camera_tracks(path)[1]
    assert list(sample_camera(t,[0.,1.,2.])['valid'])==[False,True,True]


def test_three_way_renderer_preserves_sources_and_shows_completed_forecast(tmp_path,monkeypatch):
    from PIL import Image
    from src.eval import camera_rollout as visual
    def fake_map(bounds,cache,theme):
        pixels=visual.visual.enu_to_pixel(bounds,14)
        return Image.new('RGB',(100,100),'#0b1622'),14,np.stack((pixels.min(0),pixels.max(0)))
    monkeypatch.setattr(visual.visual,'basemap',fake_map)
    camera=np.stack([np.arange(109)*10.,np.zeros(109),np.ones(109)*100],-1)
    pred=camera[19:].copy();pred[:,1]+=np.arange(90)*2
    gt=camera.copy();gt[:,1]-=np.arange(109)
    data=dict(camera_position=camera,camera_valid=np.arange(109)<60,
        gt_position=gt,gt_valid=np.ones(109,bool),prediction=pred[None],
        prediction_heading=np.zeros((1,90)),kind='aircraft',tracker_id=1,camera_future_coverage_s=41/3)
    before={key:value.copy() for key,value in data.items() if isinstance(value,np.ndarray)}
    frames,receipt=visual.render_camera_scene(data,tmp_path,snapshot=True)
    assert len(frames)==1 and frames[0].size==(1200,900) and receipt['forecast_s']==30
    assert receipt['prediction_sample_index']==0
    for key,value in before.items():np.testing.assert_array_equal(data[key],value)
    pixels=np.asarray(frames[0])
    for name,colour in visual.COLOURS.items():
        expected=np.array([int(colour[i:i+2],16) for i in [1,3,5]])
        assert np.count_nonzero(np.all(pixels==expected,axis=-1))>50,name
