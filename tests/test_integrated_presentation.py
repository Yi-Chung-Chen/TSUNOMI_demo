from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
from src.integrated.presentation import DemoRenderer,display_texts,cached_basemap
from src.eval import demo_gif as visual


def fixture():
    points=np.zeros((2,109,3));points[0,:,0]=np.arange(109)*5;points[0,:,1]=200;points[0,:,2]=100
    points[1]=points[0]+[0,300,0]
    future=np.repeat(points[None,:,19:],32,axis=0)
    objects=[dict(current_state_nll=8,anomalous=False,threat=dict(level='Normal traffic',score=None,
        reasons=['No current anomaly in the last10s']),entry=None,explanation=[],future_anomaly_horizon_s=None,
        forecast_origin_frame=None) for _ in range(2)]
    timeline=[dict(time_s=f/3,objects=deepcopy(objects)) for f in range(109)]
    for row in timeline[18:]:
        for o in row['objects']:o['forecast_origin_frame']=18;o['entry']=dict(entry_samples=0)
    bounds=np.array([[-100,-100],[1600,1100.]])
    pb=visual.enu_to_pixel(bounds,14);pb=np.array([[pb[0,0],pb[1,1]],[pb[1,0],pb[0,1]]])
    return dict(manifest=dict(scene=dict(context='daytime',scripted_noise_sigma_m_per_axis=1),
        zone=dict(polygon_enu_m=[[0,0],[100,0],[100,100],[0,100]])),timeline=timeline,
        forecasts={18:dict(position=future,medoid=0)},stream=dict(observed=points,reference=points.copy()),
        zoom=14,pb=pb,map_image=Image.new('RGB',(100,100),'#0b1622'))


def test_normal_status_has_no_threat_number():
    d=fixture();texts=display_texts(d['timeline'][18],d['manifest'])
    assert texts['threat']=='Normal traffic'
    assert texts['background']=='Normal traffic · no threat score'


def test_default_frame_does_not_draw_future_reference_or_future_observations():
    d=fixture();renderer=DemoRenderer(d);before=np.asarray(renderer.frame(18));renderer.close()
    changed=deepcopy(d);changed['stream']['reference'][:]=1e9
    changed['stream']['observed'][:,19:]=1e9
    renderer=DemoRenderer(changed);after=np.asarray(renderer.frame(18));renderer.close()
    np.testing.assert_array_equal(before,after)


def test_offline_map_cache_miss_never_calls_http(tmp_path,monkeypatch):
    import requests
    def forbidden(*args,**kwargs):raise AssertionError('Network must not be called')
    monkeypatch.setattr(requests.Session,'get',forbidden)
    with pytest.raises(FileNotFoundError,match='No complete locally cached'):
        cached_basemap(np.array([[-1000.,-1000],[1000,1000]]),tmp_path)
