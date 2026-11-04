import numpy as np
from src.integrated.track_style import causal_positions,predicted_headings,trail_alphas


def test_display_filter_is_causal_and_preserves_constant_velocity_without_lag():
    t=np.arange(50)/3;points=np.stack([30*t,2*t,100+0*t],axis=-1)[None]
    np.testing.assert_allclose(causal_positions(points),points,atol=1e-12)
    rng=np.random.default_rng(4);noisy=points+rng.normal(0,3,points.shape)
    before=noisy.copy();filtered=causal_positions(noisy)
    assert np.mean((filtered[:,6:]-points[:,6:])**2)<np.mean((noisy[:,6:]-points[:,6:])**2)
    np.testing.assert_array_equal(noisy,before)
    noisy[:,25:]+=1000
    np.testing.assert_array_equal(causal_positions(noisy)[:,:25],filtered[:,:25])


def test_orientation_uses_issued_model_forecast_and_short_circular_transition():
    # Observed movement is east; the model predicts west just across +/-pi.
    position=np.zeros((1,24,3));position[0,:,0]=np.arange(24)
    path=np.zeros((1,1,90,3));path[0,0,:,0]=-np.arange(90);path[0,0,:,1]=np.arange(90)*.001
    timeline=[{'objects':[{'forecast_origin_frame':18 if f>=18 else None}]} for f in range(24)]
    data=dict(timeline=timeline,forecasts={18:dict(position=path,medoid=0)})
    headings=predicted_headings(data,position)
    assert abs(headings[0,-1])>3 and np.all(np.abs(np.diff(headings[0,1:]))<1.5)
    other=position.copy();other[:,19:]*=-100
    np.testing.assert_array_equal(predicted_headings(data,other)[:,18:],headings[:,18:])


def test_fixed_time_tail_fades_with_age_and_has_thirty_segments():
    start,alpha=trail_alphas(60)
    assert start==30 and len(alpha)==30 and np.all(np.diff(alpha)>0)
    assert alpha[-1]==.95
    assert trail_alphas(0)[1].size==0


def test_westbound_track_initializes_at_first_valid_direction_without_half_turn():
    p=np.zeros((1,8,3));p[0,3:,0]=-10*np.arange(1,6)
    data=dict(timeline=[{'objects':[{'forecast_origin_frame':None}]} for _ in range(8)],forecasts={})
    heading=predicted_headings(data,p)
    assert np.isnan(heading[0,:3]).all()
    np.testing.assert_allclose(np.abs(heading[0,3:]),np.pi)
    # Changes after the first valid direction cannot influence earlier frames.
    changed=p.copy();changed[:,4:]=1000
    np.testing.assert_array_equal(predicted_headings(data,changed)[:,:4],heading[:,:4])


def test_first_issued_forecast_can_initialize_heading_without_observed_motion():
    p=np.zeros((1,4,3));future=np.zeros((1,1,90,3));future[0,0,:,0]=-np.arange(90)
    data=dict(timeline=[{'objects':[{'forecast_origin_frame':3 if f==3 else None}]} for f in range(4)],
              forecasts={3:dict(position=future,medoid=0)})
    heading=predicted_headings(data,p)
    assert np.isnan(heading[0,:3]).all()
    assert abs(heading[0,3])==np.pi


def test_unknown_heading_draws_dot_then_restores_directional_icon():
    from tests.test_integrated_presentation import fixture
    from src.integrated.track_style import TrackStyleRenderer
    d=fixture();d.update(grouped=True,all_anomalies=True,object_groups={},forecasts={})
    d['timeline']=d['timeline'][:2]
    for key in ['observed','reference']:d['stream'][key]=d['stream'][key][:,:2]
    renderer=TrackStyleRenderer(d)
    try:
        renderer.frame(0)
        assert all(track[2].get_marker()=='o' and track[2].get_markersize()==7 for track in renderer.tracks)
        renderer.frame(1)
        assert all(track[2].get_markersize()==24 for track in renderer.tracks)
        assert all(not isinstance(track[2].get_marker(),str) for track in renderer.tracks)
    finally:renderer.close()


def test_first_visible_heading_uses_supplied_history_even_when_forecast_disagrees():
    p=np.zeros((1,21,3));p[0,:,0]=-np.arange(21)*10
    future=np.zeros((1,1,90,3));future[0,0,:,0]=np.arange(90)*10
    data=dict(playback_start_frame=18,
              timeline=[{'objects':[{'forecast_origin_frame':18 if f>=18 else None}]} for f in range(21)],
              forecasts={18:dict(position=future,medoid=0)})
    heading=predicted_headings(data,p)
    assert abs(heading[0,18])==np.pi # prior history is west, forecast points east
    assert abs(heading[0,19])<np.pi # prediction-following easing resumes after initialization
    changed=p.copy();changed[:,19:]=1e6
    np.testing.assert_array_equal(predicted_headings(data,changed)[:,:19],heading[:,:19])


def test_first_visible_renderer_has_directional_icons_from_history():
    from tests.test_integrated_presentation import fixture
    from src.integrated.track_style import TrackStyleRenderer
    d=fixture();d.update(playback_start_frame=18,grouped=True,all_anomalies=True,object_groups={})
    # Limit to history plus first visible frame; no unused expired fixture forecast.
    d['timeline']=d['timeline'][:19]
    for key in ['observed','reference']:d['stream'][key]=d['stream'][key][:,:19]
    renderer=TrackStyleRenderer(d)
    try:
        renderer.frame(18)
        assert np.isfinite(renderer.icon_headings[:,18]).all()
        assert all(track[2].get_markersize()==24 and not isinstance(track[2].get_marker(),str) for track in renderer.tracks)
    finally:renderer.close()
