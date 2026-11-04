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
    assert abs(headings[0,-1])>3 and np.all(np.abs(np.diff(headings[0]))<1.5)
    other=position.copy();other[:,19:]*=-100
    np.testing.assert_array_equal(predicted_headings(data,other)[:,18:],headings[:,18:])


def test_fixed_time_tail_fades_with_age_and_has_thirty_segments():
    start,alpha=trail_alphas(60)
    assert start==30 and len(alpha)==30 and np.all(np.diff(alpha)>0)
    assert alpha[-1]==.95
    assert trail_alphas(0)[1].size==0
