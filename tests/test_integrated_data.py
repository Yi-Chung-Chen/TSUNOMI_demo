import numpy as np
import torch
from src.integrated.data import noisy_observations,observation_graph,scenario_stream
from src.integrated.forecast import medoid_index,constant_velocity


def fixture():
    t=np.arange(80)/3
    p=np.stack([t*20,t*0+100,t*0+110],axis=-1)[None]
    return p,np.ones(p.shape[:2],bool)


def test_noise_prefix_overlap_and_sigma_zero():
    p,_=fixture();ids=['a']
    np.testing.assert_array_equal(noisy_observations(p,ids,0,817,'s'),p)
    full=noisy_observations(p,ids,3,817,'s')
    short=noisy_observations(p[:,:40],ids,3,817,'s')
    np.testing.assert_array_equal(full[:,:40],short)
    assert not np.array_equal(full,p)
    np.testing.assert_array_equal(full[:,20:40],short[:,20:40])


def test_graph_features_do_not_read_future_or_clean_features():
    p,v=fixture();q=noisy_observations(p,['a'],1,817,'s')
    g=observation_graph(q,v,30)
    changed=q.copy();changed[:,31:]=1e8
    after=observation_graph(changed,v,30)
    for key in g['agent'].keys():
        if isinstance(g['agent'][key],torch.Tensor):torch.testing.assert_close(g['agent'][key],after['agent'][key])
    assert g['agent']['type'].item()==0 and g['agent']['position'].shape==(1,19,3)


def test_joint_medoid_is_an_existing_central_sample():
    p=np.zeros((3,2,90,3));p[1]=1;p[2]=10
    assert medoid_index(p)==1
    assert medoid_index(p+999)==1


def test_constant_velocity_is_exact_linear_numeric_reference():
    p,_=fixture();cv=constant_velocity(p,30)
    np.testing.assert_allclose(cv[0,:,0],p[0,30,0]+20*np.arange(1,91)/3)
