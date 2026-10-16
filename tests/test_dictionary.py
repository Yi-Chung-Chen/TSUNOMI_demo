import numpy as np
from src.common.geometry import contours, causal_headings, decode_tokens, match_endpoints
from src.preprocess.motion_dict import normalize_segments, kdisk_cluster


def test_transform_reconstruction_and_joint_matching():
    traj = np.zeros((2, 7, 3), np.float32)
    traj[1, :, 0] = np.arange(7)
    dz = np.zeros((2, 7), np.float32)
    dz[1] = np.arange(7)
    d = dict(traj=traj, dz=dz, cluster_centers=contours(traj), meta=dict(radius=2., lambda_z=1.))
    pos = np.asarray([[20., 30., 4.]])
    head = np.asarray([np.pi / 2])
    decoded, headings = decode_tokens(np.array([1]), pos, head, d)
    np.testing.assert_allclose(decoded[0, -1], [20., 36., 10.])
    ids = match_endpoints(decoded[:, -1], headings[:, -1], pos, head, d)
    assert ids.tolist() == [1]
    relative, z = normalize_segments(decoded, headings)
    np.testing.assert_allclose(relative, traj[1:2], atol=1e-6)
    np.testing.assert_allclose(z, dz[1:2])


def test_chord_heading_never_uses_future():
    p = np.zeros((1, 20, 3), np.float32)
    p[0, :, 0] = np.arange(20) * 10
    h = np.ones((1, 20), np.float32)
    v = np.ones((1, 20), bool)
    g = np.zeros((1, 20), bool)
    original = causal_headings(p, h, v, g)
    p[:, 11:, 1] = 1000
    changed = causal_headings(p, h, v, g)
    np.testing.assert_array_equal(original[:, :11], changed[:, :11])


def test_exact_static_token_and_reproducible_kdisk():
    rng = np.random.default_rng(42)
    p = rng.normal(size=(100, 7, 3)).astype(np.float32)
    p[:, 0] = 0
    z = rng.normal(size=(100, 7)).astype(np.float32)
    z[:, 0] = 0
    a = kdisk_cluster(p, z, 8, tol=.01)
    b = kdisk_cluster(p, z, 8, tol=.01)
    np.testing.assert_array_equal(a[0][0], 0)
    np.testing.assert_array_equal(a[1][0], 0)
    np.testing.assert_array_equal(a[0], b[0])


def test_dictionary_evaluation_horizon_counts_and_gap():
    from src.preprocess.motion_dict_eval import reconstruct
    traj = np.zeros((2, 7, 3), np.float32)
    traj[1, :, 0] = np.arange(7)
    d = dict(traj=traj, dz=np.zeros((2, 7), np.float32), cluster_centers=contours(traj),
             meta=dict(radius=2., lambda_z=1., shift=6, pos_scale=10., heading_source='data'))
    pos = np.zeros((3, 109, 3), np.float32)
    pos[:, :, 0] = np.arange(109) * 10
    h = np.zeros((3, 109), np.float32)
    v = np.ones((3, 109), bool)
    v[1, 23] = False  # mid-token gap excludes at both horizons
    v[2, 70] = False  # excludes at 30s only
    ground = np.zeros_like(v)
    e10 = reconstruct(d, pos, h, v, ground, 10)
    e30 = reconstruct(d, pos, h, v, ground, 30)
    assert e10['count'] == 2 and e30['count'] == 1
    np.testing.assert_allclose(e10['horizontal'], 0, atol=1e-5)
    np.testing.assert_allclose(e30['horizontal'], 0, atol=1e-5)
