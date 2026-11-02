import math
import numpy as np
from src.common.runway_anchors import fixed_endpoints,alignment_masks,geodetic_to_enu

ORIGIN=dict(latitude_deg=41.980103,longitude_deg=-87.903868,altitude_m=2250.)


def test_wgs84_enu_matches_a_known_reference_sample():
    expected=[2528.2901628150676,-1271.0020854745346,-1716.6899167663032]
    actual=geodetic_to_enu(41.96865694230866,-87.87336931417013,533.9373071995994,ORIGIN)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)


def test_sixteen_fixed_ends_have_reciprocal_headings():
    rows=fixed_endpoints(ORIGIN);assert len(rows)==16 and len({r['ident'] for r in rows})==16
    assert {r['ident'] for r in rows}=={'04L','22R','04R','22L','09C','27C','09L','27R','09R','27L','10C','28C','10L','28R','10R','28L'}
    for a,b in zip(rows[::2],rows[1::2]):
        delta=(a['inbound_heading_rad']-b['inbound_heading_rad'])%(2*math.pi)
        assert abs(delta-math.pi)<1e-10


def test_ground_heading_and_finite_segment_air_extensions_and_altitude():
    positions=np.array([[[50,0,1],[50,0,1],[-5,0,10],[105,0,10],[50,0,10],[-5,0,0],[-5,0,500],[-6001,0,10]]],float)
    heading=np.array([[0,math.pi/2,0,math.pi/2,0,0,0,0.]])
    valid=np.ones((1,8),bool);ground=np.array([[True,True,False,False,False,False,False,False]])
    g,a,z=alignment_masks(positions,heading,valid,ground,[0,0],[100,0])
    assert g.tolist()==[[True,False,False,False,False,False,False,False]]
    assert a.tolist()==[[False,False,True,False,False,True,False,False]]
    assert z[0,:2].all()
