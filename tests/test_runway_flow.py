import numpy as np
from src.eval.runway_flow import scenario_flow
from scripts.summarize_runway import comparison


def test_flow_votes_and_ties_match_reference_rule():
    pos=np.zeros((2,30,3));pos[0,:,0]=np.arange(30)*20;pos[1,:,0]=-np.arange(30)*20
    data=dict(pos=pos,valid=np.ones((2,30),bool),on_ground=np.ones((2,30),bool))
    assert scenario_flow(data)==dict(flow='none/tie',east_tracks=1,west_tracks=1)
    data['on_ground'][1]=False
    assert scenario_flow(data)==dict(flow='east',east_tracks=1,west_tracks=0)


def test_summary_comparison_has_explicit_metric_direction():
    assert comparison(.7,.5,'higher')=='better'
    assert comparison(10,8,'lower')=='worse'
    assert comparison(4,4,'lower')=='tie'
    assert comparison(4,5,'descriptive')=='NA'
