import numpy as np
import pytest
from PIL import Image

from src.eval import anomaly_gif as gif


def test_completed_score_availability_expiry_and_gap():
    rows=[dict(start=24,end=30,nll=7.),dict(start=30,end=36,nll=8.),dict(start=42,end=48,nll=2.)]
    assert gif.completed_score(rows,29) is None
    assert gif.completed_score(rows,30)['nll']==7.
    assert gif.completed_score(rows,35)['nll']==7.
    assert gif.completed_score(rows,36)['nll']==8.
    assert gif.completed_score(rows,41)['nll']==8.
    assert gif.completed_score(rows,42) is None
    assert gif.completed_score(rows,47) is None
    assert gif.completed_score(rows,48)['nll']==2.


def test_red_segments_strict_threshold_absolute_frames_and_gaps():
    xy=np.stack([np.arange(19),np.zeros(19)],axis=-1)
    valid=np.ones(19,bool);valid[9]=False
    rows=[dict(start=90,end=96,nll=7.),dict(start=96,end=102,nll=7.01),dict(start=102,end=108,nll=8.)]
    segments,ends=gif.high_segments(xy,valid,90,rows,7.)
    assert not (ends==96).any()  # Exactly7 is not high.
    assert not (ends<=101).any()  # A high token appears only when complete.
    assert np.count_nonzero(ends==102)==4  # Two edges touching invalid frame9 omitted.
    assert np.count_nonzero(ends==108)==6
    assert np.all(segments[:,0,0]>=6) and np.all(segments[:,1,0]<=18)


@pytest.mark.parametrize('score_label',['NLL','combined'])
def test_gt_animation_red_pixels_appear_only_after_high_token_completion(tmp_path,monkeypatch,score_label):
    def fake_map(bounds,cache,theme):
        pixels=gif.visual.enu_to_pixel(bounds,16)
        return Image.new('RGB',(100,100),'#0b1622'),16,np.stack((pixels.min(axis=0),pixels.max(axis=0)))
    monkeypatch.setattr(gif.visual,'basemap',fake_map)
    position=np.zeros((1,19,3));position[0,:,0]=np.arange(19)*10.
    data=dict(source='drone',scenario='synthetic',date='test',start_frame=24,
        gt_position=position,gt_heading=np.zeros((1,19)),valid_mask=np.ones((1,19),bool),agent_id=np.array(['drone']))
    tracks={('drone','synthetic','drone'):[dict(start=24,end=30,nll=8.),dict(start=30,end=36,nll=2.)]}
    frames,record=gif.render_scene(data,tracks,tmp_path,score_label=score_label)
    assert len(frames)==19 and record['objects']==1 and record['score_label']==score_label
    path=tmp_path/'gt_only.gif';gif.visual.save_gif(frames,path,1)
    alert=np.array([255,48,56],dtype=np.uint8)
    with Image.open(path) as decoded:
        decoded.seek(5)
        assert not np.any(np.all(np.asarray(decoded.convert('RGB'))==alert,axis=-1))
        decoded.seek(6)
        red_at_high=int(np.all(np.asarray(decoded.convert('RGB'))==alert,axis=-1).sum())
        assert red_at_high>100
        decoded.seek(12)
        # Old high-score trail remains while the current object returns low.
        red_at_low=int(np.all(np.asarray(decoded.convert('RGB'))==alert,axis=-1).sum())
        assert 0<red_at_low<red_at_high
        np.testing.assert_array_equal(np.asarray(decoded.convert('RGB')),np.asarray(frames[12].convert('RGB')))
