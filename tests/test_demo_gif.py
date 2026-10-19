import numpy as np
from PIL import Image
from src.eval import demo_gif as gif


def synthetic():
    gt = np.zeros((3,109,3))
    gt[:,:,0] = np.arange(109)[None,:]
    return dict(position=np.repeat(gt[None,:,19:],32,axis=0), gt_position=gt,
        heading=np.zeros((32,3,90)), gt_heading=np.zeros((3,109)),
        valid_mask=np.ones((3,109),bool), target_mask=np.array([True,True,False]),
        agent_id=np.array(['one','two','three']), date='synthetic', scenario='linear')


def test_origin_roundtrip_and_framing():
    zoom=12
    expected=np.array([(gif.ORIGIN[1]+180)/360,
        (1-np.arcsinh(np.tan(np.radians(gif.ORIGIN[0])))/np.pi)/2])*256*2**zoom
    np.testing.assert_allclose(gif.enu_to_pixel([0,0],zoom),expected)
    points=np.array([[-2000,-100],[1300,4000]])
    np.testing.assert_allclose(gif.pixel_to_enu(gif.enu_to_pixel(points,zoom),zoom),points,atol=1e-7)
    bounds=gif.view_bounds(points)
    assert np.all(points>=bounds[0]) and np.all(points<=bounds[1])
    pixels=gif.enu_to_pixel(points,zoom); edges=gif.enu_to_pixel(bounds,zoom)
    assert np.all(pixels>=edges.min(axis=0)) and np.all(pixels<=edges.max(axis=0))


def test_joint_argmin_uses_all_targets_full_valid_horizon():
    data=synthetic()
    data['position'] += 10
    data['position'][7,:2] = data['gt_position'][:2,19:]
    data['position'][7,2] += 10000  # non-target must not affect selection
    data['valid_mask'][1,60:] = False
    data['position'][7,1,41:] += 10000  # invalid future must not affect selection
    index,scores=gif.choose_rollout(data)
    reference=[]
    for pred in data['position']:
        reference.append(np.mean([np.linalg.norm(pred[a]-data['gt_position'][a,19:],axis=-1)
            [data['valid_mask'][a,19:]].mean() for a in [0,1]]))
    assert index == int(np.argmin(reference)) == 7
    np.testing.assert_allclose(scores,reference)


def test_synthetic_render_and_encoded_animation(tmp_path,monkeypatch):
    def fake_map(bounds,cache,theme):
        pixels=gif.enu_to_pixel(bounds,12)
        return Image.new('RGB',(100,100),'white'),12,np.stack((pixels.min(axis=0),pixels.max(axis=0)))
    monkeypatch.setattr(gif,'basemap',fake_map)
    frames,record=gif.render_window(synthetic(),tmp_path)
    assert len(frames)==record['frames']==109
    assert len(gif.frame_indices(2))==55
    still,still_record=gif.render_window(synthetic(),tmp_path,snapshot=True)
    assert len(still)==1 and still_record['final_time_s']==30
    # Direct snapshot equals the animation's final composition after the same palette.
    palette=frames[-1].copy()
    quantized=still[0].quantize(palette=palette,dither=Image.Dither.NONE)
    np.testing.assert_array_equal(np.asarray(quantized.convert('RGB')),np.asarray(frames[-1].convert('RGB')))
    path=tmp_path/'tiny.gif'; gif.save_gif(frames,path,1)
    with Image.open(path) as image:
        assert image.n_frames==109 and image.size==(1200,900)
        assert image.info['loop']==0
        image.seek(108)
        assert image.info['duration']==1500
        np.testing.assert_array_equal(np.asarray(image.convert('RGB')),np.asarray(frames[-1].convert('RGB')))


def test_reserved_colours_survive_encoding_and_delta_erases_old_icons(tmp_path):
    # Tiny coloured marks on a detailed neutral map formerly all became grey.
    grey=np.tile(np.arange(256,dtype=np.uint8),(128,1))
    background=Image.fromarray(np.repeat(grey[:,:,None],3,axis=2))
    palette=gif.fixed_palette(background)
    frames=[]; samples=[]
    for step in range(2):
        arr=np.array(background)
        for i,color in enumerate(gif.PALETTE):
            for j,variant in enumerate(gif.track_colours(color,'dark')[:2]):
                value=np.rint(variant*255).astype(np.uint8)
                x,y=10+i*24,10+step*50+j*15
                arr[y:y+3,x:x+3]=value
                samples.append((step,x+1,y+1,value))
        frames.append(Image.fromarray(arr).quantize(palette=palette,dither=Image.Dither.NONE))
    path=tmp_path/'colours.gif'; gif.save_gif(frames,path,1)
    with Image.open(path) as decoded:
        for step,x,y,value in samples:
            decoded.seek(step)
            np.testing.assert_array_equal(np.asarray(decoded.convert('RGB'))[y,x],value)
        decoded.seek(1)
        # Check the complete composite, including where icons existed in frame0.
        np.testing.assert_array_equal(np.asarray(decoded.convert('RGB')),np.asarray(frames[1].convert('RGB')))
