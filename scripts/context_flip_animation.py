"""Synchronised animation of the saved synthetic clock-flip track, no route search."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb
from src.eval.context_density import ContextDensity
from src.eval import demo_gif as visual
from scripts.context_flip_examples import render_pair,BG,MUTED,CYAN,GREEN,ORANGE


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];source=root/'outputs/context_flip_examples'
    out=root/'outputs/context_flip_animation';out.mkdir(exist_ok=True)
    protected={str(p.relative_to(root)):digest(p) for directory in ['outputs/context_flip_examples','outputs/anomaly_context_time','outputs/integrated_demo']
               for p in (root/directory).rglob('*') if p.is_file()}
    record=json.loads((source/'examples.json').read_text())
    reference=next(r for r in record['examples'] if r['id']=='synthetic_drone')
    model=ContextDensity.load(root/'outputs/anomaly_context_time/state_density.npz');cutoff=record['cutoff']
    assert digest(root/'outputs/anomaly_context_time/state_density.npz')==record['model_sha256']
    with np.load(source/'synthetic_drone_track.npz') as a:times=a['time_s'].copy();position=a['position'].copy()
    velocity=(position[-1]-position[0])/(times[-1]-times[0])
    np.testing.assert_allclose(position,position[0]+(times-times[0])[:,None]*velocity,rtol=0,atol=1e-10)
    speed=np.linalg.norm(velocity[:2]);course=np.arctan2(velocity[1],velocity[0])
    states=dict(position=position,speed=np.full(len(times),speed),course=np.full(len(times),course))
    # Display interpolation of the saved line only; no predicted trajectories.
    fps=15;display_times=np.linspace(times[0],times[-1],round((times[-1]-times[0])*fps)+1)
    display_position=np.column_stack([np.interp(display_times,times,position[:,i]) for i in range(3)])
    display_states=dict(position=display_position,speed=np.full(len(display_times),speed),course=np.full(len(display_times),course))
    day=model.score_context(display_states,0)['state_nll'];night=model.score_context(display_states,1)['state_nll']
    assert np.all(day<=cutoff) and np.all(night>cutoff)
    np.testing.assert_allclose(day,reference['day_score'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(night,reference['night_score'],atol=1e-10,rtol=0)
    case=dict(id='synthetic_drone',title='SYNTHETIC DRONE 01',kind='synthetic_drone',
              times=times,path=position,states=states,source=reference['source'],selection=reference['selection'])
    frames=[];mp4=out/'synthetic_drone_day_vs_night.mp4';gif=out/'synthetic_drone_day_vs_night.gif'

    def animate(fig,artists,map_image,zoom):
        fig.set_dpi(80);clock=fig.text(.965,.957,'',ha='right',color=MUTED,fontsize=15,fontfamily='monospace')
        projected=visual.enu_to_pixel(display_position[:,:2],zoom)
        palette=visual.fixed_palette(map_image,extra_colours=[GREEN,ORANGE,'#132434','#294052','#c77dff'])
        encoder=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x800',
                                  '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20',
                                  '-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],stdin=subprocess.PIPE)
        try:
            for frame,t in enumerate(display_times):
                clock.set_text(f't = {t:4.1f} s')
                points=projected[:frame+1];segments=np.stack([points[:-1],points[1:]],axis=1)
                colours=np.tile([*to_rgb(CYAN),1.],(len(segments),1))
                colours[:,3]=np.linspace(.15,.95,len(segments))
                for trail,icon,name in artists:
                    trail.set_segments(segments);trail.set_colors(colours)
                    icon.set_data([points[-1,0]],[points[-1,1]]);name.xy=points[-1]
                fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
                assert rgb.shape==(800,1600,3)
                encoder.stdin.write(rgb.tobytes())
                image=Image.fromarray(rgb);frames.append(image.quantize(palette=palette,dither=0))
                if frame in [0,75,150]:image.save(out/f'review_t{round(t):02d}.png')
                if frame%50==0:print(json.dumps(dict(rendered_frames=frame+1,total=len(display_times))),flush=True)
        finally:
            encoder.stdin.close()
            code=encoder.wait()
            if code:raise RuntimeError(f'ffmpeg encode exit{code}')
        # GIF has centisecond timing; alternate60/70ms for average15fps.
        boundaries=np.rint(np.arange(len(frames)+1)*100/fps).astype(int)*10
        durations=np.diff(boundaries).tolist()
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False,disposal=1)

    render_pair(case,model,cutoff,out,animation_callback=animate)
    assert protected=={p:digest(root/p) for p in protected}
    with Image.open(gif) as im:assert im.n_frames==151 and im.size==(1600,800)
    info=json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(mp4)]))['streams'][0]
    assert info['nb_frames']=='151' and info['r_frame_rate']=='15/1'
    subprocess.run([shutil.which('ffmpeg'),'-v','error','-threads','1','-i',str(mp4),'-f','null','-'],check=True)
    receipt=dict(frames=151,mp4_fps=15,playback='real time;10s trajectory',size=[1600,800],
                 gif_centisecond_timing=True,day_score=float(day[0]),night_score=float(night[0]),cutoff=cutoff,
                 source_track_sha256=digest(source/'synthetic_drone_track.npz'),
                 no_route_search=True,no_refit=True,no_forecast=True,shared_geometry=True,
                 interpolated_display_only=True,protected_files_unchanged=len(protected),mp4_full_decode=True)
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'README.md').write_text('''# Clock-context drone animation

Same accepted synthetic10s drone track, synchronised day/night, common frozen model and cutoff. Both panels share all positions/velocity/height/direction.15fps display interpolates the saved3Hz straight line only, with no new model prediction. Real-time replay;GIF loops. Scores remain11.80(day, normal) vs15.50(night, anomaly), cutoff12.64. Main increase is conditional speed rarity.

Synthetic showcase selected in the earlier experiment, not an actual drone recording or a new performance measurement. The aircraft-reference density scores kinematic state, not object identity/safety. Chicago clock categories07–19/19–07;12:00/22:00labels are counterfactual, not two actual recordings. No light/weather rendering, route changes, retraining, packages, network or changes to earlier demos. MP4/GIF1600x800; still3200x1600 and review frames included. Run: python -m scripts.context_flip_animation.
''')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
