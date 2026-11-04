"""Offline presentation revision from frozen D11; no model execution."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
from src.eval import demo_gif as visual
from src.integrated.data import verify_lock
from src.integrated.presentation import load_presentation
from src.integrated.story import write_json
from src.integrated.track_style import TrackStyleRenderer


def hashes(output):
    return {str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
            for p in output.rglob('*') if p.is_file() and 'D12' not in p.relative_to(output).parts
            and p.name!='integrated_demo_v7.zip'}


def main():
    root=Path(__file__).resolve().parents[1];output=root/'outputs/integrated_demo'
    out=output/'D12';out.mkdir(exist_ok=True);verify_lock(root);before=hashes(output)
    write_json(out/'protected_before.json',before)
    receipt=dict(stage='D12',model_execution=False,network=False,source='D11/D4_anchor',
                 display_position='causal linear fit, last7samples (2s), evaluated at now',
                 heading='latest joint medoid next2s tangent, circular easing tau0.6s; filtered history before6s',
                 history='raw noisy positions; faint full history plus age-faded recent10s',scenes=[])
    font=ImageFont.truetype(findfont('DejaVu Sans'),32)
    for name,reviews in [('A_daytime_hobbyist',[48,96,187]),('B_nighttime_approach',[48,96,168,174,190,273])]:
        fixed=load_presentation(root,name,True,True,True)
        data=load_presentation(root,name,True,True,True,run_dir=output/'D11/D4_anchor',fixed_view=fixed)
        timeline_before=json.dumps(data['timeline'],sort_keys=True)
        observed_before=data['stream']['observed'].copy()
        renderer=TrackStyleRenderer(data);directory=out/name;directory.mkdir(exist_ok=True)
        palette=visual.fixed_palette(data['map_image'],extra_colours=['#ff9f43','#ffd166','#ff626b','#132434','#294052'])
        frames=[]
        for frame in range(len(data['timeline'])):
            frames.append(renderer.frame(frame).quantize(palette=palette,dither=0))
            if frame%90==0:print(json.dumps(dict(scene=name,frames=frame+1)),flush=True)
        layouts=renderer.layout_receipt
        np.savez_compressed(directory/'display_pose.npz',position_enu=renderer.filtered,heading=renderer.icon_headings)
        renderer.close();gif=directory/f'{name}.gif';visual.save_gif(frames,gif,1);del frames;gc.collect()
        mp4=directory/f'{name}.mp4'
        subprocess.run([shutil.which('ffmpeg'),'-y','-loglevel','error','-threads','1','-i',str(gif),'-vf','fps=9',
                        '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
        renderer=TrackStyleRenderer(data,dpi=200)
        for frame in reviews:
            image=renderer.frame(frame);image.save(directory/f'review_f{frame:04d}.png')
            old=output/'D11'/name/f'review_f{frame:04d}.png'
            if old.exists():
                with Image.open(old) as im:prior=im.convert('RGB')
                compare=Image.new('RGB',(6400,1870),'#0b1622');compare.paste(prior,(0,70));compare.paste(image,(3200,70))
                draw=ImageDraw.Draw(compare)
                draw.text((35,18),f'D11 · {name} · {frame/3:.2f} s',font=font,fill='#e8eef5')
                draw.text((3235,18),'D12 · recent history + display smoothing',font=font,fill='#e8eef5')
                compare.save(out/f'comparison_{name}_f{frame:04d}.png')
        renderer.close()
        assert timeline_before==json.dumps(data['timeline'],sort_keys=True)
        np.testing.assert_array_equal(observed_before,data['stream']['observed'])
        old_timeline=json.loads((output/'D11'/name/'timeline.json').read_text())
        assert data['timeline']==old_timeline
        write_json(directory/'layout_checks.json',layouts)
        receipt['scenes'].append(dict(scene=name,frames=len(layouts),timeline_unchanged=True,observations_unchanged=True,
                                      labels_nonoverlapping=True,map_bounds=data['pb'].tolist(),review_frames=reviews))
    assert before==hashes(output);verify_lock(root)
    receipt['protected_files_unchanged']=len(before);write_json(out/'acceptance.json',receipt)
    (out/'README.md').write_text('''# Integrated demo v7 — tracker presentation

Read frozen D11 anchor forecasts only. Recent 10 seconds of raw tracker history fade from dim to bright over a faint full-history line. A fixed-duration tail is longer at higher speed. Original tracker noise remains visible.

Icons use a causal linear fit over the last 7 observed samples (2 s), evaluated at the current time. Orientation uses the latest model joint-medoid tangent over the next 2 s, circularly eased with a 0.6 s time constant. Before the first forecast at 6 s, heading uses filtered observed displacement. These are display transforms only.

Raw observations, all32 forecasts, sampled groups/percentages, anomalies, explanations, threat timelines and incident memory are unchanged. Map extents and cards retain D11 settings. B still warns at58s (22.67s before reference entry), and A has no intrusion. No inference, training, packages, network or new scenarios. Previous versions and interactive viewer are preserved.

MP4/GIF:1600x900,3x replay. Review PNG:3200x1800. Compare D11/D12 at A62.33s and B56/58/63.33s. Additional PNGs show before/after acceleration and post-intrusion memory.
''')
    archive=package(out)
    print(json.dumps(dict(completed=True,output=str(out),archive=str(archive),protected=len(before))),flush=True)


def package(out):
    archive=out.parent/'integrated_demo_v7.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='protected_before.json' and p.suffix not in ['.log','.npz']:
                z.write(p,Path('integrated_demo_v7')/p.relative_to(out))
    with ZipFile(archive) as z:assert z.testzip() is None
    return archive


if __name__=='__main__':main()
