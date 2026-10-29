"""Read-only accepted-D4 render, cached maps ONLY; no inference or network."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile,ZIP_DEFLATED
from src.eval import demo_gif as visual
from src.integrated.data import verify_lock
from src.integrated.presentation import load_presentation,DemoRenderer,viewer_bundle
from src.integrated.story import write_json


def source_hashes(output):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['A_daytime_hobbyist','B_nighttime_approach']
        for p in (output/name).rglob('*') if p.is_file()}


def main():
    root=Path(__file__).resolve().parents[1];output=root/'outputs/integrated_demo'
    gate=json.loads((output/'D4_acceptance.json').read_text());assert gate['passed'] and gate['attempt']==1 and not gate['retuned']
    verify_lock(root);before=source_hashes(output)
    config=json.loads((root/'configs/integrated_demo/v1.json').read_text());template=(root/'src/integrated/viewer.html').read_text()
    results=[]
    for scene in config['scenarios']:
        name=scene['id'];out=output/'D5'/name;out.mkdir(parents=True,exist_ok=True)
        data=load_presentation(root,name);renderer=DemoRenderer(data)
        palette=visual.fixed_palette(data['map_image'],extra_colours=['#ff626b','#132434','#294052'])
        frames=[]
        for frame in range(len(data['timeline'])):
            frames.append(renderer.frame(frame).quantize(palette=palette,dither=0))
            if frame%90==0:print(json.dumps(dict(scene=name,rendered_frames=frame+1)),flush=True)
        renderer.close();gif=out/f'{name}.gif';visual.save_gif(frames,gif,1)
        mp4=out/f'{name}.mp4';ffmpeg=shutil.which('ffmpeg');mp4_ok=False
        if ffmpeg:
            result=subprocess.run([ffmpeg,'-y','-loglevel','error','-threads','1','-i',str(gif),'-vf','fps=9',
                '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],capture_output=True,text=True)
            mp4_ok=result.returncode==0
            if not mp4_ok:(out/'mp4_error.txt').write_text(result.stderr)
        del frames;gc.collect()
        renderer=DemoRenderer(data,dpi=200)
        snapshots=(dict(initial=18,motion_change=60,first_high=round(gate['B']['first_model_based_preentry_high_s']*3),end=315)
            if scene['context']=='nighttime' else dict(initial=18,hobbyist_activity=96,low_threat=180,end=315))
        for label,frame in snapshots.items():renderer.frame(frame).save(out/f'{label}.png')
        renderer.close();bundle=viewer_bundle(data)
        html=template.replace('__DEMO_DATA__',json.dumps(bundle,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
        (out/'viewer.html').write_text(html)
        results.append(dict(scene=name,gif=str(gif.relative_to(output)),gif_bytes=gif.stat().st_size,
            mp4=str(mp4.relative_to(output)) if mp4_ok else None,pngs=[str((out/f'{label}.png').relative_to(output)) for label in snapshots],
            viewer=str((out/'viewer.html').relative_to(output)),snapshots=snapshots,map_zoom=data['zoom']))
        print(json.dumps(results[-1]),flush=True);del data,bundle;gc.collect()
    caption=('Integrated scripted MVP: reference plus scripted Gaussian position noise acts as simulated tracker output '
        '(day sigma1m/axis,night sigma3m/axis). A frozen model generates32joint futures from6s history, forecasting30s every2s. '
        'The displayed sample is the GT-free joint3D medoid. Frozen state density scores location/height,speed and direction. '
        'Threat rules apply only after observed current score exceeded cutoff within10s; others show Normal traffic and no threat score. '
        'Protected zones and day/night are scripted threat context; anomaly density has no weather/time-of-day conditioning. '
        'First fixed run: A anomalous/low threat; B high warning at56s before reference entry80.67s,24.67s lead. '
        'Future anomaly is not earlier than current anomaly here. Numeric-only CV is never displayed or substituted. '
        'Rules and samples are illustrative, not calibrated risks or independent validation.')
    d5=output/'D5';(d5/'CAPTION.md').write_text(caption+'\n')
    links=''.join(f'<li><a href="{r["scene"]}/viewer.html">{r["scene"]} interactive replay</a> · <a href="{r["scene"]}/{r["scene"]}.mp4">MP4</a> · <a href="{r["scene"]}/{r["scene"]}.gif">GIF</a></li>' for r in results)
    (d5/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SYGARD Demo</title><body style="background:#0b1622;color:#e8eef5;font:18px Arial;padding:40px"><h1>SYGARD / Integrated MVP</h1><ul>'+links+'</ul><p>Open locally. Each interactive replay embeds its own data; no server/network needed.</p></body>')
    (d5/'README.md').write_text('# Integrated MVP v1\n\nOpen index.html locally for two offline viewers with play/pause,slider,32-rollout and reference-future toggles. '
        'MP4/GIF9fps (3× recorded3Hz time); PNG3200×1800. All media read accepted D4 only, without inference. '
        'All maps come from pre-existing local cache; no map/network requests.\n\n'+caption+'\n')
    assert before==source_hashes(output);verify_lock(root)
    write_json(d5/'render_receipt.json',dict(stage='D5',media=results,d4_files_unchanged=True,model_execution=False,network_requests=False,caption=caption))
    archive=output/'integrated_demo_v1.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for p in sorted(d5.rglob('*')):
            if p.is_file():z.write(p,arcname=str(p.relative_to(d5)))
    print(json.dumps(dict(media_complete=True,archive=str(archive))),flush=True)


if __name__=='__main__':main()
