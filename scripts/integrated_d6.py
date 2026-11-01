"""Read-only accepted-D4 render, cached maps ONLY; no inference or network."""
import gc
import csv
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
        name=scene['id'];out=output/'D6'/name;out.mkdir(parents=True,exist_ok=True)
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
        warning_frames=[i for i,row in enumerate(data['timeline']) if row['scenario']['status']=='Potential intrusion']
        first_warning=warning_frames[0] if warning_frames else None
        snapshots=(dict(initial=18,motion_change=60,first_warning=first_warning if first_warning is not None else 168,end=315)
            if scene['context']=='nighttime' else dict(initial=18,hobbyist_activity=96,unusual_activity=180,end=315))
        for label,frame in snapshots.items():renderer.frame(frame).save(out/f'{label}.png')
        renderer.close();bundle=viewer_bundle(data)
        html=template.replace('__DEMO_DATA__',json.dumps(bundle,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
        (out/'viewer.html').write_text(html)
        write_json(out/'timeline.json',data['timeline'])
        with (out/'scores.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['time_s','object_id','current_state_nll','anomalous','anomaly_reason','entry_samples','scenario_status']);writer.writeheader()
            for row in data['timeline']:
                for obj in row['objects']:
                    writer.writerow(dict(time_s=row['time_s'],object_id=obj['id'],current_state_nll=obj['current_state_nll'],anomalous=obj['anomalous'],anomaly_reason=obj['anomaly_reason'],entry_samples=obj['entry']['entry_samples'] if obj['entry'] else None,scenario_status=row['scenario']['status']))
        results.append(dict(scene=name,
            first_potential_intrusion_s=data['timeline'][first_warning]['time_s'] if first_warning is not None else None,
            reference_entry_s=data['manifest']['evaluation']['reference_entry_s'],
            scenario_counts={status:sum(row['scenario']['status']==status for row in data['timeline']) for status in sorted({row['scenario']['status'] for row in data['timeline']})},gif=str(gif.relative_to(output)),gif_bytes=gif.stat().st_size,
            mp4=str(mp4.relative_to(output)) if mp4_ok else None,pngs=[str((out/f'{label}.png').relative_to(output)) for label in snapshots],
            viewer=str((out/'viewer.html').relative_to(output)),snapshots=snapshots,map_zoom=data['zoom']))
        print(json.dumps(results[-1]),flush=True);del data,bundle;gc.collect()
    caption=('Integrated demo v2: reference plus scripted Gaussian position noise acts as simulated tracker output '
        '(day sigma1m/axis, night sigma3m/axis). A frozen model generated32 joint futures from6s history, forecasting30s every2s. '
        'The displayed sample is the GT-free joint3D medoid. Orange object boxes mark current state scores above the frozen cutoff; no10s latch. '
        'Object reasons describe the largest reference-relative location/height, speed or direction score component. '
        'Scenario status: Normal traffic; Unusual activity; Potential intrusion when a currently anomalous object has at least8/32 forecasts entering the zone; '
        'Intrusion detected when a currently anomalous object is already inside. Forecast absence is explicitly unknown. '
        'No threat score, separate intent card or future anomaly warning is displayed. '
        'Protected zone and observation noise are scripted; the anomaly density has no time-of-day/weather conditioning. '
        'These are illustrative sample-based warnings, not calibrated risk probabilities. This revision only reinterprets saved D4 outputs, with no model execution.')
    d5=output/'D6';(d5/'CAPTION.md').write_text(caption+'\n')
    links=''.join(f'<li><a href="{r["scene"]}/viewer.html">{r["scene"]} interactive replay</a> · <a href="{r["scene"]}/{r["scene"]}.mp4">MP4</a> · <a href="{r["scene"]}/{r["scene"]}.gif">GIF</a></li>' for r in results)
    (d5/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SYGARD Demo</title><body style="background:#0b1622;color:#e8eef5;font:18px Arial;padding:40px"><h1>SYGARD / Integrated MVP</h1><ul>'+links+'</ul><p>Open locally. Each interactive replay embeds its own data; no server/network needed.</p></body>')
    (d5/'README.md').write_text('# Integrated demo v2\n\nOpen index.html locally for two offline viewers with play/pause,slider,32-rollout and reference-future toggles. '
        'MP4/GIF9fps (3× recorded3Hz time); PNG3200×1800. All media read accepted D4 only, without inference. '
        'All maps come from pre-existing local cache; no map/network requests.\n\n'+caption+'\n')
    assert before==source_hashes(output);verify_lock(root)
    write_json(d5/'render_receipt.json',dict(stage='D6',revision='current-only object anomaly + scenario status',media=results,d4_files_unchanged=True,model_execution=False,network_requests=False,caption=caption))
    archive=output/'integrated_demo_v2.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for p in sorted(d5.rglob('*')):
            if p.is_file():z.write(p,arcname=str(p.relative_to(d5)))
    print(json.dumps(dict(media_complete=True,archive=str(archive))),flush=True)


if __name__=='__main__':main()
