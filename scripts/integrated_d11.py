"""D11 audit and offline D10-style export; never invokes model inference."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont
from src.eval import demo_gif as visual
from src.integrated.data import verify_lock
from src.integrated.presentation import load_presentation
from src.integrated.visual_cleanup import CleanupRenderer, integer_shares
from src.integrated.story import write_json

SCENES = ['A_daytime_hobbyist', 'B_nighttime_approach']


def protected_check(root, out):
    hashes = json.loads((out/'protected_before.json').read_text())
    for name, digest in hashes.items():
        with (root/name).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == digest, name
    verify_lock(root)
    return len(hashes)


def assert_invariants(old, new):
    """Forecast evidence may differ; state evidence and incident memory must not."""
    assert old['stream'].keys()==new['stream'].keys()
    for key in old['stream']:
        np.testing.assert_array_equal(old['stream'][key], new['stream'][key])
    for key in ['cutoff_raw', 'cutoff_z', 'scene', 'zone']:
        assert old['manifest'][key] == new['manifest'][key], key
    assert len(old['timeline']) == len(new['timeline'])
    for a, b in zip(old['timeline'], new['timeline']):
        assert a['frame'] == b['frame'] and a['time_s'] == b['time_s']
        assert len(a['objects'])==len(b['objects'])
        for x, y in zip(a['objects'], b['objects']):
            for key in ['id', 'current_state_nll', 'z_state', 'current_exceeds_cutoff',
                        'anomalous', 'score_components', 'explanation', 'anomaly_reason']:
                assert x[key] == y[key], (a['frame'], key)
        for key in ['incident_latched', 'incidents']:
            assert a['scenario'][key] == b['scenario'][key], (a['frame'], key)
    np.testing.assert_array_equal(old['pb'], new['pb'])
    assert old['zoom'] == new['zoom']


def first_potential(data):
    return next((r['time_s'] for r in data['timeline']
                 if r['scenario']['status'] == 'Potential intrusion'), None)


def render(root, run_dir, out):
    root=Path(root);run_dir=root/run_dir;out=root/out;out.mkdir(parents=True,exist_ok=True)
    protected_count=protected_check(root,out)
    config=json.loads((root/'configs/integrated_demo/v2_anchor.json').read_text())
    base=json.loads((root/'configs/integrated_demo/v1.json').read_text())
    assert {k for k in config if config[k]!=base[k]}=={'version','checkpoint'}
    gate=json.loads((run_dir/'D4_acceptance.json').read_text())
    old_gate=json.loads((root/'outputs/integrated_demo/D4_acceptance.json').read_text())
    assert gate['attempt']==1 and not gate['retuned']
    receipt=dict(stage='D11',checkpoint=config['checkpoint'],inference_attempts=1,
                 d4_acceptance=gate,old_d4_acceptance=old_gate,scenes=[],comparisons=[])
    rows=[];report=[]
    font=ImageFont.truetype(findfont('DejaVu Sans'),32)
    for index,name in enumerate(SCENES):
        old=load_presentation(root,name,grouped=True,all_anomalies=True,intrusion_memory=True)
        data=load_presentation(root,name,grouped=True,all_anomalies=True,intrusion_memory=True,
                               run_dir=run_dir,fixed_view=old)
        assert_invariants(old,data)
        for filename in ['observations.npz','current_scores.npz']:
            with np.load(root/'outputs/integrated_demo'/name/filename) as a, np.load(run_dir/name/filename) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        # Compare accepted D10's rounded shares as well as its exact map extent.
        accepted=root/'outputs/integrated_demo/D10/readable_status'/name
        expected={str(f):{str(a):integer_shares(g) for a,g in per.items()} for f,per in old['object_groups'].items()}
        assert expected==json.loads((accepted/'display_percentages.json').read_text())
        prior_receipt=json.loads((accepted.parent/'render_receipt.json').read_text())
        prior_scene=next(x for x in prior_receipt['scenes'] if x['scene']==name)
        np.testing.assert_array_equal(data['pb'],prior_scene['map_pixel_bounds'])
        assert data['zoom']==prior_scene['map_zoom']
        directory=out/name;directory.mkdir(exist_ok=True)
        write_json(directory/'timeline.json',data['timeline']);write_json(directory/'future_groups.json',data['object_groups'])
        changes=[]
        for frame,pred in data['forecasts'].items():
            before=old['forecasts'][frame]
            changes.append(float(np.linalg.norm(pred['position'][:,1]-before['position'][:,1],axis=-1).mean()))
        for a,b in zip(old['timeline'],data['timeline']):
            for agent in range(len(a['objects'])):
                x,y=a['objects'][agent]['entry'],b['objects'][agent]['entry']
                rows.append(dict(scenario=name,time_s=a['time_s'],object_id=a['objects'][agent]['id'],
                    old_entry_share=x['entry_samples']/32 if x else None,
                    anchor_entry_share=y['entry_samples']/32 if y else None,
                    old_status=a['scenario']['status'],anchor_status=b['scenario']['status']))
        letter='AB'[index];oe=old_gate[letter];ne=gate[letter]
        op,np_=first_potential(old),first_potential(data)
        entry=ne['reference_entry_s']
        item=dict(scene=name,first_potential_old_s=op,first_potential_anchor_s=np_,
                  potential_lead_old_s=entry-op if entry is not None and op is not None else None,
                  potential_lead_anchor_s=entry-np_ if entry is not None and np_ is not None else None,
                  background_mean_paired_sample_change_m=float(np.mean(changes)),
                  observations_equal=True,state_scores_equal=True,cutoffs_equal=True,orange_decisions_equal=True,
                  observed_intrusion_latch_equal=True,map_extents_equal=True)
        report.append(f"| {letter} | {op} | {np_} | {oe['model_warning_lead_s']} | {ne['model_warning_lead_s']} | {item['background_mean_paired_sample_change_m']:.2f} |")
        renderer=CleanupRenderer(data);frames=[]
        palette=visual.fixed_palette(data['map_image'],extra_colours=['#ff9f43','#ffd166','#ff626b','#132434','#294052'])
        for frame in range(len(data['timeline'])):
            frames.append(renderer.frame(frame).quantize(palette=palette,dither=0))
            if frame%90==0:print(json.dumps(dict(scene=name,rendered_frames=frame+1)),flush=True)
        layouts=renderer.layout_receipt;renderer.close()
        gif=directory/f'{name}.gif';visual.save_gif(frames,gif,1);del frames;gc.collect()
        mp4=directory/f'{name}.mp4'
        subprocess.run([shutil.which('ffmpeg'),'-y','-loglevel','error','-threads','1','-i',str(gif),'-vf','fps=9',
                        '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(mp4)],check=True)
        review={187} if index==0 else {168,190}
        if index==1 and ne['first_model_based_preentry_high_s'] is not None:
            review.add(round(ne['first_model_based_preentry_high_s']*3))
        old_renderer=CleanupRenderer(old,dpi=200);renderer=CleanupRenderer(data,dpi=200)
        for frame in sorted(review):
            image=renderer.frame(frame);image.save(directory/f'review_f{frame:04d}.png')
            prior_path=accepted/f'review_f{frame:04d}.png'
            if prior_path.exists():
                with Image.open(prior_path) as im:prior=im.convert('RGB')
            else:prior=old_renderer.frame(frame)
            comparison=Image.new('RGB',(6400,1870),'#0b1622');comparison.paste(prior,(0,70));comparison.paste(image,(3200,70))
            draw=ImageDraw.Draw(comparison)
            draw.text((35,18),f'D10 / epoch016 · {name} · {frame/3:.2f} s',font=font,fill='#e8eef5')
            draw.text((3235,18),f'D11 / 16 runway-end anchors · {name} · {frame/3:.2f} s',font=font,fill='#e8eef5')
            target=out/f'comparison_{name}_f{frame:04d}.png';comparison.save(target);receipt['comparisons'].append(target.name)
        write_json(directory/'layout_checks.json',dict(all_frames=layouts,review_frames=renderer.layout_receipt))
        write_json(directory/'display_percentages.json',{f:{a:integer_shares(g) for a,g in per.items()} for f,per in data['object_groups'].items()})
        renderer.close();old_renderer.close();item.update(frames=len(layouts),review_frames=sorted(review));receipt['scenes'].append(item)
        print(json.dumps(item),flush=True)
    with (out/'zone_entry_over_time.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,name in zip(axes,SCENES):
        # Each scenario's foreground object is the first row of its own timeline.
        ids=[r['object_id'] for r in rows if r['scenario']==name];subset=[r for r in rows if r['scenario']==name and r['object_id']==ids[0]]
        for key,label in [('old_entry_share','epoch016'),('anchor_entry_share','anchor018')]:
            ax.plot([r['time_s'] for r in subset],[np.nan if r[key] is None else r[key] for r in subset],label=label)
        ax.axhline(.25,color='gray',linestyle=':',label='8/32 warning threshold');ax.set(title=name,ylabel='Drone zone-entry share',ylim=(-.02,1.02));ax.legend()
    axes[-1].set_xlabel('Scenario time (s)');fig.tight_layout();fig.savefig(out/'zone_entry_over_time.png',dpi=180);plt.close(fig)
    comparison=('''# D11 checkpoint comparison

Only the forecast checkpoint changed: epoch016 → M10 epoch018, with 16 runway-end anchors. Same seed, observations, noise, frozen density, cutoff, routes, zone, rules, 32 samples, top-k 5 and temperature 1. No tuning or re-training.

| Scenario | First Potential intrusion, old (s) | Anchor (s) | D4 pre-entry high lead, old (s) | Anchor (s) | Background paired sample change (m) |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(report)+'''

`None` means no qualifying warning/entry. D4 gate uses its original numeric rule with 10 s anomaly memory; displayed Potential intrusion uses the accepted current-only D10 rule. These are reported separately in acceptance.json. Warning lead is relative to reference entry (B: 80.6667 s), not observed noisy entry. Persistent High threat is the observed D9 incident latch, not the forecast warning.

Background change is mean 3D distance between corresponding sample indices across forecast origins, agents fixed to the background aircraft; it describes changes, not accuracy or a coupling of calibrated futures. Full per-object zone-entry shares at every 1/3 s are in zone_entry_over_time.csv; foreground traces appear in zone_entry_over_time.png. Forecasts refresh every 2 s; intervening values assess remaining sampled paths.

Assertions passed exactly: all observation/reference arrays and validity/times/IDs, all current score arrays/components/explanations, cutoffs, per-object orange decisions, and per-frame observed incident flags/IDs/first detection times. All original protected files remain SHA256-identical. Forecasts, medoids, groups, shares and forecast-derived scenario status may change. D10 readable_status renderer and map extents are retained. Interactive D9 viewer is unchanged.

Sample shares are fractions of 32 futures, not calibrated probabilities. State density has location/height, speed and course context; time-of-day and noise are scripted, not learned detector context.
''')
    (out/'comparison.md').write_text(comparison)
    b=gate['B'];warning=(f"B pre-entry model warning: {b['first_model_based_preentry_high_s']} s; lead {b['model_warning_lead_s']} s." if b['first_model_based_preentry_high_s'] is not None else 'B FAILED the pre-entry model-warning gate. No qualifying warning before entry; no tuning was performed.')
    caption='SYGARD integrated MVP: M10 trajectory forecasts use 16 runway-end anchors. Orange outlines indicate current state anomalies; endpoint percentages are rounded sample shares. Scenario High persists after an observed anomalous intrusion. Synthetic tracker observations and fictional protected zone; scripted daytime/nighttime and Gaussian noise.\n'
    (out/'caption.txt').write_text(caption)
    (out/'README.md').write_text('# Integrated demo v6 — anchor checkpoint\n\n'+warning+'\n\n'+caption+'\nMP4/GIF: 1600×900, 3× replay; review PNG: 3200×1800. Compare D10/epoch016 and D11/anchor side by side in comparison_*.png. See comparison.md and zone_entry_over_time.csv/png for results. Observations, state anomalies and incident latch are unchanged. Interactive viewer remains the D9 version. No scenario or detector tuning.\n')
    receipt.update(protected_files_unchanged=protected_check(root,out),all_invariants_passed=True,
                   media_style='unchanged D10 readable_status',model_execution_in_renderer=False)
    write_json(out/'acceptance.json',receipt)
    archive=package(out)
    protected_check(root,out)
    print(json.dumps(dict(completed=True,archive=str(archive),B=b)),flush=True)


def package(out):
    """Package media and final review evidence, excluding raw inference/logs."""
    archive=out.parent/'integrated_demo_v6.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and 'D4_anchor' not in p.relative_to(out).parts and p.name!='protected_before.json' and p.suffix!='.log':
                z.write(p,Path('integrated_demo_v6')/p.relative_to(out))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            path=out/Path(name).relative_to('integrated_demo_v6')
            assert hashlib.sha256(z.read(name)).digest()==hashlib.sha256(path.read_bytes()).digest()
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='outputs/integrated_demo/D11/D4_anchor')
    parser.add_argument('--output-dir',default='outputs/integrated_demo/D11')
    a=parser.parse_args();render(Path(__file__).resolve().parents[1],Path(a.run_dir),Path(a.output_dir))
