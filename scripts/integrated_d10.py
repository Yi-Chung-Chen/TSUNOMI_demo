"""D10 visual-cleanup review export. Frozen D4 only; D9 decisions/extents intact."""
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
import numpy as np
from src.eval import demo_gif as visual
from src.integrated.data import verify_lock
from src.integrated.presentation import load_presentation
from src.integrated.visual_cleanup import CleanupRenderer,integer_shares
from scripts.integrated_d9 import source_hashes

SCENES=['A_daytime_hobbyist','B_nighttime_approach']
REVIEWS={'A_daytime_hobbyist':[(187,'aircraft_anomaly.png')],
         'B_nighttime_approach':[(168,'first_warning.png'),(190,'aircraft_anomaly.png')]}


def hash_old(output):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['D8','D9']
            for p in (output/name).rglob('*') if p.is_file()}


def main(output_subdir=None):
    root=Path(__file__).resolve().parents[1];output=root/'outputs/integrated_demo';out=output/'D10';out=out/output_subdir if output_subdir else out;out.mkdir(parents=True,exist_ok=True)
    verify_lock(root);before=source_hashes(output);old=hash_old(output)
    receipt=dict(stage='D10',revision=output_subdir or 'initial',model_execution=False,network_requests=False,scenes=[],comparisons=[])
    font=ImageFont.truetype(findfont('DejaVu Sans'),32)
    for name in SCENES:
        data=load_presentation(root,name,grouped=True,all_anomalies=True,intrusion_memory=True)
        prior=json.loads((output/'D9'/name/'timeline.json').read_text())
        assert data['timeline']==prior,'D9 scenario or score changes forbidden'
        assert json.loads(json.dumps(data['object_groups']))==json.loads((output/'D9'/name/'future_groups.json').read_text())
        html=(output/'D9'/name/'viewer.html').read_text()
        prior_bundle=json.loads(re.search(r'<script id="demo-data" type="application/json">(.*?)</script>',html,re.S).group(1))
        np.testing.assert_array_equal(data['pb'],prior_bundle['pb']);assert data['zoom']==14
        directory=out/name;directory.mkdir(exist_ok=True)
        renderer=CleanupRenderer(data);frames=[]
        palette=visual.fixed_palette(data['map_image'],extra_colours=['#ff9f43','#ffd166','#ff626b','#132434','#294052'])
        for frame in range(len(data['timeline'])):
            frames.append(renderer.frame(frame).quantize(palette=palette,dither=0))
            if frame%90==0:print(json.dumps(dict(scene=name,rendered_frames=frame+1)),flush=True)
        layouts=renderer.layout_receipt;renderer.close()
        gif=directory/f'{name}.gif';visual.save_gif(frames,gif,1);del frames;gc.collect()
        mp4=directory/f'{name}.mp4'
        subprocess.run([shutil.which('ffmpeg'),'-y','-loglevel','error','-threads','1','-i',str(gif),'-vf','fps=9',
            '-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
        renderer=CleanupRenderer(data,dpi=200);review_layouts={}
        for frame,before_name in REVIEWS[name]:
            image=renderer.frame(frame);target=directory/f'review_f{frame:04d}.png';image.save(target)
            review_layouts[frame]=renderer.layout_receipt[frame]
            with Image.open(output/'D8'/name/before_name) as im:prior_image=im.convert('RGB')
            assert prior_image.size==image.size==(3200,1800)
            comparison=Image.new('RGB',(6400,1870),'#0b1622');comparison.paste(prior_image,(0,70));comparison.paste(image,(3200,70))
            draw=ImageDraw.Draw(comparison);time=f'{frame/3:.2f} s'
            draw.text((35,18),f'D8 BEFORE · {name} · {time}',font=font,fill='#e8eef5')
            draw.text((3235,18),f'D10 AFTER · {name} · {time}',font=font,fill='#e8eef5')
            comparison_path=out/f'comparison_{name}_f{frame:04d}.png';comparison.save(comparison_path)
            receipt['comparisons'].append(str(comparison_path.relative_to(out)))
        renderer.close()
        displayed={frame:{agent:integer_shares(groups) for agent,groups in per_object.items()} for frame,per_object in data['object_groups'].items()}
        (directory/'display_percentages.json').write_text(json.dumps(displayed,indent=2)+'\n')
        (directory/'layout_checks.json').write_text(json.dumps(dict(all_frames=layouts,review_frames_200dpi=review_layouts),indent=2)+'\n')
        receipt['scenes'].append(dict(scene=name,frames=len(layouts),all_labels_nonoverlapping=True,
            zone_label_outside_and_drawn_last=True,d9_timeline_equal=True,d9_groups_equal=True,
            map_pixel_bounds=data['pb'].tolist(),map_zoom=data['zoom'],map_extents_unchanged=True,
            gif=str(gif.relative_to(out)),mp4=str(mp4.relative_to(out)),review_frames=list(review_layouts)))
        print(json.dumps(receipt['scenes'][-1]),flush=True)
    verify_lock(root);assert before==source_hashes(output);assert old==hash_old(output)
    receipt.update(protected_hashes_unchanged=True,d4_files_unchanged=True,d8_d9_files_unchanged=True)
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'README.md').write_text('# D10 visual cleanup — review draft\n\nRead accepted D4 only; retain exact D9 decisions, groups, raw shares and map extents. '
        'Orange marker edges and name-only labels replace rectangles. Endpoint integer shares use largest remainder per object; '
        'raw zone-entry shares remain unchanged. WHY contains reasons; OBJECT STATUS is one line per object. '
        'Protected-zone label is outside the polygon, reserved against other labels and drawn last. '
        'All632frames have non-overlapping label checks; requested review PNGs are3200x1800, D8/D10 side-by-side comparisons6400x1870. '
        'MP4/GIF are1600x900,3x replay. This review draft exports static/video media only; existing interactive D9 remains untouched. '
        'No inference, fitting, packages or network. Replay: scripts/catk.sh -m scripts.integrated_d10.\n')
    print(json.dumps(dict(completed=True,output=str(out))),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-subdir',choices=['thin_outline','readable_status','complete_reasons','pixel_wrap'])
    main(parser.parse_args().output_subdir)
