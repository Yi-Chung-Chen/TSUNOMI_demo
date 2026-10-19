"""Dark-map completed-rollout stills, matching the existing v2 simulation GIFs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from src.eval.demo_gif import render_window
from src.eval.load_rollout import load_rollout


def main():
    root=Path(__file__).resolve().parents[1];output=root/'outputs/rollout_demo_stills';output.mkdir(parents=True,exist_ok=True)
    baseline=root/'outputs/demo_gif_v2/manifest.json'
    records=[]
    for case in json.loads(baseline.read_text())['windows']:
        source=root/case['input'];before=hashlib.sha256(source.read_bytes()).hexdigest()
        data=load_rollout(source)
        frames,record=render_window(data,root/'data/basemap_cache',snapshot=True,presentation=True,dpi=200)
        assert record['rollout_index']==case['rollout_index']
        name=Path(case['output']).with_suffix('.png').name
        frames[0].save(output/name,dpi=(200,200))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==before
        records.append(dict(**record,output=name,input=case['input'],input_sha256=before))
        print(name,flush=True)
    manifest=dict(source='outputs/demo_gif_v2/manifest.json',model_execution=False,training=False,
        snapshot_time_s=30,frame_index=108,size=[2400,1800],theme='dark',
        rollout_selection='Same as existing v2 GIF: lowest scene-average 3D error of32 joint rollouts; uses recorded future for selection',
        figures=records)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'CAPTION.md').write_text('Aircraft motion simulation on SYGARD O’Hare data. Solid paths and filled aircraft show '
        'recorded trajectories; dashed paths and outlined aircraft show the simulated trajectory at the end of the30-second forecast, '
        'conditioned on6seconds of recorded history. Colours distinguish aircraft. '
        'The joint rollout is the same illustrative best-of-32 scene sample shown in the existing v2 animation, '
        'selected using recorded-future scene-average3D error; it is not an unbiased typical prediction. '
        'Predicted paths end at the first missing recorded observation, following the existing GIF policy.\n')
    with ZipFile(output/'rollout_demo_stills.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix in ['.png','.md','.json']:archive.write(path,arcname=path.name)
    print('Exported10 completed-rollout dark-map stills; no model execution.',flush=True)


if __name__=='__main__':main()
