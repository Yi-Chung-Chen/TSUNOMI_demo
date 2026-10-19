"""Presentation export from saved scores and GT trajectories; no fitting/inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from src.eval.anomaly_state import write_combined_gifs


def package_images(output):
    with ZipFile(output/'scene_images.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted((output/'scenes').glob('*.png')):
            archive.write(path,arcname=path.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/anomaly_demo'))
    args=parser.parse_args();root=args.root.resolve();output=root/args.output_dir
    source=root/'outputs/anomaly_state_smoothed'
    if output.resolve() in [(root/'outputs'/name).resolve() for name in ['anomaly_state','anomaly_state_smoothed','anomaly_state_height100']]:
        parser.error('Use a separate demo directory to preserve scientific results')
    inputs=[source/'token_scores.csv',source/'summary.csv']
    before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    with inputs[0].open() as stream: rows=list(csv.DictReader(stream))
    with inputs[1].open() as stream: summary=list(csv.DictReader(stream))
    for row in summary: row['p95']=float(row['p95'])
    output.mkdir(parents=True,exist_ok=True)
    result=write_combined_gifs(root,rows,summary,output,presentation=True)
    assert before=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    result.update(source='outputs/anomaly_state_smoothed',input_sha256=before,model_execution=False,
        training=False,density_fitting=False,individual_pngs=len(result['gifs']),
        snapshot_selection='Last displayed frame of each existing scene, matching contact sheet',
        labels='Anomaly score; scene-local display names; no interval, model formula or internal identifiers')
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'README.md').write_text('# SYGARD anomaly demo\n\n'
        'scenes/: one full-resolution PNG per scene. gifs/: matching animations and contact_sheet.png.\n\n'
        'Uses the restored five-height smoothed baseline. Scores, alert cutoff and trajectory selection are unchanged. '
        'Labels use scene-local aircraft names; the manifest maps them to original object IDs. '
        'Scores are relative values, not probabilities. Red paths retain past flagged intervals; '
        'a red object box represents its current completed score. Replay uses recorded trajectories, '
        'and the combined score includes a completed two-second transition.\n')
    package_images(output)
    print(json.dumps(dict(output=str(output),pngs=len(result['gifs']),threshold=result['threshold']),indent=2))


if __name__=='__main__':
    main()
