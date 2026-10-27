"""Frozen-checkpoint inference from native camera estimates; GT is overlay-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from src.data.camera_trajectory import (load_camera_tracks,first_camera_origin,make_camera_graph,
    sample_camera,load_reference_tracks,sample_reference,camera_features)
from src.model.air_smart import AirSMART
from src.rollout.closed_loop import rollout
from src.eval.camera_rollout import render_camera_scene
from src.eval.demo_gif import save_gif
from scripts.merge_demo_val import protected_hashes


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def render_saved(root,gifs=False):
    output=root/'outputs/camera_demo'
    manifest=json.loads((output/'manifest.json').read_text())
    for case in manifest['cases']:
        with np.load(output/case['output_npz'],allow_pickle=False) as arrays:
            data={key:arrays[key] for key in arrays.files}
        frames,view=render_camera_scene(data,root/'data/basemap_cache',snapshot=True,dpi=200)
        frames[0].save(output/case['output_png']);case['view']=view
        if gifs:
            frames,_=render_camera_scene(data,root/'data/basemap_cache')
            path=output/'gifs'/Path(case['output_png']).with_suffix('.gif').name
            path.parent.mkdir(exist_ok=True);save_gif(frames,path,1)
        print(json.dumps(dict(rendered=case['output_png'])),flush=True)
    manifest['presentation']='cyan solid camera, yellow dashed prediction, green dotted GT'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with ZipFile(output/'camera_demo_images.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted((output/'scenes').glob('*.png')):archive.write(path,arcname=path.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dataset-root',type=Path,default=Path('../SYGARD-Dataset'))
    parser.add_argument('--max-cases',type=int,default=0)
    parser.add_argument('--gifs',action='store_true')
    parser.add_argument('--render-only',action='store_true',help='Read saved arrays only; no model execution')
    args=parser.parse_args();root=args.root.resolve();dataset_root=(root/args.dataset_root).resolve()
    output=root/'outputs/camera_demo';output.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        render_saved(root,args.gifs);return
    torch.set_num_threads(1);started=time.monotonic();protected=protected_hashes(root)
    checkpoint=root/'outputs/air_pre_bc/checkpoints/epoch_016.ckpt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=AirSMART(**saved['hyper_parameters']);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    audit=[];cases=[];inputs={}
    for kind in ['aircraft','drone']:
        folder=dataset_root/f'ord_{kind}_2026-09-28_1800z_3hz'
        paths=[folder/name for name in ['camera_tracks_native.jsonl','simulation_poses_native.jsonl','metadata.json']]
        inputs.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        meta=json.loads(paths[2].read_text());origin=meta['coordinate_frame']['origin_wgs84']
        assert origin['latitude_deg']==41.980103 and origin['longitude_deg']==-87.903868
        tracks=load_camera_tracks(paths[0],origin['altitude_m'])
        # Only GT visualization uses these offline identities/reference arrays.
        mapping={int(r['tracker_id']):r.get('matched_source_object_id') for r in meta['track_ids']['camera_tracks']}
        references=load_reference_tracks(paths[1],origin['altitude_m'])
        for track_id,track in tracks.items():
            frame=first_camera_origin(track);matched=mapping.get(track_id)
            reason='eligible' if frame is not None and matched in references else ('unmapped for GT overlay' if not matched else 'insufficient fresh measured camera history')
            item=dict(kind=kind,tracker_id=track_id,matched_source_object_id=matched or '',native_start_s=float(track['times'][0]),
                native_end_s=float(track['times'][-1]),distinct_timestamps=len(track['times']),
                measured_timestamps=int(track['measured'].sum()),first_eligible_frame=frame,selection=reason)
            audit.append(item)
            if reason!='eligible':continue
            if args.max_cases and len(cases)>=args.max_cases:continue
            graph,history=make_camera_graph(track,frame,pos_scale=model.token_processor.meta['pos_scale'])
            # This call receives only camera-derived history; GT is not in graph.
            prediction=rollout(model,graph,n_rollouts=1,num_k=5,temp=1.,seed=817)
            query=np.arange(frame-18,frame+91)/3
            camera=sample_camera(track,query)
            camera['valid'] &= query<=track['times'][-1]  # display stops at track lifetime
            gt,gt_valid=sample_reference(references[matched],query)
            coverage=float(camera['valid'][19:].sum()/3)
            features=camera_features(history);ground=features[1]
            case=dict(kind=kind,tracker_id=track_id,current_frame=frame,current_time_s=frame/3,
                history_start_s=(frame-18)/3,gt_object_id=matched,
                camera_future_coverage_s=coverage,gt_future_coverage_s=float(gt_valid[19:].sum()/3),
                ground_history_flips=int(np.count_nonzero(np.diff(ground.astype(int)))),ground_at_origin=bool(ground[-1]),
                history_max_age_s=float(history['age_s'].max()),type='UNK',seed=817,sample_index=0,
                current_camera_position_m=history['position'][-1].tolist())
            data=dict(kind=np.asarray(kind),tracker_id=np.asarray(track_id),current_frame=np.asarray(frame),
                query_s=query,camera_position=camera['position'],camera_valid=camera['valid'],
                camera_source_time_s=camera['source_time_s'],camera_source_measured=camera['final_measured'],
                camera_position_std_m=camera['std'],gt_position=gt,gt_valid=gt_valid,
                prediction=prediction['position'][:,0].cpu().numpy(),prediction_heading=prediction['heading'][:,0].cpu().numpy(),
                camera_future_coverage_s=np.asarray(coverage),history_ground=ground,history_heading=features[0],
                meta_json=np.asarray(json.dumps(case)))
            name=f'{kind}_track_{track_id:02d}_t{frame:04d}'
            destination=output/'rollouts'/f'{name}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination,**data)
            frames,view=render_camera_scene(data,root/'data/basemap_cache',snapshot=True,dpi=200)
            (output/'scenes').mkdir(exist_ok=True);frames[0].save(output/'scenes'/f'{name}.png')
            if args.gifs:
                frames,_=render_camera_scene(data,root/'data/basemap_cache')
                (output/'gifs').mkdir(exist_ok=True);save_gif(frames,output/'gifs'/f'{name}.gif',1)
            case.update(output_png=f'scenes/{name}.png',output_npz=f'rollouts/{name}.npz',view=view)
            cases.append(case);print(json.dumps(case),flush=True)
    write_csv(output/'track_audit.csv',audit)
    assert protected_hashes(root)==protected
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in inputs.items())
    manifest=dict(checkpoint_epoch=saved['epoch'],checkpoint_sha256=protected['outputs/air_pre_bc/checkpoints/epoch_016.ckpt'],
        model_execution=True,training=False,device='cpu',seed=817,n_rollouts=1,num_k=5,temperature=1.,
        prediction_selection='sample0; no error-based selection',gt_usage='offline identity association and overlay only',
        sampling='last event at/before query plus its own velocity * age; age<=1s',
        eligibility='19 valid history states,>=3 measured timestamps, last measurement<=1s, skip zero-velocity uninitialized states',
        ground_proxy='camera height above166m reference: enter<10m exit>20m; unvalidated fixed hysteresis',
        single_agent_scenes=True,protected_unchanged=True,input_sha256=inputs,protected_sha256=protected,
        elapsed_s=time.monotonic()-started,track_count=len(audit),case_count=len(cases),cases=cases)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Camera-conditioned trajectory demo\n\n'
        'Camera tracker (cyan), one sampled model prediction (yellow dashed), simulation GT (green). '
        'Model receives6s of camera-derived history and predicts30s. Camera includes Kalman predicted states, '
        'not only fresh detections, and ends when tracking ends. Stopped endpoint icons are labelled with their last time. '
        'One single-agent scene per offline-verified tracker ID; no GT-based stitching or subtype/heading/ground inputs. '
        'Offline segmentation identity is used only to pair the reference overlay. '
        'Types are UNK; ground flags are a fixed camera-height proxy, not known truth. '
        'Sample0 of1, seed817, top-k5, temperature1; no minADE selection.\n')
    with ZipFile(output/'camera_demo_images.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted((output/'scenes').glob('*.png')):archive.write(path,arcname=path.name)
    print(json.dumps(dict(case_count=len(cases),track_count=len(audit),elapsed_s=manifest['elapsed_s'])),flush=True)


if __name__=='__main__':main()
