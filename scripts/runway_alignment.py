"""M10 step1 fixed alignment gate. Failure stops M10 before model/training work."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from src.common.runway_anchors import fixed_endpoints,alignment_masks,EXPECTED


def main():
    root=Path(__file__).resolve().parents[1];out=root/'outputs/runway_anchor';out.mkdir(exist_ok=True)
    reference=root.parent/'m10_reference/runway_alignment_reference_output.txt'
    assert (out/'reference_output_actual.txt').read_bytes()==reference.read_bytes(), 'Unmodified reference-output gate must pass first'
    dataset=root.parent/'SYGARD-Dataset/ord_aircraft_training_800_3hz'
    names={split:(root/f'data/splits/{split}_manifest.txt').read_text().splitlines() for split in ['train','val']}
    assert len(names['train'])==618 and len(names['val'])==182 and not set(names['train'])&set(names['val'])
    origin=json.loads((dataset/names['train'][0]/'metadata.json').read_text())['coordinate_frame']['origin_wgs84']
    anchors=fixed_endpoints(origin);pairs=[anchors[i:i+2] for i in range(0,16,2)]
    counts={name:dict(ground=0,air=0,union=0) for name in EXPECTED}
    heights={name:[] for name in EXPECTED};all_runway_heights=[];matches=[]
    for split,scenarios in names.items():
        for name in scenarios:
            metadata=json.loads((dataset/name/'metadata.json').read_text())
            assert metadata['coordinate_frame']['origin_wgs84']==origin
            with np.load(root/'data/processed'/f'{name}.npz') as a:
                d={key:a[key] for key in ['pos','heading','valid','on_ground','agent_id']}
            nearby_any=np.zeros_like(d['valid'])
            for a,b in pairs:
                runway=a['runway'];pa=[a['x_m'],a['y_m']];pb=[b['x_m'],b['y_m']]
                ground,air,nearby=alignment_masks(d['pos'],d['heading'],d['valid'],d['on_ground'],pa,pb)
                eligible=d['valid'].sum(axis=1)>=10
                n_ground=ground.sum(axis=1);n_air=air.sum(axis=1);g=(n_ground>=6)&eligible;f=(n_air>=6)&eligible
                counts[runway]['ground']+=int(g.sum());counts[runway]['air']+=int(f.sum());counts[runway]['union']+=int((g|f).sum())
                for i in np.flatnonzero(g|f):
                    matches.append(dict(scenario=name,split=split,agent_id=str(d['agent_id'][i]),runway=runway,
                        ground_frames=int(n_ground[i]),air_frames=int(n_air[i])))
                if split=='train':heights[runway].extend(d['pos'][...,2][nearby].tolist());nearby_any|=nearby
            if split=='train':all_runway_heights.extend(d['pos'][...,2][nearby_any].tolist())
    fallback=float(np.median(all_runway_heights));assert np.isfinite(fallback)
    z={name:float(np.median(values)) if values else fallback for name,values in heights.items()}
    for anchor in anchors:anchor['z_m']=z[anchor['runway']]
    csv_path=root/'assets/ord_runway_ends.csv'
    with csv_path.open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=list(anchors[0]));writer.writeheader();writer.writerows(anchors)
    with (out/'alignment_track_matches.csv').open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=['scenario','split','agent_id','runway','ground_frames','air_frames']);writer.writeheader();writer.writerows(matches)
    comparisons=[]
    for name,target in EXPECTED.items():
        actual=counts[name]['union'];passed=actual==0 if target==0 else abs(actual-target)<=.05*target
        comparisons.append(dict(runway=name,expected=target,actual=actual,ground=counts[name]['ground'],air=counts[name]['air'],
            relative_error=None if target==0 else (actual-target)/target,passed=bool(passed)))
    with (out/'alignment_counts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,lineterminator='\n',fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    protected=json.loads((out/'protected_before.json').read_text())
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==sha for p,sha in protected.items())
    result=dict(stage='M10 step1',passed=all(c['passed'] for c in comparisons),scenarios=800,origin=origin,
        coordinates_source='Fixed M10 table, OurAirports/public domain; no external coordinate substitution',
        xy_height_m=origin['altitude_m'],height_note='Existing verified exact-ENU anchor CSV convention; z separately train-only processed height',
        recipe=dict(unit='scenario+agent, independent per runway, ground/air union',ground_distance_m=40,ground_heading_cosine_abs_strict_gt=.97,
            ground_projection='finite segment (0,L)',air_distance_m=120,air_height_m='processed_z<500 (no lower bound)',
            air_projection='(-6000,0) union (L,L+6000)',air_heading_constraint='abs(cos(heading-axis))>0.97',
            frames='track>=10validframes; >=6 total separately for ground and air; not required consecutive',ground_height_radius_m=100),
        ground_height_by_runway=z,all_runway_ground_median=fallback,ground_height_train_frame_counts={k:len(v) for k,v in heights.items()},
        comparisons=comparisons,protected_files_unchanged=True,model_edited=False,training_started=False,
        count_reference_script_available=True,reference_output_exact=True)
    (out/'alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
