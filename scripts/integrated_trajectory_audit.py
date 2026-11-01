"""Read-only plan-view curvature audit of all saved D4 model paths, not CV."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1];output=root/'outputs/integrated_demo';result={}
    for name in ['A_daytime_hobbyist','B_nighttime_approach']:
        records=json.loads((output/name/'forecasts.json').read_text())
        with np.load(output/name/'observations.npz') as a:observed=a['observed'].copy()
        deviations=[[],[]];ratios=[[],[]];turns=[[],[]];stationary=[0,0]
        for record in records:
            with np.load(output/record['prediction_file']) as a:prediction=a['position'].copy()
            for agent in range(prediction.shape[1]):
                p=np.concatenate([np.broadcast_to(observed[agent,record['frame'],:2],(len(prediction),1,2)),prediction[:,agent,:,:2]],axis=1)
                v=p[:,-1]-p[:,0];chord=np.linalg.norm(v,axis=-1)
                u=p-p[:,:1];fraction=np.clip((u*v[:,None]).sum(-1)/np.maximum(chord[:,None]**2,1e-12),0,1)
                deviation=np.linalg.norm(u-fraction[...,None]*v[:,None],axis=-1).max(axis=1)
                length=np.linalg.norm(np.diff(p,axis=1),axis=-1).sum(axis=1)
                start=p[:,6]-p[:,0];end=p[:,-1]-p[:,-7]
                angle=np.abs(np.arctan2(start[:,0]*end[:,1]-start[:,1]*end[:,0],(start*end).sum(-1)))*180/np.pi
                moving=chord>=30;stationary[agent]+=int((~moving).sum())
                deviations[agent].extend(deviation.tolist());ratios[agent].extend((length[moving]/chord[moving]).tolist());turns[agent].extend(angle[moving].tolist())
        objects=[]
        for agent in range(2):
            d=np.asarray(deviations[agent]);r=np.asarray(ratios[agent]);t=np.asarray(turns[agent])
            objects.append(dict(object='Drone 01' if agent==0 else 'Aircraft 01',sampled_paths=len(d),
                max_plan_view_deviation_m=dict(median=float(np.median(d)),p90=float(np.percentile(d,90)),max=float(d.max())),
                fraction_deviation_over25m=float(np.mean(d>25)),
                moving_paths_chord_at_least30m=len(r),near_stationary_paths=stationary[agent],
                moving_path_length_over_chord=dict(median=float(np.median(r)),p90=float(np.percentile(r,90))),
                moving_heading_change_degrees=dict(median=float(np.median(t)),p90=float(np.percentile(t,90)))))
        result[name]=objects
    report=dict(source='All saved D4 model position samples, observed anchor +90future frames; no CV or GT future',
        interpretation='Descriptive curvature audit; thresholds25m/30m are reporting definitions, not model or display tuning.',scenes=result)
    (root/'docs/integrated_demo_D8_trajectory_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
