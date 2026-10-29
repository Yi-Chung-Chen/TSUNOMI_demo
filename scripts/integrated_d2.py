"""D2 exact scorer reproduction and forecast scoring acceptance, no fitting."""
import csv
import json
from pathlib import Path
import numpy as np
from src.integrated.data import verify_lock
from src.integrated.geometry import forecast_intent
from src.integrated.scoring import FrozenStateScorer,COMPONENTS,future_anomaly_evidence


def main():
    root=Path(__file__).resolve().parents[1];verify_lock(root)
    config=json.loads((root/'configs/integrated_demo/v1.json').read_text())
    scorer=FrozenStateScorer(root/'outputs/anomaly_state_smoothed')
    with (root/'outputs/anomaly_state_smoothed/token_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    position=np.array([[float(r[k]) for k in ['x_m','y_m','height_m']] for r in rows])
    speed=np.array([float(r['speed_mps']) for r in rows]);course=np.array([float(r['course_rad']) for r in rows])
    direct=scorer.model.score(dict(position=position,speed=speed,course=course))
    errors={key:float(np.max(np.abs(direct[key]-np.array([float(r[key]) for r in rows])))) for key in COMPONENTS}
    assert max(errors.values())<1e-10
    out=root/'outputs/integrated_demo/D2';out.mkdir(parents=True,exist_ok=True);smokes=[]
    for path in sorted((root/'outputs/integrated_demo/D1').glob('*.npz')):
        with np.load(path) as a:
            frame=int(a['current_frame']);current=scorer.states(a['observed'][:,frame],a['observed'][:,frame-6])
            future=scorer.futures(a['prediction'],a['observed'][:,frame])
            evidence=future_anomaly_evidence(future['state_nll'],scorer.cutoff)
            np.savez_compressed(out/path.name,**{f'future_{key}':value for key,value in future.items()},
                **{f'current_{key}':value for key,value in current.items()})
            smokes.append(dict(case=path.stem,current_state_nll=current['state_nll'].tolist(),future=evidence,
                intent=forecast_intent(a['prediction'],config['zone']),endpoints=a['prediction'][:,:,-1].tolist()))
    receipt=dict(stage='D2',passed=True,reproduced_rows=len(rows),component_max_absolute_error=errors,
        cutoff_raw=scorer.cutoff,cutoff_z=scorer.z_cutoff,density_refit=False,smokes=smokes)
    verify_lock(root);(root/'docs/integrated_demo_D2.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
