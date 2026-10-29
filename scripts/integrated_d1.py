"""D1 acceptance on existing aircraft/drone segments; never runs story B."""
import json
from pathlib import Path
import numpy as np
import torch
from src.integrated.data import verify_lock,noisy_observations,observation_graph,stable_seed
from src.integrated.forecast import load_model,infer,medoid_index,constant_velocity
from src.eval.anomaly_score import load_drone


def main():
    root=Path(__file__).resolve().parents[1];config=json.loads((root/'configs/integrated_demo/v1.json').read_text())
    torch.set_num_threads(1);verify_lock(root);model=load_model(root,config['checkpoint'])
    out=root/'outputs/integrated_demo/D1';out.mkdir(parents=True,exist_ok=True)
    with np.load(root/config['background']['path']) as arrays:
        i=list(arrays['agent_id']).index(config['background']['agent_id']);start=500
        aircraft=arrays['pos'][i:i+1,start:start+109].astype(float)
    drone,_=load_drone();drone=drone['pos'][:,6:115].astype(float)
    cases=[]
    for name,reference in [('aircraft',aircraft),('drone',drone)]:
        valid=np.ones(reference.shape[:2],bool)
        obs=noisy_observations(reference,[name],1,config['seed'],'D1')
        np.testing.assert_array_equal(noisy_observations(reference,[name],0,817,name),reference)
        frame=18;seed=stable_seed(config['seed'],'D1',name,frame)
        graph=observation_graph(obs,valid,frame,model.token_processor.meta['pos_scale'])
        zero_graph=observation_graph(reference,valid,frame,model.token_processor.meta['pos_scale'])
        zero1=infer(model,zero_graph,seed);zero2=infer(model,zero_graph,seed)
        np.testing.assert_array_equal(zero1['position'],zero2['position'])
        result=infer(model,graph,seed)
        # Complete future removal preserves the exact same forecast.
        truncated=observation_graph(obs[:,:19],valid[:,:19],frame,model.token_processor.meta['pos_scale'])
        repeated=infer(model,truncated,seed)
        np.testing.assert_array_equal(result['position'],repeated['position'])
        np.savez_compressed(out/f'{name}.npz',observed=obs,valid=valid,reference=reference,current_frame=frame,
            prediction=result['position'],heading=result['heading'],cv=constant_velocity(obs,frame),medoid=medoid_index(result['position']))
        cases.append(dict(name=name,prediction_shape=list(result['position'].shape),future_deletion_exact=True,sigma_zero_forecast_reproducible=True))
    verify_lock(root)
    receipt=dict(stage='D1',passed=True,cases=cases,sigma_zero_exact=True,prefix_noise_fixed=True,protected_unchanged=True)
    (root/'docs/integrated_demo_D1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
