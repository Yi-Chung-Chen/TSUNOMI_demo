"""Execute frozen story configs ONCE; stop on failed B pre-entry warning gate."""
import argparse
import json
from pathlib import Path
import time
import torch
from src.integrated.data import verify_lock
from src.integrated.forecast import load_model
from src.integrated.scoring import FrozenStateScorer
from src.integrated.story import run_story,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='Resume only an interrupted execution with unchanged config/seeds')
    parser.add_argument('--config',default='configs/integrated_demo/v1.json')
    parser.add_argument('--output-dir',default='outputs/integrated_demo')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    lock=verify_lock(root);config=json.loads((root/args.config).read_text())
    output=root/args.output_dir;output.mkdir(parents=True,exist_ok=True);marker=output/'D4_attempt.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['status'] in ['passed','acceptance_failed']:raise SystemExit('D4 already evaluated; no automatic retuning or repeat allowed')
        if not args.resume:raise SystemExit('Interrupted execution exists; inspect logs then --resume with same locked inputs')
        assert old['lock']==lock
    write_json(marker,dict(status='running',lock=lock,attempt=1));started=time.monotonic()
    torch.set_num_threads(1);model=load_model(root,config['checkpoint']);scorer=FrozenStateScorer(root/config['density_dir'])
    cases=[run_story(root,config,scene,model,scorer,output,args.resume) for scene in config['scenarios']]
    a,b=[c['evaluation'] for c in cases]
    passed=bool(a['anomalous_low_frames']>0 and b['first_model_based_preentry_high_s'] is not None)
    receipt=dict(stage='D4',passed=passed,attempt=1,retuned=False,elapsed_s=time.monotonic()-started,
        A=a,B=b,decision='continue to D5' if passed else 'STOP: report failed gate; do not adjust scenario/zone/threshold/seed')
    verify_lock(root)
    if output.resolve()==(root/'outputs/integrated_demo').resolve():
        write_json(root/'docs/integrated_demo_D4.json',receipt)
    write_json(output/'D4_acceptance.json',receipt)
    write_json(marker,dict(status='passed' if passed else 'acceptance_failed',lock=lock,attempt=1))
    print(json.dumps(receipt,indent=2),flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
