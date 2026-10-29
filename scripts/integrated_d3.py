"""Small hand-verifiable D3 rule acceptance; no narrative model inference."""
import json
from pathlib import Path
from src.integrated.data import verify_lock
from src.integrated.threat import threat_level,AnomalyLatch


def main():
    root=Path(__file__).resolve().parents[1];verify_lock(root)
    config=json.loads((root/'configs/integrated_demo/v1.json').read_text())
    zone=config['zone'];rules=config['threat'];point=[-450,300,110]
    e=dict(entry_samples=32,total_samples=32)
    normal=threat_level(False,9,point,point,point,e,'nighttime',zone,rules)
    anomaly=threat_level(True,12,point,point,point,e,'daytime',zone,rules)
    assert normal['score'] is None and normal['level']=='Normal traffic'
    assert anomaly['score']==80 and anomaly['rule_points']=={'zone':80}
    latch=AnomalyLatch(10);assert latch.update(0,11) and latch.update(10,9) and not latch.update(10.01,9)
    receipt=dict(stage='D3',passed=True,normal_inside_at_night=normal,anomalous_inside=anomaly,
        ten_second_latch=True,no_anomaly_modifier=True,unknown_explicit=True,explanation_templates=True)
    (root/'docs/integrated_demo_D3.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
