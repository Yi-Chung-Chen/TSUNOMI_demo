"""Exact port of the supplied read-only flow-reference rule; not map alignment."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def scenario_flow(data):
    east=west=0
    for a in range(data['valid'].shape[0]):
        valid=data['valid'][a];p=data['pos'][a][valid];ground=data['on_ground'][a][valid].astype(bool)
        if len(p)<7:continue
        delta=p[6:,:2]-p[:-6,:2];speed=np.linalg.norm(delta,axis=1)/2
        rolls=ground[6:]&ground[:-6]&(speed>30)
        if rolls.sum()<3:continue
        hx=np.median(delta[rolls,0]/np.maximum(speed[rolls]*2,1e-6))
        if abs(hx)>.9:east+=int(hx>0);west+=int(hx<0)
    return dict(flow='east' if east>west else 'west' if west>east else 'none/tie',east_tracks=east,west_tracks=west)


def all_flows(processed):
    result={};counts=defaultdict(Counter)
    for path in sorted(Path(processed).glob('scenario_*_aircraft.npz')):
        with np.load(path) as a:data={key:a[key] for key in ['pos','valid','on_ground','date']}
        row=dict(date=str(data['date']),**scenario_flow(data));result[path.stem]=row;counts[row['date']][row['flow']]+=1
    return result,{day:dict(count) for day,count in counts.items()}
