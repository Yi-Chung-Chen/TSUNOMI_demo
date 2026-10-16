"""Wait for an unoccupied authorized GPU; never stop or share another job."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def available_gpus():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1,3':
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be exactly 1,3')
    def query(argument):
        return subprocess.check_output(['nvidia-smi', '-i', '1,3', argument, '--format=csv,noheader,nounits'], text=True)
    apps = list(csv.reader(query('--query-compute-apps=gpu_uuid,pid').splitlines()))
    occupied = {row[0].strip() for row in apps if row}
    rows = list(csv.reader(query('--query-gpu=index,uuid,utilization.gpu,memory.used').splitlines()))
    states = []
    for row in rows:
        index, uuid, utilization, memory = [v.strip() for v in row]
        states.append(dict(physical=int(index), logical=[1, 3].index(int(index)), uuid=uuid,
                           utilization=int(utilization), memory_mib=int(memory), occupied=uuid in occupied))
    free = [s for s in states if not s['occupied'] and s['utilization'] < 10 and s['memory_mib'] < 1024]
    return states, free


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--interval', type=int, default=30)
    p.add_argument('--output', type=Path, default=Path('outputs/m5_execution/gpu.json'))
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    previous = None
    while True:
        states, free = available_gpus()
        print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), states=states)), flush=True)
        chosen = free[0] if free else None
        if chosen and previous == chosen['physical']:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(chosen, indent=2) + '\n')
            return
        previous = chosen['physical'] if chosen else None
        if args.once:
            raise SystemExit(0 if free else 2)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
