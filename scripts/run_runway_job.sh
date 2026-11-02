#!/usr/bin/env bash
# Run OUTSIDE sandbox. Two independent processes, never DDP.
set -euo pipefail
cd "$(dirname "$0")/.."
job_name="$1"
physical_gpu="$2"
case "$job_name:$physical_gpu" in base20:1|anchor:3) ;; *) exit 2 ;; esac
out="outputs/runway_anchor/training/$job_name"
mkdir -p "$out"
if test -f "$out/completed.json"; then echo 'Already completed; refusing to retrain.'; exit 0; fi
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
export MPLCONFIGDIR=/tmp/sygard-mpl
python_bin=/local/scratch/a/chen5262/anaconda3/envs/catk/bin/python
"$python_bin" - "$physical_gpu" "$out" <<'PY'
import json,subprocess,sys,torch
from pathlib import Path
physical=int(sys.argv[1]);out=Path(sys.argv[2]);assert torch.cuda.is_available() and torch.cuda.device_count()==1
listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
row=next(r.split(', ') for r in listing.strip().splitlines() if int(r.split(',')[0])==physical)
assert int(row[2])<1024 and int(row[3])<10, f'GPU not idle: {row}'
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert row[1] not in processes, f'GPU {physical} has another compute process'
import ctypes
libs=list(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'));assert libs
runtime=ctypes.CDLL(str(libs[0]));function=runtime.cudaDeviceGetPCIBusId
function.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];function.restype=ctypes.c_int
bus=ctypes.create_string_buffer(64);assert function(bus,len(bus),0)==0
uuid=subprocess.check_output(['nvidia-smi','-i',bus.value.decode(),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert uuid==row[1],f'GPU mapping mismatch {uuid} vs {row[1]}'
record=dict(physical=physical,logical=0,uuid=row[1],runtime_pci_bus=bus.value.decode(),verified_uuid=uuid,
 device_name=torch.cuda.get_device_name(0),visible_devices=str(physical))
(out/'gpu.json').write_text(json.dumps(record,indent=2)+'\n')
PY
"$python_bin" -u -m src.run "experiment=air_pre_bc_$job_name" trainer.devices=1 "hydra.run.dir=$out" > "$out/train.log" 2>&1
"$python_bin" - "$out" <<'PY'
import csv,json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]);last=torch.load(p/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
assert last['epoch']==19 and last['global_step']==20*768
with (p/'csv/version_0/metrics.csv').open() as f:rows=list(csv.DictReader(f))
vals=[r for r in rows if r.get('val/loss')];assert len(vals)==20
best=[x for x in (p/'checkpoints').glob('epoch_*.ckpt')];assert len(best)==1
c=torch.load(best[0],map_location='cpu',weights_only=False)
record=dict(completed=True,epochs=20,global_step=last['global_step'],best_checkpoint=str(best[0]),best_epoch=c['epoch'],
 best_val_loss=float(c['callbacks'][next(k for k in c['callbacks'] if k.startswith('ModelCheckpoint'))]['best_model_score']),
 train_batch_size=32,val_batch_size=16,seed=817,selection_date='2026-10-03',device='cuda')
(p/'completed.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
PY
