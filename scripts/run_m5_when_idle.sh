#!/usr/bin/env bash
# Must be launched outside sandbox. Physical GPU visibility is fixed by catk.sh.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
on_exit() {
  task_exit_code=$?
  if (( task_exit_code != 0 )); then
    printf '\n## M5 runner ERROR — %s\n\nExit code: %s. See outputs/m5_execution/runner.log, smoke.log and bc.log. M5 is not complete; M6/M7 must wait.\n' "$(date -Is)" "$task_exit_code" >> QUESTIONS.md
    printf '\nM5 runner stopped with ERROR (exit %s); see QUESTIONS.md and outputs/m5_execution/.\n' "$task_exit_code" >> PROGRESS.md
  fi
}
trap on_exit EXIT
export CUDA_VISIBLE_DEVICES=1,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
mkdir -p outputs/m5_execution
exec 9>outputs/m5_execution/run.lock
if ! flock -n 9; then
  echo 'Another M5 execution is already waiting or running.' >&2
  exit 1
fi
scripts/catk.sh -u -m scripts.wait_for_idle_gpu --output outputs/m5_execution/gpu.json
logical_gpu="$(scripts/catk.sh -c 'import json; print(json.load(open("outputs/m5_execution/gpu.json"))["logical"])')"
scripts/catk.sh -m scripts.check_gpu_mapping outputs/m5_execution/gpu.json > outputs/m5_execution/mapping_smoke.json
echo "Starting M5 on logical GPU $logical_gpu (CUDA_VISIBLE_DEVICES=1,3)"
scripts/catk.sh -u -m src.run experiment=air_pre_bc "trainer.devices=[$logical_gpu]" trainer.max_epochs=1 trainer.limit_train_batches=16 trainer.limit_val_batches=4 data.train_batch_size=32 data.val_batch_size=16 hydra.run.dir=outputs/m5_smoke > outputs/m5_execution/smoke.log 2>&1
test -f outputs/m5_smoke/checkpoints/last.ckpt
# Same process owns the selected GPU after smoke; other jobs are never stopped.
# Recheck before full training so a new competing process is not silently shared.
scripts/catk.sh -u -m scripts.wait_for_idle_gpu --output outputs/m5_execution/gpu_full.json
logical_gpu="$(scripts/catk.sh -c 'import json; print(json.load(open("outputs/m5_execution/gpu_full.json"))["logical"])')"
scripts/catk.sh -m scripts.check_gpu_mapping outputs/m5_execution/gpu_full.json > outputs/m5_execution/mapping_bc.json
scripts/catk.sh -u -m src.run experiment=air_pre_bc "trainer.devices=[$logical_gpu]" data.train_batch_size=32 data.val_batch_size=16 hydra.run.dir=outputs/air_pre_bc > outputs/m5_execution/bc.log 2>&1
test -f outputs/air_pre_bc/checkpoints/last.ckpt
scripts/catk.sh -m scripts.verify_m5
# A milestone commit is created only after verify_m5 passes all acceptance gates.
git add src/common/attention_layer.py src/common/fourier_embedding.py src/common/mlp_layer.py src/common/weight_init.py src/metrics/cross_entropy.py src/model/agent_decoder.py src/model/air_smart.py src/model/decoder.py configs/callbacks/default.yaml configs/callbacks/model_checkpoint.yaml configs/model/air_smart.yaml configs/logger/wandb.yaml configs/trainer/default.yaml scripts/catk.sh scripts/check_overfit.py scripts/check_gpu_mapping.py scripts/wait_for_idle_gpu.py scripts/run_m5_when_idle.sh scripts/verify_m5.py tests/test_model.py tests/test_gpu_guard.py tests/test_m5_acceptance.py PROGRESS.md QUESTIONS.md README.md
scripts/catk.sh scripts/check_tracked.py
git diff --cached --check
git commit -m 'Train map-free 3D aircraft SMART behavior cloning model'
git ls-files src | wc -l
scripts/catk.sh scripts/check_tracked.py
