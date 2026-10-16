#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sygard-mpl}"
exec /local/scratch/a/chen5262/anaconda3/envs/catk/bin/python "$@"
