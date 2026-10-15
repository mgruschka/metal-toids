#!/bin/bash
# MIRIAM-style TI for one config.
#   sbatch slurm/run_ti.sbatch configs/ni_co_imidazole4.yaml          -> runs/ni_co_imidazole4 (sequential windows)
#   sbatch --array=0-11 slurm/run_ti.sbatch configs/ni_co_imidazole4.yaml
#       -> one window per task (all start from the shared NPT state);
#          afterwards:  python -m lig1264.ti <cfg> --outdir <dir> --analyze
# The prep stages (build/min/heat/NPT) must exist before an array is
# submitted: run once with --stop-after npt, e.g.
#   sbatch slurm/run_ti.sbatch configs/ni_co_imidazole4.yaml runs/ni_co_imidazole4 npt

#SBATCH --job-name=ti1264
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python

CONFIG=${1:?config yaml}
OUTDIR=${2:-runs/$(basename "$CONFIG" .yaml)}
STOP=${3:-windows}

nvidia-smi || { echo "no GPU visible"; exit 1; }
mkdir -p "$OUTDIR"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    $PY -m lig1264.ti "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --window "$SLURM_ARRAY_TASK_ID"
else
    $PY -m lig1264.ti "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --stop-after "$STOP"
fi
echo "done: $OUTDIR"
