#!/bin/bash
# Umbrella-sampling PMF for one config.  Three submissions:
#
#   1. prep + SMD + window seeding (one GPU, ~6 h):
#        sbatch slurm/run_pmf.sbatch configs/pmf/ni_imid4_a275.yaml runs/pmf/a275 seed
#   2. the umbrella windows, one per array task:
#        sbatch --array=0-119%20 slurm/run_pmf.sbatch configs/pmf/ni_imid4_a275.yaml runs/pmf/a275
#   3. WHAM (no GPU needed):
#        python -m lig1264.wham runs/pmf/a275
#
# Step 2 requires windows.json from step 1.  Every stage is --resume-safe, so
# a requeued or timed-out task picks up where it stopped.

#SBATCH --job-name=pmf1264
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python

CONFIG=${1:?config yaml}
OUTDIR=${2:-runs/pmf/$(basename "$CONFIG" .yaml)}
STOP=${3:-windows}

nvidia-smi || { echo "no GPU visible"; exit 1; }
mkdir -p "$OUTDIR"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    $PY -m lig1264.pmf "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --window "$SLURM_ARRAY_TASK_ID"
else
    $PY -m lig1264.pmf "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --stop-after "$STOP"
fi
echo "done: $OUTDIR"
