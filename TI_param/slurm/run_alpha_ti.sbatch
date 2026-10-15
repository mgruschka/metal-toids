#!/bin/bash
# Bound-state alpha-TI (TI_param/run_TI.py) for one config.  Submit from TI_param/.
#   sbatch slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti.yaml
#       -> runs/co_imid1_alpha_ti, prep + all windows sequentially + analysis
#   two-step array (1-GPU shards schedule much faster than multi-GPU jobs):
#   sbatch slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti.yaml runs/co_imid1_alpha_ti npt
#   sbatch --array=0-10 slurm/run_alpha_ti.sbatch configs/co_imid1_alpha_ti.yaml
#       -> one window per task, each from npt.xml; afterwards
#          python run_TI.py <cfg> --outdir <dir> --analyze
# Requeue-safe: --resume skips finished stages / windows.
# Arguments after the 3rd go straight to run_TI.py (e.g. --seed-from / --start-state
# / --relax-ns for the hysteresis ladder; README "Known gaps" 3).

#SBATCH --job-name=alphati
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=andrewferguson-gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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
    $PY run_TI.py "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --window "$SLURM_ARRAY_TASK_ID" "${@:4}"
else
    $PY run_TI.py "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume \
        --stop-after "$STOP" "${@:4}"
fi
echo "done: $OUTDIR"
