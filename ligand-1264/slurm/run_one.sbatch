#!/bin/bash
# One ion/ligand system on a GPU node.
#   sbatch slurm/run_one.sbatch configs/zn_ala5.yaml            -> runs/zn_ala5
#   sbatch slurm/run_one.sbatch configs/zn_ala5.yaml runs/foo   -> runs/foo
# Requeue-safe: --resume picks up from the last completed stage.

#SBATCH --job-name=lig1264
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# metal-toid env resolves nvrtc/cufft from its own lib/ (see CLAUDE.md); no cuda module
PY=/scratch/midway3/gruschka/metal-toid/bin/python

CONFIG=${1:?config yaml}
OUTDIR=${2:-runs/$(basename "$CONFIG" .yaml)}

nvidia-smi || { echo "no GPU visible"; exit 1; }
mkdir -p "$OUTDIR"

$PY -m lig1264.run "$CONFIG" --outdir "$OUTDIR" --platform CUDA --resume
$PY -m lig1264.analysis "$OUTDIR" --phase prod
echo "done: $OUTDIR"
