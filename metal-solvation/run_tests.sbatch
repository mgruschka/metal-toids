#!/bin/bash

# email on start, end, and abortion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

#SBATCH --job-name=solvation_tests

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=andrewferguson-gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4    # SET NUM THREADS
#SBATCH --gres=gpu:1		# SET NUM GPUS
#SBATCH --time=24:00:00

# Load modules
module load python/anaconda-2022.05; source activate /scratch/midway3/gruschka/alchemist/; module load rcc cmake openmpi/4.1.2+gcc-10.2.0 lapack/3.10.0 gsl/2.7 plumed/2.8.0+annfunc+opes vmd fftw3/3.3.9 gromacs/2021.4+plumed2-2.8.0 cuda/12.0; export GMXLIB="/scratch/midway3/gruschka/toid_explore/forcefields/"

nvidia-smi || { echo "no GPU visible"; exit 1; }

# ---------------- ion definition ----------------
NAME=ION
ELEMENT=Zn
MASS=65.38
CHARGE=2
SIGMA=0.1950        # nm
EPSILON=0.0125      # kJ/mol

# ---------------- C4 values to scan ----------------
C4_VALUES=(0 50 100 150 200 250)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    C4=${C4_VALUES[$SLURM_ARRAY_TASK_ID]}
    OUTDIR=runs/${ELEMENT}_c4_${C4}
else
    C4=0
    OUTDIR=runs/${ELEMENT}_126
fi

C4_FLAG=""
if [[ "$C4" != "0" ]]; then
    C4_FLAG="--c4 $C4"
fi

mkdir -p "$OUTDIR"

# --resume is safe on a fresh directory and makes requeued jobs pick up
# where they stopped rather than starting over
srun python phase1.py \
    --mode both \
    --name  "$NAME" \
    --element "$ELEMENT" \
    --mass  "$MASS" \
    --charge "$CHARGE" \
    --sigma "$SIGMA" \
    --epsilon "$EPSILON" \
    $C4_FLAG \
    --box 3.0 \
    --equil-ns 0.5 \
    --iod-ns 20 \
    --ns-per-state 5 \
    --outdir "$OUTDIR" \
    --platform CUDA \
    --resume

echo "done: $OUTDIR"

