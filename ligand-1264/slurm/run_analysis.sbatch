#!/bin/bash
# CPU post-processing (WHAM / MBAR bootstraps) off the login nodes.
#   sbatch slurm/run_analysis.sbatch "lig1264.wham runs/pmf/x --r-bound 2.63" ["lig1264.mbar runs/pmf/x --target -3.36" ...]
# Each argument is one `python -m` command line, run in order; the job stops
# at the first failure (except exit 2 = an analysis gate refusing, which is
# reported and the next command still runs).

#SBATCH --job-name=ana1264
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=andrewferguson
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

for cmd in "$@"; do
    echo "=== $(date +%T) python -m $cmd"
    $PY -m $cmd
    rc=$?
    if [[ $rc -eq 2 ]]; then
        echo "=== exit 2 (gate refused) for: $cmd"
    elif [[ $rc -ne 0 ]]; then
        echo "=== FAILED ($rc): $cmd"; exit $rc
    fi
done
echo "=== $(date +%T) done"
