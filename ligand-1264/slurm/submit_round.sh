#!/bin/bash
# Submit every config in a round directory as a sequential TI job.
#   slurm/submit_round.sh configs/round1
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in "$1"/*.yaml; do
    name=$(basename "$cfg" .yaml)
    out=runs/$(basename "$1")/$name
    mkdir -p "$out"
    sbatch --job-name="ti_${name}" slurm/run_ti.sbatch "$cfg" "$out"
done
