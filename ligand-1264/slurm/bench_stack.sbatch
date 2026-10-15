#!/bin/bash
# How many umbrella windows to stack per GPU (run_pmf_pool.sbatch STACK).
# Runs 1, 2, 3, 4 short windows concurrently on one GPU and reports the
# aggregate ns/day of each level.  Needs a seeded run to borrow the system
# and SMD frames from; windows.json is copied with kappa forced to the
# config's window_kappa, so the drift gate is exercised at production kappa.
#
#   sbatch slurm/bench_stack.sbatch <config> <seeded outdir>
#
# Uses a qos=gpu slot (same RTX 6000 / V100 hardware as the pool) for <45 min.

#SBATCH --job-name=benchstack
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python
CONFIG=${1:?config yaml}
SRC=${2:?seeded outdir to borrow system.xml / smd.dcd from}
B=runs/bench_stack/$SLURM_JOB_ID
mkdir -p "$B"
nvidia-smi -L

# system + SMD only; no window_* so every window starts fresh
for f in "$SRC"/*; do
  case $(basename "$f") in window_*|figures|pmf*|REPORT.md|ABORT) ;; *) cp -r "$f" "$B"/ ;; esac
done
$PY - "$CONFIG" "$B" <<'PYEOF'
import json, sys, yaml
from lig1264.config import load_config
from lig1264.pmf import kappa_kj_nm2
cfg_path, B = sys.argv[1:]
k = load_config(cfg_path).pmf.window_kappa
m = json.load(open(f"{B}/windows.json"))
m["kappa_kcal_per_A2"], m["kappa_kj_per_nm2"] = k, kappa_kj_nm2(k)
json.dump(m, open(f"{B}/windows.json", "w"), indent=2)
raw = yaml.safe_load(open(cfg_path))
raw["name"] = "bench_stack"
raw["pmf"].update(window_equil_ns=0.05, window_prod_ns=0.25)
yaml.safe_dump(raw, open(f"{B}/bench.yaml", "w"))
print(f"bench dir {B}: kappa {k} kcal/mol/A^2, 0.05 ns equil + 0.25 ns prod per window")
PYEOF

first=10
for n in 1 2 3 4; do
  ws=$(seq $first $((first + n - 1))); first=$((first + n + 2))
  t0=$(date +%s)
  for w in $ws; do
    $PY -m lig1264.pmf "$B/bench.yaml" --outdir "$B" --platform CUDA --resume --window "$w" \
        > "$B/bench_n${n}_w${w}.log" 2>&1 &
  done
  wait
  t1=$(date +%s)
  $PY - "$B" "$n" "$((t1 - t0))" $ws <<'PYEOF' || echo "STACK=$n: no speed (window failed; see $B/bench_n${n}_w*.log)"
import sys, csv
B, n, wall, *ws = sys.argv[1:]
sp = []
for w in ws:
    rows = list(csv.DictReader(open(f"{B}/window_{int(w):03d}.csv")))
    sp.append(float(rows[-1]["Speed (ns/day)"]))
print(f"STACK={n}: per-window {', '.join(f'{s:.0f}' for s in sp)} ns/day -> "
      f"aggregate {sum(sp):.0f} ns/day; wall {wall} s for {n} x 0.30 ns")
PYEOF
  grep -h "drift check" "$B"/bench_n${n}_w*.log || true
  # a drift abort here is a finding, not a reason to stop benchmarking
  if [ -f "$B/ABORT" ]; then echo "drift gate fired:"; cat "$B/ABORT"; mv "$B/ABORT" "$B/ABORT.n$n"; fi
done
echo "per-window logs in $B"
