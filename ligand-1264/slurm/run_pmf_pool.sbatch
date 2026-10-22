#!/bin/bash
# Umbrella windows for one PMF, packed into the qos=gpu limits.
#
#   [MARGIN=s] [STACK=k] sbatch slurm/run_pmf_pool.sbatch <config> <outdir> <shard> [nshard=4] [gpus=1]
#
# Defaults are the measured best packing: 1 GPU, 4 CPUs, 36 h, 4 shards of
# 30 windows (~55 min each at 4 ns -> ~27 h, fits one allocation, so no
# self-resubmit is needed).  Why:
#
# Queue limits (qos=gpu): MaxSubmitJobsPU=12 (array tasks count
# individually), MaxTRESPU gres/gpu=16,node=4.  Multi-GPU jobs need a whole
# free node and wait ~13-26 h (median, Sep 2026) vs 0.5 h for 1 GPU, so
# shards are 1-GPU.  For more GPUs per shard pass --gres=gpu:N
# --cpus-per-task=4N and gpus=N.
#
# STACK=k runs k windows concurrently per GPU.  Leave it at 1:
# slurm/bench_stack.sbatch (job 59717608, RTX 6000, 12k atoms, kappa 100)
# measured an aggregate 160 / 148 / 152 / 152 ns/day at k = 1 / 2 / 3 / 4 --
# the GPU is already saturated by one window (the ~1.4 busy cores are the
# CUDA driver thread + PLUMED, not idle-GPU host sync).  Kept for larger
# GPUs or smaller systems; re-benchmark before using it.
#
# MARGIN (s): stop starting new windows this long before the wall limit; set
# it to just over one window's wall time (~55-60 min on RTX 6000 / V100).

# Stream (shard, gpu g, slot k) runs the windows congruent to
# ((shard*gpus + g)*STACK + k) mod (nshard*gpus*STACK).  The split is
# static, so streams never race for a window; every window is --resume-safe
# and marks itself done, so a killed job loses at most the windows in flight.
# A shard that runs out of wall time resubmits itself -- which needs a free
# submit slot: the running job still counts against MaxSubmitJobsPU when it
# calls sbatch.  If the resubmit is rejected the job exits non-zero (FAIL
# mail) and prints the command to rerun by hand.  Resource flags and any
# --exclude node list are carried over from the running job.
# A TIMEOUT (job killed mid-window, e.g. on a slow GPU) never reaches the
# resubmit: resubmit by hand; finished windows are skipped.
#
# <outdir>/ABORT (written by pmf.py's drift gate) stops every stream before
# its next window and suppresses the resubmit.

#SBATCH --job-name=pmfpool
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=gruschka@rcc.uchicago.edu

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/midway3/gruschka/metal-toid/bin/python

CONFIG=${1:?config yaml}
OUTDIR=${2:?outdir}
SHARD=${3:?shard index}
NSHARD=${4:-4}
GPUS=${5:-1}
STACK=${STACK:-1}                # concurrent windows per GPU (env, so it survives resubmit)
MARGIN=${MARGIN:-3900}           # never start a window we cannot finish: >= one window's wall time at this STACK

# windows.json is authoritative: pmf.py --add-windows can append rows past pmf.n_windows
if [ -f "$OUTDIR/windows.json" ]; then
  NWIN=$($PY -c "import json;print(len(json.load(open('$OUTDIR/windows.json'))['windows']))")
else
  NWIN=$($PY -c "from lig1264.config import load_config;print(load_config('$CONFIG').pmf.n_windows)")
fi
NSTREAM=$((NSHARD * GPUS * STACK))
END=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep '^EndTime=' | cut -d= -f2)
DEADLINE=$(( $(date -d "$END" +%s) - MARGIN ))
echo "shard $SHARD/$NSHARD, $GPUS GPUs x $STACK stacked, $NWIN windows, stop starting new work at $(date -d @$DEADLINE)"
nvidia-smi -L || { echo "no GPU visible"; exit 1; }

for g in $(seq 0 $((GPUS - 1))); do
  for k in $(seq 0 $((STACK - 1))); do
  (
    tag="gpu$g.$k"
    stream=$(( (SHARD * GPUS + g) * STACK + k ))
    for w in $(seq "$stream" "$NSTREAM" $((NWIN - 1))); do
      marker=$(printf "%s/window_%03d_done" "$OUTDIR" "$w")
      [ -f "$marker" ] && { echo "[$tag] window $w already done"; continue; }
      [ -f "$OUTDIR/ABORT" ] && { echo "[$tag] ABORT present, stopping before window $w"; break; }
      if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[$tag] out of time before window $w"; break
      fi
      echo "[$tag] window $w start $(date)"
      CUDA_VISIBLE_DEVICES=$g $PY -m lig1264.pmf "$CONFIG" --outdir "$OUTDIR" \
          --platform CUDA --resume --window "$w" || echo "[$tag] window $w FAILED"
    done
  ) &
  done
done
wait

if [ -f "$OUTDIR/ABORT" ]; then
  echo "shard $SHARD: $OUTDIR/ABORT present -- not resubmitting:"
  cat "$OUTDIR/ABORT"
  exit 1
fi

# resubmit this shard if any of its windows are still missing
left=0
for w in $(seq 0 $((NWIN - 1))); do
  if [ $(( w % NSTREAM / (GPUS * STACK) )) -eq "$SHARD" ]; then
    [ -f "$(printf "%s/window_%03d_done" "$OUTDIR" "$w")" ] || left=$((left + 1))
  fi
done
echo "shard $SHARD: $left windows still missing"
if [ "$left" -gt 0 ]; then
  # Carry this job's actual resources forward.  sbatch CLI flags do NOT
  # propagate to a nested sbatch -- only the environment does (STACK and
  # MARGIN ride along that way) -- so without these the resubmit would
  # silently fall back to the #SBATCH header above.
  LIMIT=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep '^TimeLimit=' | cut -d= -f2)
  # keep any node exclusion (e.g. a node with a degraded GPU); Slurm has no
  # SBATCH_EXCLUDE env var, so it must be passed explicitly like the rest
  EXC=$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep '^ExcNodeList=' | cut -d= -f2)
  EXCARG=(); [ -n "$EXC" ] && [ "$EXC" != "(null)" ] && EXCARG=(--exclude="$EXC")
  CMD="STACK=$STACK MARGIN=$MARGIN sbatch --gres=gpu:$GPUS --cpus-per-task=$SLURM_CPUS_PER_TASK --time=$LIMIT ${EXCARG[*]} slurm/run_pmf_pool.sbatch $CONFIG $OUTDIR $SHARD $NSHARD $GPUS"
  if sbatch --gres="gpu:$GPUS" --cpus-per-task="$SLURM_CPUS_PER_TASK" --time="$LIMIT" "${EXCARG[@]}" \
         slurm/run_pmf_pool.sbatch "$CONFIG" "$OUTDIR" "$SHARD" "$NSHARD" "$GPUS"; then
    echo "shard $SHARD resubmitted (gpu:$GPUS x $STACK, $LIMIT)"
  else
    echo "shard $SHARD: RESUBMIT REJECTED (submit limit?).  When a slot frees, run:"
    echo "  cd $SLURM_SUBMIT_DIR && $CMD"
    exit 1
  fi
else
  echo "shard $SHARD complete"
fi
