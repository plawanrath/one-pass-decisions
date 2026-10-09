#!/usr/bin/env bash
# Second-host chain (rented x86 or GPU box). Run from a clone with the fine-tuned weights copied into models/ and the
# pools regenerated (scripts/lock_pools.py) or copied. Every sweep resumes cell by cell. Usage:
#   bash scripts/host_chain.sh cpu 8        # x86: full K sweeps at 8 threads for the three tasks (same grid and request
#                                           #      counts as the workstation), per-path scores, then a 1-thread grid
#   bash scripts/host_chain.sh cuda 8       # GPU: the same K sweep on the PyTorch paths with the CUDA device
# Environment: HOST_N requests per cell at the main thread count (default 100, as on the workstation); HOST_SKIP cells
# to leave out of the main grid (default none); the 1-thread grid runs K in {1, 5, 20} at S=512 and K in {1, 5} at
# S=2048 with HOST_N1 requests (default 50), because one sequence per question at 1 thread costs minutes per request.
# HOST_N2048 (default 50) is the request count for S=2048 cells at the main thread count (the S=512 cells use HOST_N).
# Path scores and thresholds of a second host go to results/<kind>/<size>-<task>-<host tag>.
set -uo pipefail
cd "$(dirname "$0")/.."
DEV=${1:-cpu}; THREADS=${2:-8}
N=${HOST_N:-100}; N1=${HOST_N1:-50}; N2048=${HOST_N2048:-50}; SKIP=${HOST_SKIP:-}
PY=".venv/bin/python -u"
log() { echo "[host] $(date '+%F %T') $*"; }
host=$($PY -c "from onepass.cli import host_tag; print(host_tag())")
sweep() {  # sweep <task> <threads> <n> <skip-list> <K list...>
  local task=$1 t=$2 n=$3 skip=$4; shift 4
  if [ ! -f "results/latency/$host-base-$task-t$t/summary.json" ]; then
    log "characterize $task threads=$t n=$n K=$* skip=${skip:-none}"
    $PY scripts/latency_sweep.py --size base --task $task --threads $t --n $n --n-at-S 2048:$N2048 --K "$@" --S 512 2048 ${skip:+--skip $skip} 2>&1 | tee -a logs/host_characterize_${task}_t${t}.log
  fi
}
SCORE_THREADS=$(( $(nproc) / 3 )); [ "$SCORE_THREADS" -gt 32 ] && SCORE_THREADS=32; [ "$SCORE_THREADS" -lt "$THREADS" ] && SCORE_THREADS=$THREADS
if [ "$DEV" = "cpu" ]; then
  for task in routing guardrail tracegrade; do
    [ -d "models/modernbert-base-block-$task" ] || { log "no weights for $task, skipping"; continue; }
    sweep $task $THREADS $N "$SKIP" 1 5 20 50
  done
  for task in routing guardrail tracegrade; do
    [ -d "models/modernbert-base-block-$task" ] || continue
    if [ ! -f "results/path_scores/base-$task-$host/meta.json" ]; then
      log "pathscores $task ($SCORE_THREADS threads; scoring is not a timing cell; results under base-$task-$host)"
      $PY scripts/path_scores.py --size base --task $task --paths engine-hit ort-fp32 ort-int8 --threads $SCORE_THREADS --out-suffix $host 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/host_pathscores_${task}.log
    fi
  done
  for task in routing guardrail tracegrade; do
    [ -d "models/modernbert-base-block-$task" ] || continue
    sweep $task 1 $N1 "S2048_K20 S2048_K50 S512_K50" 1 5 20 50
  done
else
  for task in routing guardrail tracegrade; do
    [ -d "models/modernbert-base-block-$task" ] || { log "no weights for $task, skipping"; continue; }
    if [ ! -f "results/latency/$host-$DEV-base-$task-t$THREADS/summary.json" ]; then
      log "characterize $task on $DEV"
      $PY scripts/latency_sweep.py --size base --task $task --threads $THREADS --device $DEV --K 1 5 20 50 --S 512 2048 \
          --paths one-seq naive-batch engine-cold engine-hit prefill 2>&1 | tee -a logs/host_characterize_${task}_${DEV}.log
    fi
  done
fi
log "host chain done"
