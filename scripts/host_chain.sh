#!/usr/bin/env bash
# Second-host chain (rented x86 or GPU box). Run from a clone with the fine-tuned weights copied into models/ and the
# pools regenerated (scripts/lock_pools.py) or copied. Usage:
#   bash scripts/host_chain.sh cpu 8        # x86: K sweep, per-path scores, certification inputs, at 8 and 1 threads
#   bash scripts/host_chain.sh cuda 8       # GPU: the same K sweep on the PyTorch paths with the CUDA device
set -uo pipefail
cd "$(dirname "$0")/.."
DEV=${1:-cpu}; THREADS=${2:-8}
PY=".venv/bin/python -u"
log() { echo "[host] $(date '+%F %T') $*"; }
for task in routing guardrail tracegrade; do
  [ -d "models/modernbert-base-block-$task" ] || { log "no weights for $task, skipping"; continue; }
  if [ "$DEV" = "cpu" ]; then
    for t in $THREADS 1; do
      log "characterize $task threads=$t"
      $PY scripts/latency_sweep.py --size base --task $task --threads $t --K 1 5 20 50 --S 512 2048 2>&1 | tee -a logs/host_characterize_${task}_t${t}.log
    done
    log "pathscores $task"
    $PY scripts/path_scores.py --size base --task $task --paths engine-hit ort-fp32 ort-int8 --threads $THREADS 2>&1 | tee -a logs/host_pathscores_${task}.log
  else
    log "characterize $task on $DEV"
    $PY scripts/latency_sweep.py --size base --task $task --threads $THREADS --device $DEV --K 1 5 20 50 --S 512 2048 \
        --paths one-seq naive-batch engine-cold engine-hit prefill 2>&1 | tee -a logs/host_characterize_${task}_${DEV}.log
  fi
done
log "host chain done"
