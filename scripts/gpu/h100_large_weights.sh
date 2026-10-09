#!/usr/bin/env bash
# GPU-host chain: the six large-encoder fine-tunes behind the paper's large rows, with the recorded recipes (routing
# 4 epochs, guardrail 1, trace grading 10 with tail truncation; block and full each), kept on the host for download.
# Every step skips when its artifact exists. Usage on the host: bash scripts/gpu/h100_large_weights.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
ft() {  # ft <task> <mode> <epochs> <tokens-per-batch>
  local task=$1 mode=$2 ep=$3 tpb=$4 name="large-$2-$1"
  if [ ! -f "results/finetune/$name/summary.json" ]; then
    log "finetune $name ($ep epochs, $tpb tokens per batch)"
    $PY scripts/finetune.py --size large --mode $mode --task $task --epochs $ep --tokens-per-batch $tpb 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_$name.log
  fi
}
ft tracegrade block 10 16384
ft tracegrade full 10 16384
ft routing block 4 16384
ft routing full 4 16384
ft guardrail block 1 8192
ft guardrail full 1 16384
ls -la models
log "h100 chain done"
