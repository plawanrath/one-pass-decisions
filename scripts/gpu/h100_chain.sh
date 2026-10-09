#!/usr/bin/env bash
# GPU-host chain (one rented H100). Every step skips when its artifact exists, so a re-run resumes. Order:
#   large-encoder fine-tunes (routing 4 epochs, guardrail 1, trace grading 2; full and block each)
#   -> guardrail base two-epoch pair -> mask-price rows -> CUDA K sweep on the PyTorch paths (three tasks).
# Usage on the host, from the synced clone:  bash scripts/gpu/h100_chain.sh 2>&1 | tee -a logs/h100_chain.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
ft() {  # ft <size> <task> <mode> <epochs> <tokens-per-batch> <tag>
  local size=$1 task=$2 mode=$3 ep=$4 tpb=$5 tag=$6
  local name="$size-$mode-$task${tag:+-$tag}"
  if [ ! -f "results/finetune/$name/summary.json" ]; then
    log "finetune $name ($ep epochs, $tpb tokens per batch)"
    $PY scripts/finetune.py --size $size --mode $mode --task $task --epochs $ep --tokens-per-batch $tpb ${tag:+--tag $tag} 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_$name.log
  fi
}
ft large routing block 4 16384 ""
ft large routing full 4 16384 ""
ft large guardrail block 1 8192 ""
ft large guardrail full 1 16384 ""
if [ -f data/pools/tracegrade/POOL.json ]; then
  ft large tracegrade block 2 16384 ""
  ft large tracegrade full 2 16384 ""
fi
ft base guardrail block 2 8192 e2
ft base guardrail full 2 16384 e2
log "mask-price rows"
$PY scripts/mask_price.py --size large --tasks routing guardrail tracegrade 2>&1 | tee -a logs/g1_mask_large.log
$PY scripts/mask_price.py --size base --tasks guardrail --tag e2 2>&1 | tee -a logs/g1_mask_e2.log
for task in routing guardrail tracegrade; do
  [ -d "models/modernbert-base-block-$task" ] || continue
  if [ ! -f "results/latency/x86_64-linux-cuda-base-$task-t8/summary.json" ]; then
    log "characterize $task on cuda"
    $PY scripts/latency_sweep.py --size base --task $task --threads 8 --device cuda --K 1 5 20 50 --S 512 2048 \
        --paths one-seq naive-batch engine-cold engine-hit prefill 2>&1 | tee -a logs/characterize_${task}_cuda.log
  fi
done
log "h100 chain done"
