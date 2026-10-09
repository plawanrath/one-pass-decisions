#!/usr/bin/env bash
# GPU-host chain: seeds 1 and 2 of the full-attention arm for ModernBERT-large and EuroBERT-610M on contracts; every full
# run also scores its weights under the block mask (served-masked), so each seed covers two arms.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[large-seeds] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));print('test', round(d['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
for s in 1 2; do
  ft large-full-contractnli-s$s --size large --mode full --epochs 4 --seed $s --tag s$s &
  ft eurobert-large-full-contractnli-lr1e5-s$s --size eurobert-large --mode full --epochs 4 --lr 1e-5 --micro-tokens 8192 --seed $s --tag lr1e5-s$s &
  wait
done
log "chain done"
