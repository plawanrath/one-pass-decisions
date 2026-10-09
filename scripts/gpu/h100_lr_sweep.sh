#!/usr/bin/env bash
# GPU-host chain for ADR-0008 E17: ModernBERT-base full attention on contracts at three learning rates beside the
# default 3e-5, five epochs, each keeping its best development-accuracy checkpoint; the three runs share the GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[lr] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

log "phase 0: tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests > logs/lr_sweep_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/lr_sweep_tests.log; exit 1; }
grep -E "passed|failed" logs/lr_sweep_tests.log | tail -1

log "phase 1: full-arm learning rates"
ft() {  # lr
  local lr=$1 tag="lr${1/-/}-bestacc" name="base-full-contractnli-lr${1/-/}-bestacc"
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli --size base --mode full --epochs 5 --lr $lr \
      --select-points 10 --select-metric accuracy --tag $tag 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));print('dev', round(max(p['dev_accuracy'] for p in d['train']['selection']),4), 'test', round(d['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
pids=()
for lr in 1e-5 2e-5 5e-5; do ft $lr & pids+=($!); done
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  [ $alive = 1 ] || break
  sleep 300
  for lr in 1e5 2e5 5e5; do tail -n 1 logs/ft_base-full-contractnli-lr$lr-bestacc.log 2>/dev/null | grep "\[train\]\|\[select\]" | sed "s/^/[lr] lr$lr /"; done
done
wait
log "chain done"
