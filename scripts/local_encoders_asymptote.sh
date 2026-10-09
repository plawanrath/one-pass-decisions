#!/usr/bin/env bash
# Workstation chain for ADR-0005 (everything the workstation can run): the chunked asymptote on the CPU first, alone
# (CPU timing never runs beside another job), then the second encoder family on MPS: EuroBERT-210m on routing (four
# epochs) and contracts (five), routing seed replicates, and the mask-price comparisons. EuroBERT-610m and CUAD go to the GPU host:
# measured on MPS, a single 4,096-token 610m sequence needs about 70 GB and CUAD's 8k sequences exceed 80 GB. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[encoders] $(date '+%F %T') $*" | tee -a logs/local_encoders_asymptote.log; }
mkdir -p logs

log "chunked asymptote (CPU, 8 threads, nothing else running)"
[ -f results/chunked_asymptote/arm64-darwin-base-t8/summary.json ] || \
  $PY scripts/chunked_asymptote.py --threads 8 --n 30 --lengths 512 1024 2048 3072 4096 6144 8000 2>&1 | grep --line-buffered -E "^T=|Error|Traceback" | tee -a logs/local_encoders_asymptote.log

ft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
log "EuroBERT-210m (MPS)"
ft eurobert-base-full-routing --size eurobert-base --mode full --task routing --epochs 4
ft eurobert-base-block-routing --size eurobert-base --mode block --task routing --epochs 4
$PY scripts/mask_price.py --size eurobert-base --tasks routing 2>&1 | tee -a logs/local_encoders_asymptote.log
ft eurobert-base-full-contractnli --size eurobert-base --mode full --task contractnli --epochs 5 --micro-tokens 4096
ft eurobert-base-block-contractnli --size eurobert-base --mode block --task contractnli --epochs 5 --micro-tokens 5120
$PY scripts/mask_price.py --size eurobert-base --tasks contractnli 2>&1 | tee -a logs/local_encoders_asymptote.log
log "routing seed replicates, EuroBERT-210m"
for s in 1 2; do
  ft eurobert-base-full-routing-s$s --size eurobert-base --mode full --task routing --epochs 4 --seed $s --tag s$s
  ft eurobert-base-block-routing-s$s --size eurobert-base --mode block --task routing --epochs 4 --seed $s --tag s$s
  $PY scripts/mask_price.py --size eurobert-base --tasks routing --tag s$s 2>&1 | tee -a logs/local_encoders_asymptote.log
done
log "encoders and asymptote chain done"
