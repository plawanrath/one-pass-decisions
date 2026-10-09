#!/usr/bin/env bash
# After the equal-steps queue (MPS) finishes: the mask-price comparisons at equal steps, then the System 2 model on the
# document tasks (MLX; never beside an MPS job), then the risk-controlled thresholds and the cascade on long documents.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[after] $(date '+%F %T') $*"; }
until grep -q "equal-steps queue done" logs/local_equal_steps.log 2>/dev/null; do sleep 60; done
log "equal-steps queue finished"
# the ModernBERT-base contract run of that queue ran out of MPS memory without gradient accumulation: rerun it with
# micro-batches (the same update; tests/test_cuad_windowed_train.py)
name=base-block-contractnli-steps
[ -f results/finetune/$name/summary.json ] || $PY scripts/finetune.py --task contractnli --size base --mode block --tokens-per-batch 16384 \
    --max-seq-tokens 5120 --micro-tokens 5120 --match-steps-of base-full-contractnli --tag steps 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
$PY scripts/mask_price.py --size base --tasks guardrail tracegrade contractnli --tag steps --full-tag "" 2>&1 | tail -4
$PY scripts/mask_price.py --size eurobert-base --tasks contractnli --tag steps --full-tag "" 2>&1 | tail -2
log "System 2 on contracts"
$PY scripts/system2_documents.py --task contractnli --splits test cal 2>&1 | grep --line-buffered -E "accuracy|Error|Traceback"
log "System 2 on clauses"
$PY scripts/system2_documents.py --task cuad --splits test cal 2>&1 | grep --line-buffered -E "accuracy|Error|Traceback"
$PY scripts/risk_long.py 2>&1 | grep -v skip
$PY scripts/cascade_long.py 2>&1 | grep -v skip
log "after-steps chain done"
