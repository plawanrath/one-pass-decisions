#!/usr/bin/env bash
# Block arms at the full-attention arm's optimizer step count on the workstation (MPS), for the tasks where packing
# reduces the step count: guardrail, trace grading and contracts at the base size (ModernBERT and EuroBERT-210M).
# Every step skips when its artifact exists. MPS only: nothing else runs on the GPU meanwhile.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[steps] $(date '+%F %T') $*"; }
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
ft base-block-tracegrade-steps tracegrade --size base --mode block --tokens-per-batch 16384 --match-steps-of base-full-tracegrade --tag steps
ft base-block-guardrail-steps guardrail --size base --mode block --tokens-per-batch 8192 --match-steps-of base-full-guardrail --tag steps
ft base-block-contractnli-steps contractnli --size base --mode block --tokens-per-batch 16384 --max-seq-tokens 5120 --match-steps-of base-full-contractnli --tag steps
ft eurobert-base-block-contractnli-steps contractnli --size eurobert-base --mode block --tokens-per-batch 16384 --max-seq-tokens 5120 --micro-tokens 5120 --match-steps-of eurobert-base-full-contractnli --tag steps
log "equal-steps queue done"
