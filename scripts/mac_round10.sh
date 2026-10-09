#!/usr/bin/env bash
# Workstation chain (MPS, one job at a time; never beside an MLX job): the third task (ECtHR, ADR-0010), the
# encoder-decoder baseline on contracts with a same-host latency ratio against the engine, then more seeds of the
# large full-attention arms on contracts. ModernBERT at 4,096 tokens runs one sequence per micro-batch on MPS (the
# local layers' sliding-window mask is materialized; a 16,384-token batch exceeds the workstation's memory).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
log() { echo "[mac] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
}
log "phase 1: ECtHR full attention (best dev accuracy; served-masked scores included)"
ft base-full-ecthr ecthr --size base --mode full --epochs 1 --lr 2e-5 --select-points 10 --select-metric accuracy --micro-tokens 4096
log "phase 2: ECtHR mask-trained at the full arm's optimizer steps"
ft base-block-ecthr-steps ecthr --size base --mode block --max-seq-tokens 5120 --match-steps-of base-full-ecthr --micro-tokens 5120 --tag steps
log "phase 3: encoder-decoder baseline (Flan-T5 base) on contracts"
name=encdec-t5-base-contractnli
[ -f results/encdec_finetune/$name/summary.json ] || $PY scripts/encdec_finetune.py --task contractnli --model t5-base --lr 1e-4 --epochs 3 --select-points 6 --device mps 2>&1 \
    | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/encdec_finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
log "phase 4: same-host latency on the workstation GPU (MPS): encoder-decoder against the engine at 4,096 tokens, 17 questions"
[ -d models/$name ] && $PY scripts/encdec_serve.py --model models/$name --task contractnli --cells 4096:17 --n 30 --warmup 5 --agreement-docs 10 --device mps 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
$PY scripts/latency_sweep.py --size base --task contractnli --device mps --cells 4096:17 --n 30 --warmup 5 --stock \
    --paths engine-cold engine-hit one-seq --out-tag mps-encdec 2>&1 | grep --line-buffered -E "S=|Error|Traceback"
log "phase 5: seeds 1 and 2 of the large full-attention arms on contracts (served-masked scores included)"
for s in 1 2; do
  ft large-full-contractnli-s$s contractnli --size large --mode full --epochs 4 --seed $s --micro-tokens 4096 --tag s$s
  ft eurobert-large-full-contractnli-lr1e5-s$s contractnli --size eurobert-large --mode full --epochs 4 --lr 1e-5 --seed $s --micro-tokens 8192 --tag lr1e5-s$s
done
log "chain done"
