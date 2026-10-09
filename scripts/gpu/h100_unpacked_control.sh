#!/usr/bin/env bash
# GPU-host chain: is the mask-trained arm's ranking cost a property of packing? ModernBERT base on contracts, block mask
# with one question per sequence, the retuned full arm's exact recipe (5 epochs, lr 2e-5, best dev accuracy over 10
# points), seeds 20260927, 1, 2, all three at once (base, 5k tokens). REHEARSAL: EXTRA, TAGSFX.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; TAGSFX=${TAGSFX:-}; SFX=${TAGSFX:+-$TAGSFX}
log() { echo "[unpacked] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4), 'steps', d['train']['steps'])" 2>/dev/null)"
}
R="--size base --mode block --max-questions-per-seq 1 --max-seq-tokens 5120 --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy"
ft base-block-contractnli-unpacked$SFX $R --tag unpacked$SFX &
for s in 1 2; do ft base-block-contractnli-unpacked-s$s$SFX $R --seed $s --tag unpacked-s$s$SFX & done
wait
log "chain done"
