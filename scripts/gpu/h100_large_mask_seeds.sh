#!/usr/bin/env bash
# GPU-host chain: the two ModernBERT-large mask-trained contract seeds (1 and 2; block mask, 4,616 steps, the first run's
# recipe), retrained so their weights are kept for release (the earlier runs' weights were lost in transfer). Tags
# e30-s1r / e30-s2r; the analysis uses these runs, whose weights are released. arguments "3 4" (or SEEDS="3 4") add seeds 3 and 4 (tags
# e30-s3r / e30-s4r). REHEARSAL: EXTRA, TAGSFX.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
EXTRA=${EXTRA:-}; TAGSFX=${TAGSFX:-}; SFX=${TAGSFX:+-$TAGSFX}
log() { echo "[large-seeds] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4))" 2>/dev/null)"
}
for s in ${*:-${SEEDS:-1 2}}; do ft large-block-contractnli-e30-s${s}r$SFX --size large --mode block --max-seq-tokens 5120 --max-steps 4616 --seed $s --tag e30-s${s}r$SFX & done
wait
ls -la models/ | grep e30
log "chain done"
