#!/usr/bin/env bash
# GPU-host chain: the served-windowed result as a curve. The one-window full-attention clause fine-tunes (ModernBERT base
# seeds 0/1, EuroBERT-610M seeds 0/1; uploaded) served with no training under 2 and 8 windows of 8,192 tokens and under
# 4 windows of 4,096 (the 4 x 8,192 point exists), test pool only. The GPU runs nothing else. REHEARSAL: positional args
# "rehearse" (tiny windows, six records; the provider runs --tasks under timeout, so no VAR= prefixes).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
W=8192; W4=4096; EV=""; SFX=""
if [ "${1:-}" = rehearse ]; then W=256; W4=128; EV="--max-eval 6"; SFX="-rh"; fi
log() { echo "[sweep] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
if [ ! -f data/pools/cuad/train.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified ($got)"
fi
[ -f results/finetune/base-windowed-cuad-steps/summary.json ] || { log "missing results/finetune/base-windowed-cuad-steps/summary.json (upload it)"; exit 1; }
for m in models/_from_h100_cuad/modernbert-base-full-cuad models/_from_h100_seed/modernbert-base-full-cuad-s1 \
         models/_from_h100_eurobert_cuad/eurobert-large-full-cuad models/_from_h100_eurobert_cuad/eurobert-large-full-cuad-s1; do
  [ -d $m ] || { log "missing $m"; continue; }
  n=$(basename $m); n=${n#modernbert-}
  for cfg in "2 $W" "8 $W" "4 $W4"; do
    set -- $cfg; k=$1; w=$2
    out=$n-w${k}x$((w / 1024))k$SFX
    [ -f results/served_windowed/$out/summary.json ] || $PY scripts/served_windowed.py --model $m --name $out --splits test \
      --window-tokens $w --max-state-tokens $((k * w)) $EV 2>&1 | grep --line-buffered -E "\[served-windowed\]|Error|Traceback"
  done
done
log "chain done"
