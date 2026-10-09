#!/usr/bin/env bash
# GPU-host chain: cache capacity. Hit rate and contracts per second against the memory given to cached document state,
# under Zipf popularity over the 606 ContractNLI contracts (exponent fitted from a public production trace, and 0.3
# below and above it), for the engine (ModernBERT base and large) and the decoders on the tight cascade path
# (SmolLM2-360M, SmolLM2-1.7B, SmolLM3-3B). The GPU runs nothing else during this chain.
# REHEARSAL: DEVICE=cpu EXTRA="--tiny --docs 8 --n 20 --warm 5 --budgets-gb 0.0005 0.005 --max-state-tokens 256 --sim-grid 5" TAG=rh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
DEVICE=${DEVICE:-cuda}; EXTRA=${EXTRA:-}; TAG=${TAG:-}
log() { echo "[cache] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
[ -f data/pools/contractnli/train.jsonl ] || { log "missing data/pools/contractnli"; exit 1; }
$PY -m pytest -q tests/test_cache_capacity.py tests/test_decoder_cascade.py tests/test_engine.py 2>&1 | tail -2
ZIPF="$*"  # exponents as arguments (the provider runs tasks under timeout, so no VAR= prefixes); default: from the trace fit
log "exponents: ${ZIPF:-$(.venv/bin/python -c 'import sys;sys.path.insert(0,"scripts");from cache_capacity import default_exponents;print(*default_exponents())')}"
for m in engine:base engine:large decoder:smol360 decoder:smol1.7 decoder:smol3; do
  log "model $m"
  $PY scripts/cache_capacity.py --model $m --device $DEVICE ${TAG:+--tag $TAG} ${ZIPF:+--zipf $ZIPF} $EXTRA 2>&1 | grep --line-buffered -E "^\[cache\]|Error|Traceback"
done
log "chain done"
