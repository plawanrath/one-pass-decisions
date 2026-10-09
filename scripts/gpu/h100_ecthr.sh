#!/usr/bin/env bash
# GPU-host chain: the third long-document task (LexGLUE ECtHR, 10 Convention articles as three-way questions per case).
# Locks the pools on the host and checks the manifest, then fine-tunes ModernBERT base with full attention (best dev
# accuracy, which also scores the weights under the block mask: served-masked) and under the block mask at the full
# arm's optimizer steps (mask-trained), and runs the long-document gate analyses.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[ecthr] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task ecthr "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4))" 2>/dev/null)"
}
if [ ! -f data/pools/ecthr/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/ecthr/POOL.json'))['manifest_sha256'])")
  mv data/pools/ecthr/POOL.json data/pools/ecthr/POOL.tracked.json
  $PY scripts/lock_pools.py --task ecthr || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/ecthr/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "ECtHR manifest mismatch"; exit 1; }
  rm data/pools/ecthr/POOL.tracked.json
  log "ECtHR pools verified"
fi
log "phase 1: full attention (best dev accuracy; served-masked scores included)"
ft base-full-ecthr --size base --mode full --epochs 1 --lr 2e-5 --select-points 10 --select-metric accuracy
log "phase 2: mask-trained at the full arm's optimizer steps"
ft base-block-ecthr-steps --size base --mode block --max-seq-tokens 5120 --match-steps-of base-full-ecthr --tag steps
log "chain done"
