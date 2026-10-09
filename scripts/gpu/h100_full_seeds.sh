#!/usr/bin/env bash
# GPU-host chain: more seeds of the full-attention arm on the long-document tasks; every full run also scores its
# weights under the block mask (served-masked), so each seed covers two arms. Usage: h100_full_seeds.sh contracts|clauses
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
WHICH=${1:?contracts or clauses}
log() { echo "[seeds] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));print('test', round(d['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
if [ "$WHICH" = clauses ] && [ ! -f data/pools/cuad/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified"
fi
log "phase 1: full-arm seeds ($WHICH)"
if [ "$WHICH" = contracts ]; then
  for s in 1 2; do
    ft base-full-contractnli-lr2e5-bestacc-s$s contractnli --size base --mode full --epochs 5 --lr 2e-5 --select-points 10 --select-metric accuracy --seed $s --tag lr2e5-bestacc-s$s &
  done
else
  ft base-full-cuad-s2 cuad --size base --mode full --epochs 3 --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384 --seed 2 --tag s2 &
fi
wait
log "chain done"
