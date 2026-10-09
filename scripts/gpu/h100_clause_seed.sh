#!/usr/bin/env bash
# GPU-host chain for ADR-0008 E18: a second seed (1) of the three clause arms with the recipes of the first seed. Phase 0
# verifies the code on CUDA and rebuilds the CUAD pools from the raw corpus against the tracked manifest. Phase 1 trains
# in two GPU queues: (A) full attention, three epochs; (B) block, then windowed, at the full arm's optimizer step count
# (the step count depends on the pool, not the seed, so B reads it from the first seed's run and starts at once).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[seed] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

log "phase 0: tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests > logs/clause_seed_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/clause_seed_tests.log; exit 1; }
grep -E "passed|failed" logs/clause_seed_tests.log | tail -1
log "phase 0: CUAD pools from the raw corpus, verified against the tracked manifest"
if [ ! -f data/pools/cuad/train.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified ($got)"
fi

log "phase 1: clause fine-tunes, seed 1"
ft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task cuad --seed 1 "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
(
  ft base-full-cuad-s1 --size base --mode full --epochs 3 --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384 --tag s1
  log "queue A done"
) > logs/seed_queue_a.log 2>&1 &
QA=$!
(
  ft base-block-cuad-steps-s1 --size base --mode block --max-questions-per-seq 41 --max-seq-tokens 12288 --max-seqs 1 --match-steps-of base-full-cuad --tag steps-s1
  ft base-windowed-cuad-steps-s1 --size base --mode windowed --max-questions-per-seq 41 --max-seqs 1 --match-steps-of base-full-cuad --tag steps-s1 --other-modes block
  log "queue B done"
) > logs/seed_queue_b.log 2>&1 &
QB=$!
while kill -0 $QA 2>/dev/null || kill -0 $QB 2>/dev/null; do
  sleep 300
  grep -h "\[seed\]" logs/seed_queue_a.log logs/seed_queue_b.log 2>/dev/null | sort | tail -4
done
wait $QA $QB
grep -h "\[seed\]" logs/seed_queue_a.log logs/seed_queue_b.log | sort
log "chain done"
