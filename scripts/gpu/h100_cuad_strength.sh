#!/usr/bin/env bash
# GPU-host chain for ADR-0005 (only the cells the workstation cannot run): the GPU baseline at strength (E5) and the
# CUAD fine-tunes (E2-E4). Phase 0 verifies the code on CUDA (the tests, including the FlashAttention varlen kernel)
# and rebuilds the CUAD pools from the raw corpus, checked against the tracked manifest. Phase 1 times the GPU alone:
# bf16, cuDNN attention off; one-seq eager and compiled, unpadded varlen one-seq and batched, padded batching, the
# engine cold and resident, on the contract shapes (ContractNLI S 2,048/4,096 K 17; CUAD S 8,192 K 41), the windowed
# engine at S 32,768 K 41, and the K sweep on routing. Phase 2 trains CUAD in two GPU queues: (A) full attention,
# then block and windowed at the same optimizer step count; (B) block at equal epochs, then the truncated-input
# scoring of the full model (needs A's first run); (C) EuroBERT-610m on contracts. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[cuad] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

log "phase 0: tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests > logs/cuad_strength_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/cuad_strength_tests.log; exit 1; }
grep -E "passed|failed" logs/cuad_strength_tests.log | tail -1
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

log "phase 1: GPU baseline at strength (GPU exclusive)"
STRONG="one-seq compile-one-seq varlen-one-seq varlen-batched naive-batch engine-cold engine-hit"
lat() {  # task size cells... ; extra args via LAT_EXTRA
  local task=$1 size=$2; shift 2
  $PY scripts/latency_sweep.py --size $size --task $task --device cuda --dtype bf16 --no-cudnn-sdp --n 50 --cells "$@" \
      --paths $STRONG ${LAT_EXTRA:-} --out-tag strength 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"
}
lat contractnli base 2048:17 4096:17
lat routing base 512:1 512:2 512:5 512:10 512:20 512:50 2048:1 2048:2 2048:5 2048:10 2048:20 2048:50
# CUAD's latency does not depend on the fine-tuned weights: the pretrained encoder with an untrained head (--stock)
LAT_EXTRA="--stock --n-at-S 8192:30" lat cuad base 8192:41
# beyond one window only the windowed engine reads the whole state; the baselines read their 8,190-token maximum
$PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --n 20 --cells 32768:41 --stock \
    --paths varlen-batched engine-windowed-cold engine-windowed-hit --out-tag strength-windowed 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"

log "phase 2: CUAD fine-tunes"
ft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task cuad "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
(
  ft base-full-cuad --size base --mode full --epochs 3 --tokens-per-batch 65536 --max-seqs 64 --micro-tokens 16384
  ft base-block-cuad-steps --size base --mode block --max-questions-per-seq 41 --max-seq-tokens 12288 --max-seqs 1 --match-steps-of base-full-cuad --tag steps
  ft base-windowed-cuad-steps --size base --mode windowed --max-questions-per-seq 41 --max-seqs 1 --match-steps-of base-full-cuad --tag steps --other-modes block
  log "queue A done"
) > logs/cuad_queue_a.log 2>&1 &
QA=$!
(
  ft base-block-cuad-e3 --size base --mode block --max-questions-per-seq 41 --max-seq-tokens 12288 --max-seqs 1 --epochs 3 --tag e3
  while [ ! -f results/finetune/base-full-cuad/summary.json ]; do sleep 60; done
  $PY scripts/trunc_eval.py --size base --task cuad --tokens 512 1024 2048 4096 --device cuda 2>&1 | grep --line-buffered "acc="
  log "queue B done"
) > logs/cuad_queue_b.log 2>&1 &
QB=$!
# queue C: EuroBERT-610m on contracts (the workstation measured about 70 GB for one 4,096-token sequence): four
# epochs both arms, and the block arm at the full arm's optimizer step count, as for the ModernBERT large model
cft() {  # name args...
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli --size eurobert-large "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name"
}
(
  cft eurobert-large-full-contractnli --mode full --epochs 4 --micro-tokens 8192
  cft eurobert-large-block-contractnli --mode block --epochs 4 --micro-tokens 6144
  cft eurobert-large-block-contractnli-steps --mode block --match-steps-of eurobert-large-full-contractnli --tag steps --micro-tokens 6144
  log "queue C done"
) > logs/cuad_queue_c.log 2>&1 &
QC=$!
wait $QA $QB $QC
cat logs/cuad_queue_a.log logs/cuad_queue_b.log logs/cuad_queue_c.log
log "cuad chain done"

# phase 3 (appended after phase 1 was found to re-tokenize the state per question on the baseline paths): the GPU
# baseline at strength again, every baseline tokenizing the state once per request, GPU exclusive after training
log "phase 3: GPU baseline at strength, state tokenized once per request"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_varlen.py 2>&1 | tail -1
LAT3() { local task=$1 size=$2; shift 2
  $PY scripts/latency_sweep.py --size $size --task $task --device cuda --dtype bf16 --no-cudnn-sdp --n 50 --cells "$@" \
      --paths $STRONG ${LAT_EXTRA:-} --out-tag strength-tok1 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"; }
LAT_EXTRA="" LAT3 contractnli base 2048:17 4096:17
LAT_EXTRA="--stock --n-at-S 8192:30" LAT3 cuad base 8192:41
LAT_EXTRA="" LAT3 routing base 512:1 512:2 512:5 512:10 512:20 512:50 2048:1 2048:2 2048:5 2048:10 2048:20 2048:50
$PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --n 20 --cells 32768:41 --stock \
    --paths varlen-batched engine-windowed-cold engine-windowed-hit --out-tag strength-tok1-windowed 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"
log "phase 3 done"
