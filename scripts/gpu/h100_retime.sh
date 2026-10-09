#!/usr/bin/env bash
# GPU-host chain: re-time the main-text GPU cells in one session after the host-overhead reductions (tokenizer memo and
# batched state tokenization, cascade indices built once per batch, one host copy of the probabilities, cache fill sliced
# straight to the local tail). Phase 0: the identity tests on CUDA. Phase 1: latency of the strongest baseline and the
# cascade engine (Table gpu_strength rows, the K sweep of Figure gpu_k, the windowed row with the split engine). Phase 2:
# model FLOP/s per cold request (Table kernels). Phase 3: throughput, 1 to 64 contracts, engine and baseline. Phase 4: the
# 64-contract profiles. Pretrained encoders (latency does not depend on the weights); GPU exclusive; about 1.5 to 2 h.
# Every output carries the "retimesession" tag, so the earlier sessions' files stay as they are.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[retime] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
TAG=retimesession
log "phase 0: identity and kernel tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_host_overhead.py tests/test_flex.py tests/test_fused_trim.py > logs/retime_tests.log 2>&1 || { log "tests FAILED"; tail -40 logs/retime_tests.log; exit 1; }
grep -E "passed|failed" logs/retime_tests.log | tail -1
if [ ! -f data/pools/cuad/test.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified"
fi
lat() { $PY scripts/latency_sweep.py --device cuda --dtype bf16 --no-cudnn-sdp --stock "$@" 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"; }
log "phase 1: latency (baseline and cascade engine in the same session)"
for size in base large; do
  lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths varlen-batched --out-tag strength-$TAG
  lat --size $size --task contractnli --n 50 --cells 2048:17 4096:17 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
  lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths varlen-batched --out-tag strength-$TAG
  lat --size $size --task contractnli --n 30 --cells 4096:1 4096:2 4096:5 4096:10 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
done
lat --size base --task cuad --n 30 --cells 8192:41 --paths varlen-batched --out-tag strength-$TAG
lat --size base --task cuad --n 30 --cells 8192:41 --paths engine-cold engine-cold-fill engine-hit --engine-kernel cascade --out-tag strength-$TAG
lat --size base --task cuad --n 20 --cells 32768:41 --paths varlen-batched engine-windowed-cold engine-windowed-hit --engine-kernel split --out-tag strength-$TAG-windowed
log "phase 2: model FLOP/s per cold request"
for size in base large; do
  $PY scripts/attention_kernels.py --size $size --task contractnli --cell 4096:17 --device cuda --n 20 --tag=-cascade-$TAG 2>&1 | grep --line-buffered -E "\[kernels\]|Error|Traceback"
done
log "phase 3: throughput, 1 to 64 contracts (engine and unpadded baseline)"
for size in base large; do
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size $size --stock --device cuda --dtype bf16 --engine-kernel cascade --tag $TAG-$size-cascade 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
done
$PY scripts/engine_throughput.py --task cuad --cell 8192:41 --N 1 4 16 32 --stock --device cuda --dtype bf16 --engine-kernel cascade --no-baseline --tag $TAG-cascade 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
log "phase 4: 64-contract profiles"
for size in base large; do
  $PY scripts/engine_profile.py --size $size --task contractnli --cell 4096:17 --N 64 --kernel cascade --batch-tokens 65536 131072 --tag $TAG 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
done
log "chain done"
