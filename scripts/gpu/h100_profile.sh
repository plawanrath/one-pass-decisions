#!/usr/bin/env bash
# GPU-host chain: where the engine's time goes on the H100 with the split kernel. (1) One cold request under the profiler
# with model FLOP/s, base and large (scripts/attention_kernels.py). (2) 64 contracts at once at three step budgets, with
# the kernel-category profile (scripts/engine_profile.py). Pretrained encoders: latency does not depend on the weights.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[prof] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log "phase 0: kernel tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_flex.py tests/test_fused_trim.py > logs/profile_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/profile_tests.log; exit 1; }
grep -E "passed|failed" logs/profile_tests.log | tail -1
log "phase 1: one cold request, split kernel"
for size in base large; do
  $PY scripts/attention_kernels.py --size $size --task contractnli --cell 4096:17 --device cuda --n 20 --tag=-split 2>&1 | grep --line-buffered -E "\[kernels\]|Error|Traceback"
done
log "phase 2: 64 contracts at once"
$PY scripts/engine_profile.py --size base --task contractnli --cell 4096:17 --N 64 --batch-tokens 65536 131072 262144 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
$PY scripts/engine_profile.py --size large --task contractnli --cell 4096:17 --N 64 --batch-tokens 65536 131072 2>&1 | grep --line-buffered -E "\[profile\]|Error|Traceback"
log "chain done"
