#!/usr/bin/env bash
# GPU-host chain (ADR-0006 follow-up): the engine with ModernBERT-large on contract review, the size whose accuracy
# matches the fine-tuned decoder baseline. Latency does not depend on the weights, so the pretrained encoder with an
# untrained head runs (--stock). The base engine is re-timed in the same session as the anchor to the earlier session
# that measured the decoder. GPU exclusive throughout.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[large] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
.venv/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda', ok, torch.version.cuda); sys.exit(0 if ok else 1)" || { log "CUDA unusable on this host (driver too old for the locked torch)"; exit 1; }
log "tests on CUDA (engine, fused fill, trimmed cache, varlen)"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_engine.py tests/test_fused_trim.py tests/test_varlen.py 2>&1 | tail -1
$PY scripts/dump_requests.py --task contractnli --cells 2048:17 4096:17 4096:1 4096:2 4096:5 4096:10 --n 80 --test-docs 0
for size in large base; do
  $PY scripts/latency_sweep.py --size $size --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 50 --cells 2048:17 4096:17 \
      --paths compile-one-seq varlen-one-seq varlen-batched naive-batch engine-cold-fill engine-hit --out-tag strength-fused 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
$PY scripts/latency_sweep.py --size large --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 30 --cells 4096:1 4096:2 4096:5 4096:10 \
    --paths compile-one-seq varlen-batched engine-cold-fill engine-hit --out-tag strength-fused 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
$PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size large --stock --device cuda --dtype bf16 --tag fused-large 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
log "chain done"
# appended: the ModernBERT-base contract block arm at equal steps (the workstation's run measured ~10 h with gradient
# accumulation on MPS; here ~25 min), then the analysis files the workstation needs
log "base-block-contractnli-steps on the GPU"
[ -f results/finetune/base-block-contractnli-steps/summary.json ] || $PY scripts/finetune.py --task contractnli --size base --mode block --tokens-per-batch 16384 \
    --max-seq-tokens 5120 --match-steps-of base-full-contractnli --tag steps 2>&1 | grep --line-buffered -E "test|Error|Traceback" | tail -3
log "appended run done"
