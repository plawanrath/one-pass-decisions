#!/usr/bin/env bash
# GPU-host chain: the fine-tuned ModernBERT-base contract model over the whole contracts test pool through masked SDPA
# (fp32 reference, bf16) and the cascade kernel (bf16): accuracy and decision agreement.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[acc] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "phase 1: accuracy through each kernel"
.venv/bin/python -u scripts/cascade_accuracy.py --model models/_from_h100_flex/modernbert-base-block-contractnli-bestacc 2>&1 | grep --line-buffered -E "\[acc\]|Error|Traceback"
log "chain done"
