#!/usr/bin/env bash
# GPU-host chain for the contract-review task (ContractNLI, whole contracts as states, 17 hypotheses each): full-
# attention and block-mask fine-tunes at both sizes, the full-attention model's accuracy with the contract truncated,
# and the bf16 GPU baselines at the task's shape (cuDNN attention off). Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[cnli] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_contractnli.py > logs/cnli_tests.log 2>&1 || { log "tests FAILED"; tail -20 logs/cnli_tests.log; exit 1; }
ft() {  # size mode epochs [seed]
  local name="$1-$2-contractnli" sd=""; [ -n "${4:-}" ] && { name="$name-s$4"; sd="--seed $4 --tag s$4"; }
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --size $1 --mode $2 --task contractnli --epochs $3 $sd 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(grep -o '"accuracy": [0-9.]*' results/finetune/$name/summary.json 2>/dev/null | head -1)"
}
log "phase 1: base full and block"
ft base full 5 & ft base block 5 & wait
log "phase 2: large full and block; truncated-context accuracy of the base full-attention model"
ft large full 4 & ft large block 4 &
$PY scripts/trunc_eval.py --size base --task contractnli --tokens 256 512 1024 2048 --device cuda > logs/cnli_trunc.log 2>&1 &
wait
grep acc= logs/cnli_trunc.log
log "phase 3: GPU baselines bf16 at the contract shape (GPU exclusive)"
$PY scripts/latency_sweep.py --size base --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --n 50 --cells 2048:17 4096:17 \
    --paths one-seq compile-one-seq hf-one-seq hf-batched naive-batch engine-cold engine-hit --out-tag shape-nocudnn 2>&1 | grep --line-buffered -E "^S=|Error|error"
log "phase 4: seed replicates at the base size"
ft base full 5 1 & ft base block 5 1 & wait
log "cnli chain done"
