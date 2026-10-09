#!/usr/bin/env bash
# GPU-host chain: (1) the tight decoder path (HF, prefix cached once, every suffix in one batch against it) for
# SmolLM2-360M (fine-tuned) and SmolLM2-1.7B (pretrained: latency does not depend on the weights), contracts and clauses;
# (2) the joint-packing arm on base contracts: fine-tune (all questions of a contract in one sequence under full
# attention, the block arm's step count and selection), then answer flips under question reorder and drop, and latency
# beside the cascade engine. Timing phases run with the GPU to themselves.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[jd] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log "phase 0: tests"
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_joint.py tests/test_decoder.py tests/test_flex.py > logs/jd_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/jd_tests.log; exit 1; }
grep -E "passed|failed" logs/jd_tests.log | tail -1
log "phase 1: tight decoder path (GPU exclusive)"
for m in models/_from_h100_flex/decoder-smol360-contractnli-lr1e4:contractnli:4096:17 HuggingFaceTB/SmolLM2-1.7B-Instruct:contractnli:4096:17 \
         models/_from_h100_flex/decoder-smol360-cuad-lr1e4:cuad:8192:41 HuggingFaceTB/SmolLM2-1.7B-Instruct:cuad:8192:41; do
  IFS=: read -r path task S K <<< "$m"
  $PY scripts/decoder_serve.py --backend hf-tight --model $path --task $task --cells $S:$K --n 30 --warmup 5 --agreement-docs 0 --max-model-len 8192 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
log "phase 2: joint-packing fine-tune"
[ -f results/finetune/base-joint-contractnli-steps/summary.json ] || $PY scripts/finetune.py --task contractnli --size base --mode joint --tokens-per-batch 16384 \
    --max-seq-tokens 5120 --match-steps-of base-full-contractnli --select-points 10 --select-metric accuracy --tag steps 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_base-joint-contractnli-steps.log
log "done joint: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/base-joint-contractnli-steps/summary.json'));print('test', round(d['splits']['test']['accuracy'],4), 'aurc', round(d['splits']['test']['aurc'],4))" 2>/dev/null)"
log "phase 3: joint flips and latency (GPU exclusive)"
$PY scripts/joint_packing.py --model models/modernbert-base-joint-contractnli-steps 2>&1 | grep --line-buffered -E "\[joint\]|Error|Traceback"
log "chain done"
