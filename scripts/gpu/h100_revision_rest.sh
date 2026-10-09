#!/usr/bin/env bash
# The rest of the revision's GPU work (ADR-0003, ADR-0004): the guardrail late-interaction and bi-encoder fine-tunes, the
# step-matched large block-mask contract fine-tune, and the
# guardrail static-int8 scoring on the host CPU in parallel; then, GPU and host exclusive, the bf16 baselines with the
# cuDNN attention backend off and on warmed shapes, and the guardrail per-turn replay on the GPU for the engine and
# both baselines. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "phase 1: guardrail late/bi fine-tunes (GPU) and guardrail static int8 (CPU, 96 threads)"
[ -f results/path_scores/base-guardrail-x86_64-linux/ort-int8-static/scores_test.jsonl ] || \
  OMP_NUM_THREADS=96 $PY scripts/path_scores.py --size base --task guardrail --paths ort-int8-static --threads 96 --out-suffix x86_64-linux > logs/static_int8_guardrail.log 2>&1 &
# the step-matched large block-mask contract fine-tune: block mode packs 17 questions per sequence, so 4 epochs gave it
# 615 optimizer steps against 4,761 for full attention; 30 epochs give about 4,600
[ -f results/finetune/large-block-contractnli-e30/summary.json ] || $PY scripts/finetune.py --size large --mode block --task contractnli --epochs 30 --tag e30 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_large-block-contractnli-e30.log &
for kind in late bi; do
  [ -f results/finetune/base-$kind-guardrail/summary.json ] || $PY scripts/finetune_late.py --size base --kind $kind --task guardrail --epochs 1 --max-train 15000 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_base-$kind-guardrail.log
  log "done base-$kind-guardrail: $(grep acc= logs/ft_base-$kind-guardrail.log | tail -n 1)"
done
wait
grep acc= logs/static_int8_guardrail.log
log "phase 2: bf16 GPU baselines (GPU and host exclusive)"
cells() { case $1 in tracegrade) echo "384:4 2048:4 512:20";; *) echo "32:2 1024:2 512:20";; esac; }
for variant in "nocudnn --no-cudnn-sdp" "warm --warm-shapes"; do
  set -- $variant
  for task in routing guardrail tracegrade; do
    log "GPU baselines bf16 $task shape-$1"
    $PY scripts/latency_sweep.py --size base --task $task --device cuda --dtype bf16 --n 100 --cells $(cells $task) \
        --paths one-seq compile-one-seq hf-one-seq hf-batched naive-batch engine-cold engine-hit --out-tag shape-$1 $2 2>&1 | grep --line-buffered -E "^S=|Error|error"
  done
done
log "phase 3: guardrail per-turn replay on the GPU"
for srv in engine naive-batch one-seq; do
  log "GPU replay $srv"
  $PY scripts/load_drift.py --size base --task guardrail --device cuda --dtype bf16 --no-cudnn-sdp --threads 8 --max-state-tokens 1024 --max-batch-tokens 4096 \
      --slicing per-turn --subset system-prompt --server $srv --rates 5 10 20 40 --max-requests 1500 --max-wall 300 --tag e1-gpu-$srv 2>&1 | grep --line-buffered -E "^r|Error"
done
log "h100 rest chain done"
