#!/usr/bin/env bash
# GPU-host chain: (1) large trace-grading encoder, block and full, 10 epochs, with tail truncation of over-long
# transcripts; (2) the base encoders served in bf16 and fp16 on the GPU (PyTorch, SDPA attention), scored on every
# pool split for the threshold study (results under path_scores/base-<task>-h100); (3) the tuned GPU baseline: one-seq through
# torch.compile beside the eager paths and the engine, at the traces' shapes and at S=512 K=20.
# Every step skips when its artifact exists. Usage on the host: bash scripts/gpu/h100_paths_and_large_tracegrade.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
for mode in block full; do
  name="large-$mode-tracegrade"
  if [ ! -f "results/finetune/$name/summary.json" ]; then
    log "finetune $name (10 epochs, tail truncation)"
    $PY scripts/finetune.py --size large --mode $mode --task tracegrade --epochs 10 --tokens-per-batch 16384 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_$name.log
  fi
done
for task in routing guardrail tracegrade; do
  log "path scores $task (cuda bf16, fp16)"
  $PY scripts/path_scores.py --size base --task $task --paths torch-cuda-bf16 torch-cuda-fp16 --out-suffix h100 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/pathscores_${task}_cuda.log
done
cells() { case $1 in tracegrade) echo "384:4 2048:4 512:20";; *) echo "32:2 1024:2 512:20";; esac; }
for task in routing guardrail tracegrade; do
  log "tuned GPU baseline $task"
  $PY scripts/latency_sweep.py --size base --task $task --threads 8 --device cuda --n 100 --cells $(cells $task) \
      --paths one-seq compile-one-seq naive-batch engine-cold engine-hit --out-tag shape 2>&1 | grep --line-buffered -E "^S=|Error|error" | tee -a logs/cuda_shape_$task.log
done
log "h100 chain done"
