#!/usr/bin/env bash
# GPU-host chain for the revision (ADR-0003). Phase 1, the GPU alone: where a GPU request's time goes, and the GPU
# baseline at strength (bf16; one-seq eager, torch.compile, the stock forward unpadded under FlashAttention-2 when it
# installs, padded batching; the engine cold and resident) at the traces' shapes and S=512 K=20. Phase 2, in parallel:
# GPU queue A (chunked trace-grading fine-tunes, base and large; batch-shape-dependent bf16 scores for the threshold study;
# truncated-context accuracy), GPU queue B (late-interaction and bi-encoder baselines), and the host CPU (static int8
# ONNX export, MatMul/Gemm only, scored on every split for the threshold study). Phase 3: the bf16 baselines without the
# per-shape cuDNN attention plans, and the GPU load replay. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
CHUNK_OK=1
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_model.py -k chunked > logs/chunked_tests.log 2>&1 || { CHUNK_OK=0; log "chunked tests FAILED; chunked fine-tunes skipped"; tail -20 logs/chunked_tests.log; }
log "flash-attn install (bounded; the baseline falls back to SDPA and says so)"
timeout 1200 .venv/bin/pip install -q flash-attn --no-build-isolation > logs/flash_attn_install.log 2>&1 && log "flash-attn installed" || log "flash-attn not available; hf cells run SDPA"
log "phase 1: GPU overhead breakdown and GPU baselines (GPU exclusive)"
$PY scripts/gpu_overhead.py --task routing --S 512 --K 1 20 --n 30 2>&1 | tee -a logs/gpu_overhead.log
cells() { case $1 in tracegrade) echo "384:4 2048:4 512:20";; *) echo "32:2 1024:2 512:20";; esac; }
for task in routing guardrail tracegrade; do
  log "GPU baselines bf16 $task"
  $PY scripts/latency_sweep.py --size base --task $task --device cuda --dtype bf16 --n 100 --cells $(cells $task) \
      --paths one-seq compile-one-seq hf-one-seq hf-batched naive-batch engine-cold engine-hit --out-tag shape 2>&1 | grep --line-buffered -E "^S=|Error|error" | tee -a logs/gpu_baselines_$task.log
done
log "phase 2: parallel queues"
(
  for task in routing guardrail tracegrade; do
    $PY scripts/path_scores.py --size base --task $task --paths ort-int8-static --threads 16 --out-suffix x86_64-linux 2>&1 | grep --line-buffered -E "acc=|Error" | tee -a logs/static_int8_$task.log
  done
  echo "[h100] cpu queue done"
) > logs/queue_cpu.log 2>&1 &
CPU=$!
(
  for size in base large; do
    name="$size-chunked-tracegrade"
    [ "$CHUNK_OK" = 1 ] && { [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --size $size --mode chunked --task tracegrade --epochs 10 2>&1 | grep --line-buffered -v "it/s\]" | tee -a logs/ft_$name.log; }
  done
  for task in routing guardrail tracegrade; do
    $PY scripts/path_scores.py --size base --task $task --paths torch-cuda-bf16-bs1 torch-cuda-bf16-bs64 --out-suffix h100 2>&1 | grep --line-buffered -E "acc=|Error" | tee -a logs/batchshape_$task.log
    $PY scripts/trunc_eval.py --size base --task $task --tokens 64 128 256 --device cuda 2>&1 | tee -a logs/trunc_$task.log
  done
  echo "[h100] gpu queue A done"
) > logs/queue_a.log 2>&1 &
QA=$!
# queue B: one process per fine-tune, CPU threads capped (six trainers on 208 cores oversubscribe otherwise); routing at
# 2 epochs to fit the session; the two guardrail fine-tunes one after the other (about 26 GB each)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ft() {  # kind task epochs [max-train]
  local name="base-$1-$2" mt=""; [ -n "${4:-}" ] && mt="--max-train $4"
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune_late.py --size base --kind $1 --task $2 --epochs $3 $mt 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
}
(
  ft late routing 2 & ft bi routing 2 & ft late tracegrade 10 & ft bi tracegrade 10 &
  (ft late guardrail 1 15000; ft bi guardrail 1 15000) &
  wait
  echo "[h100] gpu queue B done"
) > logs/queue_b.log 2>&1 &
QB=$!
wait $CPU $QA $QB
cat logs/queue_cpu.log logs/queue_a.log logs/queue_b.log | grep -E "acc=|done|Error|Traceback" | tail -60
# phase 3, GPU and host exclusive again. With PyTorch's defaults on the H100 every first-seen sequence length builds a
# cuDNN attention plan (about 30 ms, against about 2 ms of kernels per forward), so the bf16 baselines are timed again
# with the cuDNN backend off, and with the default backend on shapes already seen; then the guardrail per-turn replay
# (system-prompt subset) on the GPU for the engine and both baselines
for variant in "nocudnn --no-cudnn-sdp" "warm --warm-shapes"; do
  set -- $variant
  for task in routing guardrail tracegrade; do
    log "GPU baselines bf16 $task shape-$1"
    $PY scripts/latency_sweep.py --size base --task $task --device cuda --dtype bf16 --n 100 --cells $(cells $task) \
        --paths one-seq compile-one-seq hf-one-seq hf-batched naive-batch engine-cold engine-hit --out-tag shape-$1 $2 2>&1 | grep --line-buffered -E "^S=|Error|error"
  done
done
for srv in engine naive-batch one-seq; do
  log "GPU replay $srv"
  $PY scripts/load_drift.py --size base --task guardrail --device cuda --dtype bf16 --no-cudnn-sdp --threads 8 --max-state-tokens 1024 --max-batch-tokens 4096 \
      --slicing per-turn --subset system-prompt --server $srv --rates 5 10 20 40 --max-requests 1500 --max-wall 300 --tag e1-gpu-$srv 2>&1 | grep --line-buffered -E "^r|Error"
done
log "h100 chain done"
