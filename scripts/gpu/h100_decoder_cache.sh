#!/usr/bin/env bash
# GPU-host chain for ADR-0006 (only what the workstation cannot run). Phase 0: tests on CUDA, CUAD pools rebuilt and
# verified, request lists. Phase 1, GPU exclusive (nothing else running on the host): the cache study (bytes, cold with the fused fill, hit, trimmed and untrimmed), the GPU baseline at strength
# re-timed beside the fused engine, and throughput against concurrent contracts. Phase 2, three GPU queues: (A) the
# decoder baseline fine-tunes (SmolLM2-1.7B, contracts and clauses, two learning rates); (B) the encoder reruns (large
# guardrail and trace grading at equal steps; dev-AURC checkpoint selection on contracts and clauses); (C) the
# EuroBERT-610m full-attention arm at two learning rates; meanwhile a separate vLLM environment and the System 2
# weights install. Phase 3, GPU exclusive: the decoders served with prefix
# caching in vLLM (latency, throughput, agreement with the packed scoring), the System 2 model's latency, and the
# engine re-timed at the end. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
VPY="env PYTHONPATH=. .venv-vllm/bin/python -u"
log() { echo "[h100] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
MISTRAL_REV=95a6d26c4bfb886c58daf9d3f7332c857cb27b43
VLLM_VERSION=0.30.0

log "phase 0: tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests > logs/h100_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/h100_tests.log; exit 1; }
grep -E "passed|failed" logs/h100_tests.log | tail -1
if [ ! -f data/pools/cuad/train.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
  log "CUAD pools verified ($got)"
fi
$PY scripts/dump_requests.py --task contractnli --cells 2048:17 4096:17 4096:1 4096:2 4096:5 4096:10 --n 80 --test-docs 20
$PY scripts/dump_requests.py --task cuad --cells 8192:41 32768:41 --n 40 --test-docs 10

log "phase 1: cache study, strength re-timed with the fused engine, throughput (GPU exclusive)"
[ -f results/cache_study/x86_64-linux-cuda-bf16/summary.json ] || $PY scripts/cache_study.py --device cuda --dtype bf16 --n 30 --warmup 5 2>&1 | grep --line-buffered -E "^S=|KB/token|Error|Traceback"
STRONG="one-seq compile-one-seq varlen-one-seq varlen-batched naive-batch engine-cold engine-cold-fill engine-cold-fill-unfused engine-hit"
lat() { local task=$1 tag=$2; shift 2
  $PY scripts/latency_sweep.py --size base --task $task --device cuda --dtype bf16 --no-cudnn-sdp --cells "$@" \
      --paths $STRONG ${LAT_EXTRA:-} --out-tag $tag 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"; }
LAT_EXTRA="--n 50" lat contractnli strength-fused 2048:17 4096:17
LAT_EXTRA="--n 30 --stock" lat cuad strength-fused 8192:41
# the K sweep on contracts for the latency-against-K figure (S = 4,096)
$PY scripts/latency_sweep.py --size base --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --n 30 --cells 4096:1 4096:2 4096:5 4096:10 \
    --paths compile-one-seq varlen-batched naive-batch engine-cold engine-cold-fill engine-hit --out-tag strength-fused 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"
$PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --n 20 --cells 32768:41 --stock \
    --paths varlen-batched engine-windowed-cold engine-windowed-hit --out-tag strength-fused-windowed 2>&1 | grep --line-buffered -E "^S=|Error|error|Traceback"
$PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --device cuda --dtype bf16 --tag fused 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
$PY scripts/engine_throughput.py --task cuad --cell 8192:41 --N 1 4 16 32 --stock --device cuda --dtype bf16 --tag fused 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"

log "phase 2: fine-tunes in three GPU queues (vLLM and the System 2 weights install meanwhile)"
# vLLM in its own environment (it pins its own torch); the System 2 weights in their native format (one 48 GB file)
(
  python3.11 -m venv .venv-vllm && .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" > logs/vllm_install.log 2>&1 \
    && .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> logs/vllm_install.log 2>&1 \
    && echo ok > logs/vllm_ready || echo failed > logs/vllm_ready
  .venv-vllm/bin/hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506 --revision $MISTRAL_REV \
      --include consolidated.safetensors params.json tekken.json SYSTEM_PROMPT.txt --local-dir models/mistral-small-3.2 \
      > logs/mistral_download.log 2>&1 && echo ok > logs/mistral_ready || echo failed > logs/mistral_ready
) &
BG=$!
dft() {  # name task lr
  local name=$1 task=$2 lr=$3
  [ -f "results/decoder_finetune/$name/summary.json" ] || $PY scripts/decoder_finetune.py --task $task --decoder smol1.7 --lr $lr --epochs 3 --tag ${name##*-} > logs/ft_$name.log 2>&1
  log "done $name: $(grep -E '^\[decoder\] test' logs/ft_$name.log | tail -1)"
}
(
  dft decoder-smol1.7-contractnli-lr1e5 contractnli 1e-5
  dft decoder-smol1.7-contractnli-lr3e5 contractnli 3e-5
  dft decoder-smol1.7-cuad-lr1e5 cuad 1e-5
  dft decoder-smol1.7-cuad-lr3e5 cuad 3e-5
  log "queue A done"
) > logs/h100_queue_a.log 2>&1 &
QA=$!
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
(
  ft large-block-tracegrade-steps tracegrade --size large --mode block --tokens-per-batch 16384 --match-steps-of large-full-tracegrade --tag steps
  ft large-block-guardrail-steps guardrail --size large --mode block --tokens-per-batch 8192 --match-steps-of large-full-guardrail --tag steps
  ft base-block-cuad-sel cuad --size base --mode block --max-questions-per-seq 41 --max-seq-tokens 12288 --max-seqs 1 --match-steps-of base-full-cuad --select-points 10 --tag sel
  ft base-windowed-cuad-sel cuad --size base --mode windowed --max-questions-per-seq 41 --max-seqs 1 --match-steps-of base-full-cuad --select-points 10 --tag sel --other-modes block
  ft large-full-contractnli-sel contractnli --size large --mode full --epochs 4 --select-points 10 --tag sel
  ft large-block-contractnli-sel contractnli --size large --mode block --max-seq-tokens 5120 --match-steps-of large-full-contractnli --select-points 10 --tag sel
  log "queue B done"
) > logs/h100_queue_b.log 2>&1 &
QB=$!
(
  ft eurobert-large-full-contractnli-lr1e5 contractnli --size eurobert-large --mode full --epochs 4 --micro-tokens 8192 --lr 1e-5 --tag lr1e5
  ft eurobert-large-full-contractnli-lr2e5 contractnli --size eurobert-large --mode full --epochs 4 --micro-tokens 8192 --lr 2e-5 --tag lr2e5
  log "queue C done"
) > logs/h100_queue_c.log 2>&1 &
QC=$!
wait $QA $QB $QC
cat logs/h100_queue_a.log logs/h100_queue_b.log logs/h100_queue_c.log
# encoder weights are not fetched (the scores are the result): free the disk for the System 2 weights and serving
rm -rf models/modernbert-large-* models/eurobert-large-* models/modernbert-base-*-sel
wait $BG

log "phase 3: decoders served with prefix caching (GPU exclusive)"
if [ "$(cat logs/vllm_ready 2>/dev/null)" = ok ]; then BACKEND=vllm; DPY=$VPY; else BACKEND=hf; DPY=$PY; log "vLLM did not install; HF prefix-cache fallback"; tail -5 logs/vllm_install.log; fi
best() {  # task -> the decoder run with the higher dev accuracy
  .venv/bin/python - "$1" <<'EOF'
import json, sys
from pathlib import Path
runs = [p for p in Path("results/decoder_finetune").glob(f"decoder-smol1.7-{sys.argv[1]}-lr*") if (p / "summary.json").exists()]
print(max(runs, key=lambda p: json.loads((p / "summary.json").read_text())["splits"]["dev"]["accuracy"]).name if runs else "")
EOF
}
for task in contractnli cuad; do
  b=$(best $task)
  [ -n "$b" ] || { log "no decoder for $task"; continue; }
  echo "$b" > results/decoder_finetune/selected_$task.txt
  if [ $task = contractnli ]; then cells="2048:17 4096:17"; tp="1 4 16 64"; else cells="8192:41"; tp="1 4 16 32"; fi
  $DPY scripts/decoder_serve.py --backend $BACKEND --model models/$b --task $task --cells $cells --n 50 --warmup 10 \
      --max-model-len 8192 --gpu-mem 0.85 --throughput $tp 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  [ $task = contractnli ] && $DPY scripts/decoder_serve.py --backend $BACKEND --model models/$b --task $task --cells 4096:1 4096:2 4096:5 4096:10 \
      --n 30 --warmup 5 --max-model-len 8192 --gpu-mem 0.85 --agreement-docs 0 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
if [ $BACKEND = vllm ] && [ "$(cat logs/mistral_ready 2>/dev/null)" = ok ]; then
  $VPY scripts/decoder_serve.py --backend vllm --mistral models/mistral-small-3.2 --task contractnli --cells 4096:17 --n 30 --warmup 5 \
      --max-model-len 8192 --gpu-mem 0.92 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  $VPY scripts/decoder_serve.py --backend vllm --mistral models/mistral-small-3.2 --task cuad --cells 8192:41 32768:41 --n 20 --warmup 3 \
      --max-model-len 36864 --gpu-mem 0.92 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
else
  log "System 2 latency skipped (vLLM or weights unavailable)"
fi
# the engine and the strongest baseline once more at the end, to bound run-to-run drift around the decoder cells
LAT_EXTRA="--n 50" lat contractnli strength-fused-end 2048:17 4096:17
LAT_EXTRA="--n 30 --stock" lat cuad strength-fused-end 8192:41
log "chain done"
# phase 4 (appended after the first cache study ran out of memory at 32k: every timing engine kept all its entries):
# the cache study again, one entry resident at a time, GPU exclusive after everything else
log "phase 4: cache study (one entry resident)"
$PY scripts/cache_study.py --device cuda --dtype bf16 --n 30 --warmup 5 2>&1 | grep --line-buffered -E "^S=|KB/token|Error|Traceback"
log "phase 4 done"
# phase 6 (inserted before phase 5: in phase 3 vLLM's engine could not start because its kernel compiler needs the ninja
# binary on PATH). The decoders and the System 2 model served again, GPU exclusive, before the phase-5 training reruns.
log "phase 6: decoders served with prefix caching (ninja on PATH)"
.venv-vllm/bin/pip install -q ninja > logs/vllm_ninja.log 2>&1
export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
command -v ninja || log "ninja still missing"
for task in contractnli cuad; do
  b=$(cat results/decoder_finetune/selected_$task.txt 2>/dev/null)
  [ -n "$b" ] || b=$(best $task)
  [ -n "$b" ] || { log "no decoder for $task"; continue; }
  echo "$b" > results/decoder_finetune/selected_$task.txt
  if [ $task = contractnli ]; then cells="2048:17 4096:17"; tp="1 4 16 64"; else cells="8192:41"; tp="1 4 16 32"; fi
  $VPY scripts/decoder_serve.py --backend vllm --model models/$b --task $task --cells $cells --n 50 --warmup 10 \
      --max-model-len 8192 --gpu-mem 0.85 --throughput $tp 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  [ $task = contractnli ] && $VPY scripts/decoder_serve.py --backend vllm --model models/$b --task $task --cells 4096:1 4096:2 4096:5 4096:10 \
      --n 30 --warmup 5 --max-model-len 8192 --gpu-mem 0.85 --agreement-docs 0 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
if false; then  # System 2 latency dropped: this vLLM cannot load its Pixtral architecture with the installed transformers, and the weights file did not download
  $VPY scripts/decoder_serve.py --backend vllm --mistral models/mistral-small-3.2 --task contractnli --cells 4096:17 --n 30 --warmup 5 \
      --max-model-len 8192 --gpu-mem 0.92 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  $VPY scripts/decoder_serve.py --backend vllm --mistral models/mistral-small-3.2 --task cuad --cells 8192:41 32768:41 --n 20 --warmup 3 \
      --max-model-len 36864 --gpu-mem 0.92 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
fi
LAT_EXTRA="--n 50" lat contractnli strength-fused-end 2048:17 4096:17
log "phase 6 done"
# phase 5 (appended: in phase 2 three queues did not fit the GPU together; the windowed clause run and both EuroBERT-610m
# runs ran out of memory at their first steps). Rerun with the GPU to themselves after every timed cell.
log "phase 5: reruns of the out-of-memory phase-2 runs"
ft base-windowed-cuad-sel cuad --size base --mode windowed --max-questions-per-seq 41 --max-seqs 1 --match-steps-of base-full-cuad --select-points 10 --tag sel --other-modes block
rm -f logs/ft_eurobert-large-full-contractnli-lr1e5.log logs/ft_eurobert-large-full-contractnli-lr2e5.log
ft eurobert-large-full-contractnli-lr1e5 contractnli --size eurobert-large --mode full --epochs 4 --micro-tokens 8192 --lr 1e-5 --tag lr1e5 &
E1=$!
ft eurobert-large-full-contractnli-lr2e5 contractnli --size eurobert-large --mode full --epochs 4 --micro-tokens 8192 --lr 2e-5 --tag lr2e5 &
E2=$!
wait $E1 $E2
log "phase 5 done"
# phase 7 (appended: in phase 6 vLLM's FlashInfer sampler failed to JIT-compile with the host's CUDA 12.4 compiler;
# the PyTorch sampler needs no build, smoke-tested on this host). Decoders served, GPU exclusive after the reruns.
log "phase 7: decoders served with prefix caching (PyTorch sampler)"
export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
for task in contractnli cuad; do
  b=$(cat results/decoder_finetune/selected_$task.txt 2>/dev/null)
  [ -n "$b" ] || { log "no decoder for $task"; continue; }
  if [ $task = contractnli ]; then cells="2048:17 4096:17"; tp="1 4 16 64"; else cells="8192:41"; tp="1 4 16 32"; fi
  $VPY scripts/decoder_serve.py --backend vllm --model models/$b --task $task --cells $cells --n 50 --warmup 10 \
      --max-model-len 8192 --gpu-mem 0.85 --throughput $tp 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  [ $task = contractnli ] && $VPY scripts/decoder_serve.py --backend vllm --model models/$b --task $task --cells 4096:1 4096:2 4096:5 4096:10 \
      --n 30 --warmup 5 --max-model-len 8192 --gpu-mem 0.85 --agreement-docs 0 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
LAT_EXTRA="--n 50" lat contractnli strength-fused-end 2048:17 4096:17
log "phase 7 done"
