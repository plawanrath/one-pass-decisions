#!/usr/bin/env bash
# GPU-host chain (review round 5): the engine's block mask executed as a FlexAttention block mask. Phase 0: CUDA check,
# tests, CUAD pools, request lists. Phase 1, GPU exclusive: agreement and kernel profile (which SDPA backend the
# additive mask selects; what replaces it; model FLOP/s), then the engine with FlexAttention beside the SDPA engine and
# the strongest baselines in the same session (base and large on contracts, the K sweep, clauses, the windowed state)
# and throughput. Phase 2, two GPU queues: (A) SmolLM2-360M fine-tuned as the smaller decoder baseline (two learning
# rates per task); (B) each arm at its own best development-accuracy checkpoint within the shared step budget (base
# ModernBERT and EuroBERT-210m on contracts; the selected weights are kept for release). Phase 3, GPU exclusive: the
# smaller decoder served by vLLM with prefix caching. Every step skips when its artifact exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
VPY="env PYTHONPATH=. .venv-vllm/bin/python -u"
log() { echo "[flex] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
VLLM_VERSION=0.30.0

.venv/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda', ok, torch.version.cuda); sys.exit(0 if ok else 1)" || { log "CUDA unusable on this host"; exit 1; }
log "phase 0: tests on CUDA"
.venv/bin/python -m pytest -q -p no:cacheprovider tests > logs/flex_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/flex_tests.log; exit 1; }
grep -E "passed|failed" logs/flex_tests.log | tail -1
if [ ! -f data/pools/cuad/train.jsonl ]; then
  want=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  mv data/pools/cuad/POOL.json data/pools/cuad/POOL.tracked.json
  $PY scripts/lock_pools.py --task cuad || { log "pool lock FAILED"; exit 1; }
  got=$(.venv/bin/python -c "import json;print(json.load(open('data/pools/cuad/POOL.json'))['manifest_sha256'])")
  [ "$want" = "$got" ] || { log "CUAD manifest mismatch: $got != $want"; exit 1; }
  rm data/pools/cuad/POOL.tracked.json
fi
$PY scripts/dump_requests.py --task contractnli --cells 2048:17 4096:17 4096:1 4096:2 4096:5 4096:10 --n 80 --test-docs 20
$PY scripts/dump_requests.py --task cuad --cells 8192:41 32768:41 --n 40 --test-docs 10

log "phase 1: agreement and kernel profile (GPU exclusive)"
for size in base large; do
  $PY scripts/attention_kernels.py --size $size --task contractnli --cell 4096:17 --device cuda --n 20 2>&1 | grep --line-buffered -E "\[kernels\]|Error|Traceback"
done
log "phase 1: latency, SDPA and FlexAttention engines beside the baselines (same session)"
STRONG="compile-one-seq varlen-one-seq varlen-batched naive-batch engine-cold engine-cold-fill engine-hit"
for size in base large; do
  for kern in sdpa flex; do
    paths="$STRONG"; [ $kern = flex ] && paths="engine-cold engine-cold-fill engine-hit"
    $PY scripts/latency_sweep.py --size $size --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 50 --cells 2048:17 4096:17 \
        --paths $paths --engine-kernel $kern --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
    $PY scripts/latency_sweep.py --size $size --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 30 --cells 4096:1 4096:2 4096:5 4096:10 \
        --paths $paths --engine-kernel $kern --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  done
done
for kern in sdpa flex; do
  paths="$STRONG"; [ $kern = flex ] && paths="engine-cold engine-cold-fill engine-hit"
  $PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 30 --cells 8192:41 \
      --paths $paths --engine-kernel $kern --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  wpaths="varlen-batched engine-windowed-cold engine-windowed-hit"; [ $kern = flex ] && wpaths="engine-windowed-cold engine-windowed-hit"
  $PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 20 --cells 32768:41 \
      --paths $wpaths --engine-kernel $kern --out-tag strength-flexsession-windowed 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
log "phase 1: throughput"
for size in base large; do
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size $size --stock --device cuda --dtype bf16 --tag flexsession-$size 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size $size --stock --device cuda --dtype bf16 --engine-kernel flex --no-baseline --tag flexsession-$size-flex 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
done
$PY scripts/engine_throughput.py --task cuad --cell 8192:41 --N 1 4 16 32 --stock --device cuda --dtype bf16 --engine-kernel flex --no-baseline --tag flexsession-flex 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"

log "phase 2: smaller decoder and the per-arm best-checkpoint protocol (vLLM installs meanwhile)"
(
  python3.11 -m venv .venv-vllm && .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" ninja > logs/vllm_install.log 2>&1 \
    && .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> logs/vllm_install.log 2>&1 \
    && echo ok > logs/vllm_ready || echo failed > logs/vllm_ready
) &
BG=$!
dft() {  # name task lr
  local name=$1 task=$2 lr=$3
  [ -f "results/decoder_finetune/$name/summary.json" ] || $PY scripts/decoder_finetune.py --task $task --decoder smol360 --lr $lr --epochs 3 --tag ${name##*-} > logs/ft_$name.log 2>&1
  log "done $name: $(grep -E '^\[decoder\] test' logs/ft_$name.log | tail -1)"
}
(
  dft decoder-smol360-contractnli-lr3e5 contractnli 3e-5
  dft decoder-smol360-contractnli-lr1e4 contractnli 1e-4
  dft decoder-smol360-cuad-lr3e5 cuad 3e-5
  dft decoder-smol360-cuad-lr1e4 cuad 1e-4
  log "queue A done"
) > logs/flex_queue_a.log 2>&1 &
QA=$!
ft() {  # name task args...
  local name=$1 task=$2; shift 2
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task $task "$@" 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;print(round(json.load(open('results/finetune/$name/summary.json'))['splits']['test']['accuracy'],4))" 2>/dev/null)"
}
(
  # one queue, sequentially: the earlier session showed concurrent large jobs running out of memory
  ft base-full-contractnli-bestacc contractnli --size base --mode full --epochs 5 --select-points 10 --select-metric accuracy --tag bestacc
  ft base-block-contractnli-bestacc contractnli --size base --mode block --tokens-per-batch 16384 --max-seq-tokens 5120 --match-steps-of base-full-contractnli --select-points 10 --select-metric accuracy --tag bestacc
  ft eurobert-base-full-contractnli-bestacc contractnli --size eurobert-base --mode full --epochs 5 --micro-tokens 8192 --select-points 10 --select-metric accuracy --tag bestacc
  ft eurobert-base-block-contractnli-bestacc contractnli --size eurobert-base --mode block --tokens-per-batch 16384 --max-seq-tokens 5120 --micro-tokens 5120 --match-steps-of eurobert-base-full-contractnli --select-points 10 --select-metric accuracy --tag bestacc
  log "queue B done"
) > logs/flex_queue_b.log 2>&1 &
QB=$!
wait $QA $QB
cat logs/flex_queue_a.log logs/flex_queue_b.log
wait $BG

log "phase 3: the smaller decoder served with prefix caching (GPU exclusive)"
export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
best() {
  .venv/bin/python - "$1" <<'EOF'
import json, sys
from pathlib import Path
runs = [p for p in Path("results/decoder_finetune").glob(f"decoder-smol360-{sys.argv[1]}-lr*") if (p / "summary.json").exists()]
print(max(runs, key=lambda p: json.loads((p / "summary.json").read_text())["splits"]["dev"]["accuracy"]).name if runs else "")
EOF
}
if [ "$(cat logs/vllm_ready 2>/dev/null)" = ok ]; then
  for task in contractnli cuad; do
    b=$(best $task); [ -n "$b" ] || continue
    echo "$b" > results/decoder_finetune/selected_smol360_$task.txt
    if [ $task = contractnli ]; then cells="4096:17"; tp="1 4 16 64"; else cells="8192:41"; tp="1 4 16 32"; fi
    $VPY scripts/decoder_serve.py --backend vllm --model models/$b --task $task --cells $cells --n 50 --warmup 10 \
        --max-model-len 8192 --gpu-mem 0.85 --throughput $tp 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  done
else
  log "vLLM did not install"; tail -5 logs/vllm_install.log
fi
# keep only the weights that will be released: the selected per-arm checkpoints and the selected smaller decoders
for d in models/decoder-smol360-*; do
  n=$(basename $d); grep -qx "$n" results/decoder_finetune/selected_smol360_*.txt 2>/dev/null || rm -rf "$d"
done
log "chain done"
# phase 4 (appended): the split kernel — each row's state prefix as unmasked variable-length FlashAttention (local layers
# windowed), question rows masked — in the same session, GPU exclusive after every other phase
log "phase 4: split kernel"
for size in base large; do
  $PY scripts/attention_kernels.py --size $size --task contractnli --cell 4096:17 --device cuda --n 20 --tag -split 2>&1 | grep --line-buffered -E "\[kernels\]|Error|Traceback"
  $PY scripts/latency_sweep.py --size $size --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 50 --cells 2048:17 4096:17 \
      --paths engine-cold engine-cold-fill engine-hit --engine-kernel split --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  $PY scripts/latency_sweep.py --size $size --task contractnli --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 30 --cells 4096:1 4096:2 4096:5 4096:10 \
      --paths engine-cold engine-cold-fill engine-hit --engine-kernel split --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size $size --stock --device cuda --dtype bf16 --engine-kernel split --no-baseline --tag flexsession-$size-split 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
done
$PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 30 --cells 8192:41 \
    --paths engine-cold engine-cold-fill engine-hit --engine-kernel split --out-tag strength-flexsession 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
$PY scripts/latency_sweep.py --size base --task cuad --device cuda --dtype bf16 --no-cudnn-sdp --stock --n 20 --cells 32768:41 \
    --paths engine-windowed-cold engine-windowed-hit --engine-kernel split --out-tag strength-flexsession-windowed 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
$PY scripts/engine_throughput.py --task cuad --cell 8192:41 --N 1 4 16 32 --stock --device cuda --dtype bf16 --engine-kernel split --no-baseline --tag flexsession-split 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
log "phase 4 done"
# phase 5 (appended): numerics or bug? both kernels in fp32, and the bf16 noise floor (SDPA bf16 vs SDPA fp32)
log "phase 5: fp32 agreement"
$PY scripts/flex_agreement.py --size base --task contractnli --cell 4096:17 --requests 10 2>&1 | grep --line-buffered -E "\[agreement\]|Error|Traceback"
log "phase 5 done"
