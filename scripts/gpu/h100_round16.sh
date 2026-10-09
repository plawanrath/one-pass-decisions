#!/usr/bin/env bash
# GPU-host chain: what a careful reviewer asks next.
# Phase 1: two more mask-trained seeds on contracts (ModernBERT base, the first run's recipe: block mask, 5,951 steps).
# Phase 2: a stronger decoder, SmolLM3-3B, fine-tuned as the SmolLM2 decoders were (block-causal, letters' logits) at
# learning rates 1e-5 and 3e-5, selected on development accuracy. vLLM installs in the background meanwhile.
# Phase 3 (GPU to itself): reuse across requests on two documented review patterns (scripts/session_replay.py); the
# engine's throughput re-timed as the in-session anchor; a production encoder server (vLLM pooling runner, ModernBERT
# base, one sequence per question; scripts/encoder_server_bench.py); the selected SmolLM3 served with vLLM prefix
# caching and on the tight PyTorch path. Phase 4: two more mask-trained seeds for ModernBERT-large (the first run's
# 4,616 steps). REHEARSAL: EXTRA / DEXTRA (fine-tune / decoder arguments), TAGSFX, SKIP_TIMING=1.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
VPY="env PYTHONPATH=. .venv-vllm/bin/python -u"
EXTRA=${EXTRA:-}; DEXTRA=${DEXTRA:-}; TAGSFX=${TAGSFX:-}; SKIP_TIMING=${SKIP_TIMING:-}; SFX=${TAGSFX:+-$TAGSFX}
VLLM_VERSION=0.30.0
log() { echo "[round16] $(date '+%F %T') $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for t in contractnli cuad; do [ -f data/pools/$t/test.jsonl ] || { log "missing data/pools/$t"; exit 1; }; done
[ -f data/cache/requests/contractnli.json ] || { log "missing request cache"; exit 1; }
ft() {
  local name=$1; shift
  [ -f "results/finetune/$name/summary.json" ] || $PY scripts/finetune.py --task contractnli "$@" $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/finetune/$name/summary.json'));s=d['splits'];print('test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4))" 2>/dev/null)"
}
if [ -z "$SKIP_TIMING" ] && [ ! -f logs/vllm_ready ]; then
  ( python3.11 -m venv .venv-vllm && .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" ninja > logs/vllm_install.log 2>&1 \
    && .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> logs/vllm_install.log 2>&1 \
    && echo ok > logs/vllm_ready || echo failed > logs/vllm_ready ) &
fi
log "phase 1: mask-trained seeds 1 and 2 (contracts, base)"
for s in 1 2; do ft base-block-contractnli-steps-s$s$SFX --size base --mode block --max-seq-tokens 5120 --max-steps 5951 --seed $s --tag steps-s$s$SFX & done
wait
log "phase 2: SmolLM3-3B fine-tunes (lr 1e-5, 3e-5)"
for lr in 1e5 3e5; do
  name=decoder-smol3-contractnli-lr$lr$SFX
  [ -f results/decoder_finetune/$name/summary.json ] || $PY scripts/decoder_finetune.py --task contractnli --decoder smol3 --lr ${lr/e/e-} --epochs 3 --tag lr$lr$SFX $DEXTRA > logs/ft_$name.log 2>&1
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/decoder_finetune/$name/summary.json'));s=d['splits'];print('dev', round(s['dev']['accuracy'],4), 'test', round(s['test']['accuracy'],4), 'aurc', round(s['test']['aurc'],4))" 2>/dev/null)"
done
best=$(.venv/bin/python - "$SFX" <<'PYEOF'
import json, sys
from pathlib import Path
runs = [p for p in Path("results/decoder_finetune").glob("decoder-smol3-contractnli-lr*") if (p / "summary.json").exists() and p.name.endswith(sys.argv[1] or p.name[-3:])]
print(max(runs, key=lambda p: json.loads((p / "summary.json").read_text())["splits"]["dev"]["accuracy"]).name if runs else "")
PYEOF
)
[ -n "$best" ] && echo "$best" > results/decoder_finetune/selected_smol3_contractnli.txt && log "selected $best"
if [ -z "$SKIP_TIMING" ]; then
  wait  # vLLM install
  log "phase 3: timing (GPU to itself)"
  $PY scripts/session_replay.py --device cuda --contracts 30 2>&1 | grep --line-buffered -E "\[session\]|Error|Traceback"
  $PY scripts/engine_throughput.py --task contractnli --cell 4096:17 --N 1 4 16 64 --size base --stock --device cuda --dtype bf16 --engine-kernel cascade --tag round16-base-cascade 2>&1 | grep --line-buffered -E "^N=|Error|Traceback"
  export PATH="$PWD/.venv-vllm/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0
  if [ "$(cat logs/vllm_ready 2>/dev/null)" = ok ]; then
    $VPY scripts/encoder_server_bench.py --n 30 --throughput 1 4 16 64 2>&1 | grep --line-buffered -E "\[encoder-server\]|Error|Traceback"
    [ -n "$best" ] && $VPY scripts/decoder_serve.py --backend vllm --model models/$best --task contractnli --cells 4096:17 --n 50 --warmup 10 \
        --max-model-len 8192 --gpu-mem 0.85 --throughput 1 4 16 64 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  else
    log "vLLM did not install"; tail -5 logs/vllm_install.log
  fi
  [ -n "$best" ] && $PY scripts/decoder_serve.py --backend hf-tight --model models/$best --task contractnli --cells 4096:17 --n 30 --warmup 5 --agreement-docs 0 --max-model-len 8192 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
fi
log "phase 4: mask-trained seeds 1 and 2 (contracts, large)"
for s in 1 2; do ft large-block-contractnli-e30-s$s$SFX --size large --mode block --max-seq-tokens 5120 --max-steps 4616 --seed $s --tag e30-s$s$SFX & done
wait
log "chain done"
