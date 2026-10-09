#!/usr/bin/env bash
# GPU-host chain: the encoder-decoder baseline on contracts. Flan-T5 base and large fine-tuned as classifiers (the
# encoder reads the contract alone, the decoder one question; option-letter readout), best development-accuracy
# checkpoint, scored on dev/cal/test; then, with the GPU to itself, latency (cold: encoder + cross-attention keys and
# values + one decoder batch; resident: the decoder batch only) on the engine's request lists and throughput at 1 to 64
# contracts. Expected about 1.5 H100 hours including setup.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python -u"
log() { echo "[encdec] $(date '+%F %T') $*"; }
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log "phase 0: tests"
.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from encdec_finetune import ENCDECS; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(*ENCDECS['t5-base'][:1], revision=ENCDECS['t5-base'][1])" > /dev/null 2>&1  # the tests use the cached tokenizer
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_encdec.py tests/test_decoder.py > logs/encdec_tests.log 2>&1 || { log "tests FAILED"; tail -30 logs/encdec_tests.log; exit 1; }
grep -E "passed|failed" logs/encdec_tests.log | tail -1
log "phase 1: request lists"
$PY scripts/dump_requests.py --task contractnli --cells 2048:17 4096:17 4096:1 4096:2 4096:5 4096:10 --n 80 --test-docs 20
log "phase 2: fine-tunes (concurrent)"
ft() {  # model lr
  local name=encdec-$1-contractnli
  [ -f results/encdec_finetune/$name/summary.json ] || $PY scripts/encdec_finetune.py --task contractnli --model $1 --lr $2 --epochs 3 --select-points 6 2>&1 \
      | grep --line-buffered -v "it/s\]" > logs/ft_$name.log
  log "done $name: $(.venv/bin/python -c "import json;d=json.load(open('results/encdec_finetune/$name/summary.json'));t=d['splits']['test'];print('test', round(t['accuracy'],4), 'aurc', round(t['aurc'],4), 'step', d['train']['selected_step'])" 2>/dev/null)"
}
ft t5-base 1e-4 &
ft t5-large 5e-5 &
wait
log "phase 3: latency and throughput (GPU exclusive)"
for m in t5-base t5-large; do
  [ -d models/encdec-$m-contractnli ] || { log "no weights for $m"; continue; }
  $PY scripts/encdec_serve.py --model models/encdec-$m-contractnli --task contractnli --cells 4096:17 \
      --n 50 --warmup 10 --throughput 1 2 4 8 16 32 64 --agreement-docs 20 2>&1 | grep --line-buffered -E "^S=|agree|Error|Traceback"
  $PY scripts/encdec_serve.py --model models/encdec-$m-contractnli --task contractnli --cells 2048:17 4096:1 4096:2 4096:5 4096:10 \
      --n 50 --warmup 10 --agreement-docs 0 2>&1 | grep --line-buffered -E "^S=|Error|Traceback"
done
log "chain done"
