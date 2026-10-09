# RUNBOOK

Every number in the paper has a command here. Results land in `results/<experiment>/` as JSON summaries plus
per-row JSONL; fine-tuned weights land in `models/` (not tracked). Hosts are labelled by chip family and OS only.

## 0. One-time setup

```bash
bash scripts/env/setup.sh                       # Python 3.11 venv from the frozen lock for this platform, plus `pip install -e .`
.venv/bin/python scripts/env/pin_models.py --from-lock   # downloads the pinned model revisions into the Hugging Face cache
```

`scripts/env/requirements.txt` is the loose, hand-edited set. `scripts/env/requirements.lock.<platform>.txt` is the
frozen set from a real install on that platform (`bash scripts/env/lock.sh` regenerates it). On a platform without a
lock, `setup.sh` resolves from the loose set and writes a new lock. `scripts/env/models.lock.txt` pins every upstream
model to a commit hash.

## 1. Tests

```bash
.venv/bin/python -m pytest -q -m "not slow and not gpu"     # fast subset (CI, ~5 s): layout, masks, exact-reuse invariants, engine, gate, pools, traces
.venv/bin/python -m pytest -q                               # everything, including the ONNX export round trip
```

The invariance tests in `tests/test_model.py` are the mechanism's correctness argument: the state's activations do not
change when questions are added, a question's scores do not change with the questions packed beside it, and the
cache-hit path reproduces the cold packed pass.

## 2. Data

- `data/pools/<task>/{train,dev,cal,test}.jsonl` + `POOL.json` — locked decision pools (manifest with n, seed, sha256).
  Built once by `scripts/lock_pools.py`; loaders verify the hash. Sources and licences:
  - routing: CLINC150 (`clinc/clinc_oos`, config `plus`), CC BY 3.0; official splits kept.
  - guardrail: `deepset/prompt-injections` (Apache-2.0) and `jackhhao/jailbreak-classification` (Apache-2.0); group split by turn.
  - tracegrade: trajectories from a code agent over `openai/gsm8k` (MIT), labels programmatic or System 2 self-consistency.
- `results/traces/` — request traces (guardrail pipeline with simulated Poisson arrivals; agent harness with measured step times).

## 3. Pools, traces, fine-tunes, K = 1 timings

```bash
.venv/bin/python scripts/lock_pools.py --task routing
.venv/bin/python scripts/lock_pools.py --task routing150          # the 150-option layout
.venv/bin/python scripts/lock_pools.py --task guardrail
.venv/bin/python scripts/traces_guardrail.py --rate 20
.venv/bin/python scripts/traces_agent.py --n 300 --max-steps 8 2>&1 | tee logs/traces_agent.log     # hours; resumable
.venv/bin/python scripts/silver_labels.py --samples 5 2>&1 | tee logs/silver.log                    # hours; resumable; locks the tracegrade pools
.venv/bin/python scripts/finetune.py --size base --mode full  --task routing 2>&1 | tee logs/ft_base_full_routing.log
.venv/bin/python scripts/finetune.py --size base --mode block --task routing 2>&1 | tee logs/ft_base_block_routing.log
.venv/bin/python scripts/baselines_k1.py --size base --task routing --threads 8
```

Long runs: `.venv/bin/python scripts/... 2>&1 | tee logs/<name>.log`. Fine-tunes checkpoint every 200 steps and
resume from `models/<name>/ckpt.pt`; the agent harness and the System 2 labeller skip finished items on restart.

## Rented GPU host (one H100 session)
```
.venv/bin/python scripts/gpu/provider.py session --provider runpod --hours 5 --ssh-key <passphrase-free key> --tasks "bash scripts/gpu/h100_chain.sh"   # launch, sync, run, fetch, terminate
bash scripts/gpu/h100_chain.sh 2>&1 | tee -a logs/h100_chain.log   # on the host: large fine-tunes, guardrail e2 pair, mask-price rows, CUDA K sweep
```
Every step skips when its artifact exists; results sync back into `results/finetune/large-*`, `results/mask_price/large/`,
`results/latency/x86_64-linux-cuda-*`.

## Review-driven runs: System 2 configuration, extra served paths, offered-load replay
```bash
# cascade with option-likelihood System 2 and three exemplars (default); zero-shot decode as the ablation
.venv/bin/python scripts/cascade_e2e.py --size base --task routing --n 1000 --audit 0.05 --s1-ms 8.97
.venv/bin/python scripts/cascade_e2e.py --size base --task routing --n 1000 --audit 0.05 --s1-ms 8.97 --s2-mode generate --s2-shots 0 --tag zeroshot-generate
# half-precision and compiled served paths, then the threshold study picks up every path directory
.venv/bin/python scripts/path_scores.py --size base --task routing --paths torch-mps-bf16 torch-mps-fp16
.venv/bin/python scripts/path_scores.py --size base --task routing --paths compiled-int8 compiled-int4 --threads 8
.venv/bin/python scripts/risk_paths.py --size base --task routing
# closed-loop replay by offered rate (simulated-arrival trace) or by agent count (measured agent trace)
.venv/bin/python scripts/load_drift.py --size base --task guardrail --trace results/traces/guardrail/requests.jsonl --rates 1 2 4 8 16 --max-requests 600 --threads 8 --tau 0.578
.venv/bin/python scripts/load_drift.py --size base --task tracegrade --trace results/traces/agent/requests_arrival_sample.jsonl --speedups 16 64 256 --threads 8
```

## GPU follow-up session and human labels
```bash
.venv/bin/python scripts/gpu/provider.py session --provider runpod --hours 3 --ssh-key <passphrase-free key> \
    --tasks "bash scripts/gpu/h100_paths_and_large_tracegrade.sh" \
    --exclude "modernbert-base-full-*" "*-s1" "*-s2" "*-e2" "*-e4" "*-e4-*" "compiled4" "compiled8"
.venv/bin/python scripts/risk_paths.py --size base --task routing --suffix h100          # likewise guardrail, tracegrade
.venv/bin/python scripts/human_labels.py sheet --n-steps 50 --out labels                    # blind labelling page
.venv/bin/python scripts/human_labels.py score --labels tracegrade_human_labels.json --key labels/key.json
```

## Revision: chunked mask, scheduler, GPU baselines at strength, contract review
```bash
# contract review (ContractNLI, CC BY 4.0): unpack contract-nli.zip into data/raw/contractnli, then lock the pools
.venv/bin/python scripts/lock_pools.py --task contractnli
# GPU sessions: the revision chain, its remainder, and the contract-review chain (fine-tunes, truncation, GPU cells)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 3 --ssh-key <key> --tasks "bash scripts/gpu/h100_revision.sh" --fetch-models models/_from_h100
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 3 --ssh-key <key> --tasks "bash scripts/gpu/h100_revision_rest.sh" --fetch-models models/_from_h100
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 3 --ssh-key <key> --tasks "bash scripts/gpu/h100_contractnli.sh" --fetch-models models/_from_h100_cnli
# mask price: chunked arm, contract task, step-matched large block run against the original full fine-tune
.venv/bin/python scripts/mask_price.py --size base --tasks tracegrade --arm chunked
.venv/bin/python scripts/mask_price.py --size base --tasks routing guardrail tracegrade contractnli
.venv/bin/python scripts/mask_price.py --size large --tag e30 --full-tag "" --tasks contractnli
# replays (CPU, nothing else running): per-turn guardrail (P1, E1), scheduler ablations (P2), agent trace block vs chunked (P3b)
.venv/bin/python scripts/load_drift.py --size base --task guardrail --threads 8 --max-state-tokens 1024 --max-batch-tokens 4096 --slicing per-turn --subset system-prompt --server engine --rates 1.16 2.33 3.78 5.81 --max-requests 1500 --max-wall 1200 --tag e1-sysprompt-engine
.venv/bin/python scripts/load_drift.py --size base --task guardrail --threads 8 --max-state-tokens 1024 --max-batch-tokens 4096 --trace results/traces/guardrail/requests.jsonl --server engine --order edf --split 6 --rates 1 2 --max-requests 600 --max-wall 1200 --tag p2-edf-split6
.venv/bin/python scripts/load_drift.py --size base --task tracegrade --trace results/traces/agent/requests_arrival_sample.jsonl --mask chunked --speedups 16 64 256 --threads 8 --max-batch-tokens 4096 --max-state-tokens 1024 --max-wall 1200 --tag p3-chunked
# truncated-context operating points (state capped at S) and the contract shape on the workstation
.venv/bin/python scripts/latency_sweep.py --size base --task contractnli --threads 8 --n 30 --cells 2048:17 4096:17 --paths one-seq naive-batch engine-cold engine-hit --out-tag shape
.venv/bin/python scripts/latency_sweep.py --size base --task contractnli --threads 8 --n 30 --cells 512:17 1024:17 2048:17 --paths one-seq --cap-state --out-tag pareto-capped
# static int8 thresholds (P4)
.venv/bin/python scripts/path_scores.py --size base --task guardrail --paths ort-int8-static --threads 12
.venv/bin/python scripts/risk_paths.py --size base --task tracegrade --suffix x86_64-linux
```

## Second encoder family, CUAD with a windowed state, GPU baseline at strength, chunked asymptote

```bash
# CUAD (CC BY 4.0): unpack CUAD_v1.zip (sha256 88b694d9...d18e) into data/raw/cuad, then lock the pools
.venv/bin/python scripts/lock_pools.py --task cuad
# workstation: chunked asymptote on the CPU (alone), then EuroBERT-210m on routing and contracts (MPS), mask price
bash scripts/local_encoders_asymptote.sh
# GPU host: tests on CUDA, CUAD pools rebuilt and verified, GPU baseline at strength, CUAD and EuroBERT-610m fine-tunes
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 7 --ssh-key <key> \
    --tasks "bash scripts/gpu/h100_cuad_strength.sh" --fetch-models models/_from_h100_cuad \
    --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" \
    --upload models/modernbert-base-block-contractnli models/modernbert-base-block-routing
# analysis after the fetch
.venv/bin/python scripts/mask_price.py --size base --tasks cuad --tag steps --full-tag ""           # E3, equal steps
.venv/bin/python scripts/mask_price.py --size base --tasks cuad --tag e3 --full-tag ""              # E3, equal epochs
.venv/bin/python scripts/mask_price.py --size base --tasks cuad --tag steps --arm windowed --ref block   # E4, all questions
.venv/bin/python scripts/cuad_beyond_window.py --size base --tag steps                             # E4, evidence beyond the window
.venv/bin/python scripts/mask_price.py --size eurobert-large --tasks contractnli                    # E1, large, equal epochs
.venv/bin/python scripts/mask_price.py --size eurobert-large --tasks contractnli --tag steps --full-tag ""
```

## Document review, decoder baseline, cache capacity (ADR-0006)
```bash
# workstation: block arms at equal optimizer steps (MPS), then the System 2 model on the document tasks (MLX, alone)
bash scripts/local_equal_steps.sh
.venv/bin/python scripts/system2_documents.py --task contractnli --splits test cal
.venv/bin/python scripts/system2_documents.py --task cuad --splits test cal
# risk-controlled thresholds and cascade on long documents (from the fine-tunes' scores)
.venv/bin/python scripts/risk_long.py
.venv/bin/python scripts/cascade_long.py
# GPU host: cache study, strength with the fused engine, throughput, decoder fine-tunes and vLLM serving, reruns
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 8 --ssh-key ~/.ssh/onepass_rental \
    --tasks "bash scripts/gpu/h100_decoder_cache.sh" \
    --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache" \
    --upload models/modernbert-base-block-contractnli results/finetune/large-full-tracegrade/summary.json \
             results/finetune/large-full-guardrail/summary.json results/finetune/base-full-cuad/summary.json results/finetune/large-full-contractnli/summary.json
# after the fetch: the replay over measured latencies and bytes
.venv/bin/python scripts/contract_replay.py --study results/cache_study/x86_64-linux-cuda-bf16/summary.json --task contractnli
.venv/bin/python scripts/contract_replay.py --study results/cache_study/x86_64-linux-cuda-bf16/summary.json --task cuad
```
```bash
# GPU host (follow-up): ModernBERT-large engine for the accuracy-matched decoder comparison, base anchor, base contracts at equal steps
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 1.5 --ssh-key ~/.ssh/onepass_rental \
    --cuda-versions 13.0 --tasks "bash scripts/gpu/h100_large_engine.sh" --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad" "/data/cache"
# workstation: System 2 on the document tasks (MLX), then the risk-controlled thresholds and the cascade
bash scripts/local_system2_documents.sh
# analysis and paper
.venv/bin/python scripts/decoder_compare.py
.venv/bin/python docs/paper/tools/make_tables.py && .venv/bin/python scripts/make_figures.py && .venv/bin/python docs/paper/tools/build.py --figures --strict
```

## FlexAttention engine, calibration bootstrap, per-arm best checkpoints, smaller decoder (ADR-0007)
```bash
.venv/bin/python scripts/risk_long.py            # includes the calibration bootstrap (1,000 resamples, paired across arms)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 4 --ssh-key ~/.ssh/onepass_rental \
    --cuda-versions 13.0 --tasks "bash scripts/gpu/h100_flex.sh" --fetch-models models/_from_h100_flex \
    --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache" \
    --upload results/finetune/base-full-contractnli/summary.json results/finetune/eurobert-base-full-contractnli/summary.json
```

### Attention kernels and the matched-accuracy decoder table (H100 session 11 artifacts)
```bash
# fp32/bf16 agreement of the three kernels (H100 only; the split kernel is compared in bf16, FlashAttention has no fp32)
.venv/bin/python scripts/flex_agreement.py --size base --task contractnli --cell 4096:17 --requests 10
# kernel profile and model FLOP/s (note: pass the tag with "=", it starts with a dash)
.venv/bin/python scripts/attention_kernels.py --size base --task contractnli --cell 4096:17 --device cuda --n 20 --tag=-split
# mask price under per-arm best dev-accuracy checkpoints
.venv/bin/python scripts/mask_price.py --size base --tasks contractnli --tag bestacc --full-tag bestacc
.venv/bin/python scripts/mask_price.py --size eurobert-base --tasks contractnli --tag bestacc --full-tag bestacc
# decoder comparison with the split engines, SmolLM2-360M and matched-accuracy pairs; then tables and figures
.venv/bin/python scripts/decoder_compare.py
.venv/bin/python docs/paper/tools/make_tables.py && .venv/bin/python scripts/make_figures.py
.venv/bin/python docs/paper/tools/build.py --figures --strict
```

### Cascade kernel (mask-free engine attention) and engine utilization
```bash
.venv/bin/python -m pytest -q tests/test_flex.py                   # cascade equals masked SDPA (CPU reference), both families
bash scripts/gpu/h100_cascade.sh                                   # H100: agreement, profile, latency beside split, throughput, 64-contract profile
.venv/bin/python scripts/cascade_diag.py                           # H100: varlen log-sum-exp check; plain vs fill paths per contract
.venv/bin/python scripts/cascade_accuracy.py --model models/_from_h100_flex/modernbert-base-block-contractnli-bestacc   # H100: test accuracy per kernel
.venv/bin/python scripts/engine_profile.py --size base --N 64 --kernel cascade --batch-tokens 65536 131072
.venv/bin/python scripts/model_flops.py                            # workstation: model FLOPs per contract and utilization
.venv/bin/python scripts/decoder_compare.py && .venv/bin/python docs/paper/tools/make_tables.py && .venv/bin/python scripts/make_figures.py
```


## Three arms on long documents, clustered intervals, full-arm seeds, joint packing, tight decoder path

```bash
.venv/bin/python scripts/cluster_long_doc.py          # question- vs contract-clustered bootstrap for every long-document comparison; results/cluster_long/summary.json
.venv/bin/python scripts/risk_resplit.py              # calibration/test re-splits and document-level Learn-then-Test; results/risk_resplit/summary.json
.venv/bin/python scripts/cost_model.py                # closed-form token/pair cost model against measured cells; results/cost_model/, figures/cost_model.pdf
.venv/bin/python scripts/risk_long.py                 # coverage at the risk-controlled threshold per arm (incl. served-masked)
bash scripts/gpu/h100_full_seeds.sh contracts         # H100: base contracts full arm, lr 2e-5 best dev, seeds 1 and 2 (served-masked scores included)
bash scripts/gpu/h100_full_seeds.sh clauses           # H100: base clauses full arm, seed 2
bash scripts/gpu/h100_joint_decoder.sh                # H100: tight decoder path (both decoders, both tasks), joint-packing fine-tune, flips and latency
.venv/bin/python scripts/joint_packing.py --model models/modernbert-base-joint-contractnli-steps   # flips under reorder/drop, latency
.venv/bin/python scripts/decoder_compare.py           # pairs each decoder's faster path (vLLM or hf-tight) with the matching engine
.venv/bin/python docs/paper/tools/make_tables.py      # three_arms, decoder_compare and the rest
```

## Training-length control, third task, encoder-decoder baseline (ADR-0009, ADR-0010)

```bash
bash scripts/gpu/h100_epoch_control.sh all            # H100: full arm at the mask-trained arm's epochs (contracts 3 seeds, clauses 1), dev curve, final step
bash scripts/mac_round10.sh                           # workstation: ECtHR arms, Flan-T5 base + same-host MPS latency, large seeds (serial)
.venv/bin/python scripts/lock_pools.py --task ecthr   # LexGLUE ECtHR pools (pinned revision)
.venv/bin/python scripts/ecthr_analysis.py            # three arms, length bins, document-level gate; results/ecthr/summary.json
.venv/bin/python scripts/clause_f1.py                 # clause F1 / AUPR per arm and seed; results/clause_f1/summary.json
bash scripts/gpu/h100_retime.sh                       # H100 (not yet run): re-time the main-text cells after the host-overhead cut
bash scripts/gpu/h100_encdec.sh                       # H100 (not yet run): Flan-T5 base/large fine-tunes, latency, throughput
bash scripts/gpu/h100_ecthr.sh                        # H100 alternative to the workstation ECtHR runs
bash scripts/gpu/h100_large_seeds.sh                  # H100 alternative to the workstation large seeds
```

## Brief mask adaptation (ADR-0011)

```bash
.venv/bin/python -m pytest -q tests/test_warmstart.py      # zero adaptation steps reproduce served-masked scores exactly
EXTRA="--max-train 24 --max-eval 12 --max-state-tokens 256" TAGSFX="-rh" bash scripts/gpu/h100_adapt.sh   # workstation rehearsal of the whole chain
bash scripts/gpu/h100_adapt.sh                              # H100: base rehearsals, ModernBERT-large and EuroBERT-610M retrained + adapted (keep to two jobs at a time)
bash scripts/gpu/h100_adapt_rerun.sh                        # H100: the two runs that ran out of memory with three jobs on one GPU
.venv/bin/python scripts/adapt_analysis.py                  # adapted vs full / served-masked / mask-trained; results/adapt/summary.json
```

## Round-12 baselines: multi-head, held-out questions, decoupled lower layers, throughput fix

```bash
.venv/bin/python -m pytest -q tests/test_multihead.py tests/test_decoupled.py
SKIPTP=1 EXTRA="--max-train 24 --max-eval 12 --max-state-tokens 256" MHEXTRA="--max-train 40 --max-eval 20 --max-state-tokens 256" TAGSFX="-rh" bash scripts/gpu/h100_round12.sh   # workstation rehearsal
bash scripts/gpu/h100_round12.sh                       # H100: throughput (one tokenization per contract), multi-head x3, held-out arm, decoupled l=7/11/15
.venv/bin/python scripts/deformer_cost.py              # model FLOPs per contract for decoupled depths; results/deformer_cost/summary.json
.venv/bin/python scripts/round12_compare.py            # results/round12/summary.json (multi-head, held-out, decoupled; multi-head FLOPs)
```

## A non-legal long-document task, served-masked beyond one window, joint packing re-timed, pooled seeds
```bash
# QuALITY (stories) and RACE (intermediate training only): lock the pools (QuALITY.v1.0.1.zip unpacked into data/raw/quality)
.venv/bin/python scripts/lock_pools.py --task quality
.venv/bin/python scripts/lock_pools.py --task race
# GPU host: joint-packing re-time, QuALITY-shaped latency, clause full-attention weights served under the four-window
# mask, RACE intermediate training, QuALITY full arms (base seeds 0/1/2, large seed 0) and brief adaptation
bash scripts/gpu/h100_round15.sh
bash scripts/gpu/h100_joint_flash.sh        # joint packing through FlashAttention beside the masked joint path and the engine
# workstation (existing scores): one price per encoder pooled over runs, calibration of every arm, joint-packing FLOPs
.venv/bin/python scripts/seed_pooled.py
.venv/bin/python scripts/served_windowed_compare.py
.venv/bin/python scripts/deformer_cost.py --size base --cell 4096:17 --requests 20
```

## Round 16: mask-trained seeds, a 3B decoder, an encoder server, reuse across requests
```bash
EXTRA="--max-train 24 --max-eval 12 --max-state-tokens 256 --max-steps 3" DEXTRA="--tiny --max-train-docs 2 --max-eval-docs 2 --max-state-tokens 256 --device cpu" SKIP_TIMING=1 TAGSFX=rh bash scripts/gpu/h100_round16.sh   # workstation rehearsal
bash scripts/gpu/h100_round16.sh            # H100: mask-trained seeds, SmolLM3-3B, encoder server (vLLM pooling), SmolLM3 served
bash scripts/gpu/h100_session_replay.sh     # H100: reuse across requests on two review patterns (warmed shapes)
.venv/bin/python scripts/seed_pooled.py && .venv/bin/python scripts/decoder3b_compare.py
```

## Graph-captured resident path and compiled cold pass
```bash
.venv/bin/python -m pytest -q tests/test_graphs.py                 # static-shape resident path and compiled segments == eager engine (CPU; CUDA capture test on a GPU host)
DEVICE=cpu DTYPE=fp32 LIMIT=2 TAG=rh bash scripts/gpu/h100_graph_engine.sh   # workstation rehearsal (delete results/graph_engine/*-rh.json, results/throughput/*rh-*.json)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 3 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_graph_engine.sh" --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf"
```

## Packing dose-response, pooled-table seeds, cold-path diagnosis, joint control for full-attention weights
```bash
# workstation rehearsal (CPU, tiny; deletes nothing by itself: remove results/finetune/*-rh, models/*-rh afterwards)
INIT=models TAGSFX=rh SKIP_TIMING=1 JEXTRA="--device cpu --limit-docs 1" \
  EXTRA="--max-train 24 --max-eval 12 --max-state-tokens 256 --max-steps 3 --device cpu" bash scripts/gpu/h100_packing_dose_and_seeds.sh
.venv/bin/python scripts/cold_path_diag.py --device cpu --smoke
# GPU host
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 7.5 --ssh-key ~/.ssh/onepass_rental \
    --cuda-versions 13.0 --tasks "bash scripts/gpu/h100_packing_dose_and_seeds.sh" --fetch-models models/_from_h100_dose \
    --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache" \
    --upload models/_from_h100_lr/modernbert-base-full-contractnli-lr2e5-bestacc models/_from_h100_seed/modernbert-base-full-cuad-s1 \
             models/_from_h100_round15/modernbert-base-full-quality-s1 models/_from_h100_round15/modernbert-base-full-quality-s2 \
             models/_from_h100_round15/modernbert-base-full-race models/_from_h100_round15/modernbert-large-full-race \
             results/finetune/base-full-contractnli-lr2e5-bestacc/summary.json results/finetune/base-full-contractnli-lr2e5-bestacc/scores_test.jsonl
# after the fetch
.venv/bin/python scripts/packing_dose.py
.venv/bin/python scripts/seed_pooled.py
for m in models/_from_h100_dose/*; do n=$(basename $m); .venv/bin/python scripts/verify_checkpoints.py --model $m --result ${n#modernbert-} --docs 2; done   # task per run: pass --task cuad / quality where needed
```

## Decoder resident latency (tight PyTorch path) and matched-accuracy load curve

```bash
.venv/bin/python -m pytest -q tests/test_decoder_cascade.py      # cascade decoder == per-question prefix-cached decoder (CPU; CUDA test on a GPU host)
DEV=cpu TAGSFX=rh REXTRA="--tiny --n 2 --warmup 1 --hot-states 2 --max-state-tokens 256" \
  LEXTRA="--tiny --rates 20 200 --n 6 --max-batch 4 --limit-requests 4 --max-state-tokens 256" \
  bash scripts/gpu/h100_decoder_resident.sh                      # workstation rehearsal (delete results/decoder_{resident,load}/*-rh.json after)
bash scripts/gpu/h100_decoder_resident.sh                        # H100: about 20 to 30 min after setup
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 1 --ssh-key ~/.ssh/onepass_rental \
  --tasks "bash scripts/gpu/h100_decoder_resident.sh" \
  --upload models/_from_h100_flex/decoder-smol360-contractnli-lr1e4 models/_from_h100_flex/decoder-smol360-cuad-lr1e4
```
Writes results/decoder_resident/x86_64-linux-cuda-bf16-{contractnli,cuad}.json and results/decoder_load/x86_64-linux-cuda-bf16.json.

## Cache capacity under trace-fitted popularity
```bash
.venv/bin/python -m pytest -q tests/test_cache_capacity.py
DEVICE=cpu EXTRA="--tiny --docs 8 --n 20 --warm 5 --budgets-gb 0.0005 0.005 --max-state-tokens 256 --sim-grid 5" TAG=rh \
  bash scripts/gpu/h100_cache_capacity.sh                           # workstation rehearsal (delete results/cache_capacity/*-rh.json)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 5 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_cache_capacity.sh 0.45 0.78 1.11" \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" \
  --upload results/trace_popularity/summary.json
```
Writes results/cache_capacity/x86_64-linux-cuda-bf16-{engine-base,engine-large,decoder-smol360,decoder-smol1.7,decoder-smol3}.json.

## Round 19: EuroBERT-2.1B at matched size, symmetric mask-trained seeds, compiled baseline
```bash
.venv/bin/python -m pytest -q tests/test_round19.py   # gradient checkpointing = same gradients; 2.1B-shaped EuroBERT serves masked exactly; compiled baseline == eager; bf16 save
bash scripts/gpu/h100_round19_train.sh rehearse --max-train 24 --max-eval 12 --max-state-tokens 256 --max-steps 2 --device cpu   # workstation rehearsal (EuroBERT-210m stands in; delete *-rh results/models/logs after)
bash scripts/gpu/h100_compiled_baseline.sh rehearse   # workstation rehearsal (delete results/latency/*strength-compiled-rh*)
# training pod (EuroBERT-2.1B full x2 + adapted x2, bf16 checkpoints; symmetric seeds 1/2)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 6.5 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_round19_train.sh" --fetch-models models/_from_h100_round19 \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache"
# timing pod (GPU otherwise idle)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 1.5 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_compiled_baseline.sh" \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache"
# after the fetch
for m in models/_from_h100_round19/*; do n=$(basename $m); r=${n#modernbert-}; .venv/bin/python scripts/verify_checkpoints.py --model $m --result $r --docs 2; done
.venv/bin/python scripts/seed_pooled.py && .venv/bin/python scripts/packing_dose.py
```
Writes results/finetune/eurobert-xl-{full-contractnli-lr1e5{,-s1},block-contractnli-adapt2{,-s1}}, results/finetune/base-block-contractnli-pack16-steps-s{1,2}, results/latency/x86_64-linux-cuda-bf16-{base,large}-contractnli-t8-strength-compiled-stock.

## Trace popularity, per-label price and cascade granularity (workstation, no GPU)
```bash
.venv/bin/python scripts/trace_popularity.py        # Zipf fits of the public Mooncake traces (data/raw/mooncake, pinned commit) -> results/trace_popularity
.venv/bin/python scripts/label_and_cascade.py       # served-masked price per gold label; cascade to SmolLM3-3B by escalated share -> results/label_and_cascade
```

## Full-attention weights served under other masks: chunked trace grading, EuroBERT past one window
```bash
.venv/bin/python scripts/score_served.py --size base --task tracegrade --mode chunked          # full-attention weights under the chunked mask (CPU)
.venv/bin/python scripts/score_served.py --size large --task tracegrade --mode chunked --model-dir models/cuda/modernbert-large-full-tracegrade
bash scripts/gpu/h100_eurobert_cuad.sh rehearse --max-train 8 --max-eval 6 --max-state-tokens 256 --window-tokens 256 --max-steps 2 --device cpu   # workstation rehearsal (delete *-rh after)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 7.5 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_eurobert_cuad.sh" --fetch-models models/_from_h100_eurobert_cuad \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache"
.venv/bin/python scripts/served_windowed_compare.py      # both families, per-family beyond-window subsets
```
```bash
# one more EuroBERT-2.1B seed at its tuned learning rate (contracts)
bash scripts/gpu/h100_xl_seed.sh 2 rehearse --max-train 24 --max-eval 12 --max-state-tokens 256 --max-steps 2 --device cpu   # workstation rehearsal (delete *-rh after)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 4 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_xl_seed.sh 2" --fetch-models models/_from_h100_xl_seed \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache"
```
```bash
# decoder counterpart of the served-windowed result: SmolLM2-360M clause fine-tune under four parallel windows (CPU, ~1 h)
.venv/bin/python -m pytest -q tests/test_decoder_windowed.py
.venv/bin/python scripts/decoder_windowed.py --model models/_from_h100_flex/decoder-smol360-cuad-lr1e4
```

## Window sweep, clause compute, decoder sink control
```bash
.venv/bin/python scripts/clause_flops.py                 # model FLOPs per clause contract, one and four windows (CPU)
.venv/bin/python scripts/decoder_windowed.py --model models/_from_h100_flex/decoder-smol360-cuad-lr1e4 --bos-each   # begin token per window
bash scripts/gpu/h100_window_sweep.sh rehearse           # workstation rehearsal (delete results/served_windowed/*-rh after)
.venv/bin/python scripts/gpu/provider.py session --provider runpod --gpu "NVIDIA H100 80GB HBM3" --hours 2.5 --ssh-key ~/.ssh/onepass_rental --cuda-versions 13.0 \
  --tasks "bash scripts/gpu/h100_window_sweep.sh" \
  --upload results/finetune/base-windowed-cuad-steps/summary.json models/_from_h100_cuad/modernbert-base-full-cuad models/_from_h100_seed/modernbert-base-full-cuad-s1 \
           models/_from_h100_eurobert_cuad/eurobert-large-full-cuad models/_from_h100_eurobert_cuad/eurobert-large-full-cuad-s1 \
  --exclude "/models/*" "/data/pools/cuad/*.jsonl" "/data/raw/cuad/CUAD_v1.zip" "/data/raw/cuad/CUAD_v1/full_contract_pdf" "/data/cache"
.venv/bin/python scripts/window_sweep_compare.py         # results/served_windowed/sweep.json
```
