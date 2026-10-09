"""The encoder-decoder baseline (Flan-T5 base, contract encoded once, hypotheses decoded against it) against the engine
with ModernBERT base on contracts: accuracy and AURC (paired bootstrap 10,000 resamples clustered by contract), model
FLOPs per contract, cache bytes per token, and latency measured on the same host (the workstation GPU, MPS, fp32).

    .venv/bin/python scripts/encdec_compare.py
Writes results/encdec_compare/summary.json.
"""
from __future__ import annotations

import json

from onepass.analysis import paired_compare, read_scores
from onepass.bench.timing import TimingCell
from onepass.cli import RESULTS, write_json

ED = RESULTS / "encdec_finetune" / "encdec-t5-base-contractnli"
EDL = RESULTS / "latency" / "arm64-darwin-mps-fp32-encdec-encdec-t5-base-contractnli-contractnli"
ENL = RESULTS / "latency" / "arm64-darwin-mps-base-contractnli-t8-mps-encdec-stock"
FULL = RESULTS / "finetune" / "base-full-contractnli-lr2e5-bestacc"

if __name__ == "__main__":
    t5 = read_scores(ED / "scores_test.jsonl")
    out = {"encdec": json.loads((ED / "summary.json").read_text())["splits"]["test"], "encdec_meta": json.loads((EDL / "meta.json").read_text()),
           "engine_flops_per_contract": json.loads((RESULTS / "attention_kernels" / "model_flops.json").read_text())["encoders"]["base"]["engine_flops"],
           "engine_kb_per_token": json.loads((RESULTS / "cache_study" / "x86_64-linux-cuda-bf16" / "summary.json").read_text())["analytic"]["encoders"]["base"]["trimmed_bytes_per_token_bf16"] / 1024}
    for name, f in (("served-masked", FULL / "scores_test_as_block.jsonl"), ("full", FULL / "scores_test.jsonl")):
        c = paired_compare(read_scores(f), t5, cluster=True)
        out[f"engine_{name}_minus_encdec"] = {"accuracy_pp": c["accuracy_pp"], "aurc": c["aurc"], "n_groups": c["n_groups"]}
    lat = {}
    for k, d, path in (("encdec_cold", EDL, "encdec-cold"), ("encdec_resident", EDL, "encdec-resident"), ("engine_cold", ENL, "engine-cold"), ("engine_hit", ENL, "engine-hit")):
        f = next(iter(sorted(d.glob(f"cell_S4096_K17_{path}*.json"))), None)
        lat[k] = TimingCell.load(f).p50_ms if f else None
    out["latency_ms_mps_S4096_K17"] = lat
    write_json(RESULTS / "encdec_compare" / "summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "encdec_meta"}, indent=1, default=str)[:1500])
