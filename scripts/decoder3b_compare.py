"""A stronger decoder baseline: SmolLM3-3B fine-tuned as the SmolLM2 decoders were (block-causal, the option letters'
logits), against the ModernBERT-large engine on contracts: accuracy (paired bootstrap clustered by contract, 10,000
resamples), model FLOPs per contract (scripts/model_flops.py conventions), key-value bytes per token, and latency and
throughput on the faster of vLLM with prefix caching and the tight PyTorch path.

    .venv/bin/python scripts/decoder3b_compare.py
Reads results/decoder_finetune/selected_smol3_contractnli.txt and the latency directories decoder_serve.py writes;
the engine's numbers are the large pair's in results/decoder_compare/summary.json. Writes results/decoder3b/summary.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder_serve import Prompter  # noqa: E402
from model_flops import decoder_flops  # noqa: E402

from onepass.analysis import paired_compare, read_scores
from onepass.bench.timing import TimingCell
from onepass.cli import DECODERS, REPO, RESULTS, write_json

H = "x86_64-linux-cuda-bf16"


def p50(d: Path, path: str) -> float | None:
    f = d / f"cell_S4096_K17_{path}.json"
    return TimingCell.load(f).p50_ms if f.exists() else None


if __name__ == "__main__":
    sel = (RESULTS / "decoder_finetune" / "selected_smol3_contractnli.txt").read_text().strip()
    dec = read_scores(RESULTS / "decoder_finetune" / sel / "scores_test.jsonl")
    summ = json.loads((RESULTS / "decoder_finetune" / sel / "summary.json").read_text())
    refs = {"MB-l engine, served masked": RESULTS / "finetune" / "large-full-contractnli" / "scores_test_as_block.jsonl",
            "MB-l engine, adapted": RESULTS / "finetune" / "large-block-contractnli-adapt2" / "scores_test.jsonl",
            "MB-l full attention": RESULTS / "finetune" / "large-full-contractnli" / "scores_test.jsonl",
            "SmolLM2-1.7B": RESULTS / "decoder_finetune" / "decoder-smol1.7-contractnli-lr3e5" / "scores_test.jsonl"}
    out = {"decoder": sel, "test": summ["splits"]["test"], "dev": summ["splits"]["dev"], "kv_bytes_per_token_bf16": summ["kv_bytes_per_token_bf16"],
           "minus_decoder_pp": {}}
    for name, f in refs.items():
        c = paired_compare(read_scores(f), dec, cluster=True)
        out["minus_decoder_pp"][name] = {"accuracy_pp": c["accuracy_pp"], "aurc": c["aurc"]}
        print(f"[3b] {name} minus SmolLM3-3B: {c['accuracy_pp']['point']:+.1f} [{c['accuracy_pp']['ci_low']:+.1f}, {c['accuracy_pp']['ci_high']:+.1f}] pp", flush=True)
    from transformers import AutoConfig, AutoTokenizer

    repo, rev = DECODERS["smol3"]
    cfg = AutoConfig.from_pretrained(repo, revision=rev)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    per_layer = (cfg.hidden_size * cfg.num_attention_heads * hd + 2 * cfg.hidden_size * cfg.num_key_value_heads * hd
                 + cfg.num_attention_heads * hd * cfg.hidden_size + 3 * cfg.hidden_size * cfg.intermediate_size)
    n_ne = per_layer * cfg.num_hidden_layers
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())["4096:17"]
    pr = Prompter(lambda t: list(tok.encode(t, add_special_tokens=False)), [tok.bos_token_id] if tok.bos_token_id is not None else [], [0], False, 4096)
    fl = statistics.mean(decoder_flops(cfg, n_ne, len(pr.prefix(r["state"])), [len(pr.suffix(q)) for q in r["questions"]]) for r in raw)
    mf = json.loads((RESULTS / "attention_kernels" / "model_flops.json").read_text())
    out["flops_per_contract"] = fl
    out["nonembedding_params"] = n_ne
    out["engine_large_flops_per_contract"] = mf["encoders"]["large"]["engine_flops"]
    out["flop_ratio"] = fl / mf["encoders"]["large"]["engine_flops"]
    print(f"[3b] SmolLM3-3B {fl / 1e12:.2f} TFLOP per contract ({out['flop_ratio']:.1f}x the large engine); KV {summ['kv_bytes_per_token_bf16'] / 1024:.0f} KB/token", flush=True)
    vd = RESULTS / "latency" / f"{H}-decoder-{sel}-contractnli"
    td = RESULTS / "latency" / f"{H}-decoder-{sel}-contractnli-hf-tight"
    cold = {k: v for k, v in {"vllm": min(filter(None, [p50(vd, "decoder-cold-batch"), p50(vd, "decoder-cold-first")]), default=None),
                              "tight": min(filter(None, [p50(td, "decoder-cold-batch"), p50(td, "decoder-cold-first")]), default=None)}.items() if v}
    warm = {k: v for k, v in {"vllm": p50(vd, "decoder-warm"), "tight": p50(td, "decoder-warm")}.items() if v}
    tp = json.loads((vd / "throughput_S4096_K17.json").read_text()) if (vd / "throughput_S4096_K17.json").exists() else {}
    pair = next(p for p in json.loads((RESULTS / "decoder_compare" / "summary.json").read_text())["pairs"] if p["engine"] == "MB-l")
    out["latency_ms"] = {"decoder_cold": cold, "decoder_warm": warm, "engine_cold": pair["engine_cold_ms"], "engine_resident": pair["engine_hit_ms"]}
    # the largest concurrency both sides measured: 16 contracts (the 3B decoder's 64-contract cell did not complete; the
    # engine's 16-contract point is from the re-timed session the main-text figure uses)
    etp = json.loads((RESULTS / "throughput" / f"{H}-contractnli-S4096-K17-retimesession-large-cascade.json").read_text())["engine"]
    out["throughput_contracts_per_s"] = {"N": 16, "decoder": max(tp["16"].values()) if "16" in tp else None, "engine": etp["16"]}
    if cold:
        out["cold_speedup"] = min(cold.values()) / pair["engine_cold_ms"]
    if warm:
        out["resident_speedup"] = min(warm.values()) / pair["engine_hit_ms"]
    if out["throughput_contracts_per_s"]["decoder"]:
        out["throughput_ratio_16"] = etp["16"] / out["throughput_contracts_per_s"]["decoder"]
    print("[3b] latency", out["latency_ms"], "throughput", out["throughput_contracts_per_s"], flush=True)
    write_json(RESULTS / "decoder3b" / "summary.json", out)
