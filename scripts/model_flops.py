"""Model FLOPs per contract and the utilization they imply on the H100 (review: is the engine compute-efficient?).

    .venv/bin/python scripts/model_flops.py
Per contract of the timed request lists (data/cache/requests/contractnli.json, cell 4096:17): the engine's packed pass
(linear layers over every token, attention over the pairs the block mask allows, per layer type), the unpadded baseline
(one full-attention encode per question), and a prefix-cached decoder (the prefix once with causal attention, then each
question's suffix attending to the prefix and causally to itself). Linear FLOPs are 2 x non-embedding parameters x
tokens; attention FLOPs are 4 x query width x allowed pairs per layer. Divided by the measured medians (latency cells,
throughput files) this gives model TFLOP/s. Writes results/attention_kernels/model_flops.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_kernels import model_flops  # noqa: E402
from decoder_serve import Prompter  # noqa: E402

from onepass.bench.timing import TimingCell
from onepass.cli import ENCODERS, REPO, RESULTS, write_json
from onepass.model.encoder import OnePassEncoder, load_tokenizer_adapter
from onepass.model.layout import layout_request
from onepass.schema import Question

H = "x86_64-linux-cuda-bf16"
CELL = "4096:17"


def decoder_flops(cfg, n_params_nonemb: int, prefix: int, suffixes: list[int]) -> float:
    width = cfg.num_attention_heads * (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    lin = 2.0 * n_params_nonemb * (prefix + sum(suffixes))
    pairs = prefix * (prefix + 1) / 2 + sum(s * prefix + s * (s + 1) / 2 for s in suffixes)
    return lin + 4.0 * width * cfg.num_hidden_layers * pairs


def p50(d: str, c: str, p: str) -> float:
    return TimingCell.load(RESULTS / "latency" / d / f"cell_{c}_{p}.json").p50_ms


if __name__ == "__main__":
    raw = json.loads((REPO / "data" / "cache" / "requests" / "contractnli.json").read_text())[CELL]
    reqs = [(r["state"], [Question(q["kind"], q["text"], q["options"], qid=q["qid"]) for q in r["questions"]], r["questions"]) for r in raw]
    out = {"cell": CELL, "n_requests": len(reqs), "encoders": {}, "decoders": {}}
    for size in ("base", "large"):
        enc = OnePassEncoder.from_pretrained(*ENCODERS[size]).eval()
        _, tok = load_tokenizer_adapter(*ENCODERS[size])
        eng = [model_flops(enc, layout_request(tok, s, qs, "block", 4096, 24, state_ids=tok.encode(s)), "block") for s, qs, _ in reqs]
        base = [sum(model_flops(enc, layout_request(tok, s, [q], "full", 4096, 24, state_ids=tok.encode(s)), "full") for q in qs) for s, qs, _ in reqs]
        out["encoders"][size] = {"engine_flops": statistics.mean(eng), "baseline_flops": statistics.mean(base)}
        print(f"[flops] {size}: engine {statistics.mean(eng) / 1e12:.2f} TFLOP/contract, unpadded baseline {statistics.mean(base) / 1e12:.2f}", flush=True)
    from transformers import AutoConfig, AutoTokenizer

    for name, path in (("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"), ("SmolLM2-360M", str(REPO / "models" / "_from_h100_flex" / "decoder-smol360-contractnli-lr1e4"))):
        cfg = AutoConfig.from_pretrained(path)
        dtok = AutoTokenizer.from_pretrained(path)
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        per_layer = (cfg.hidden_size * cfg.num_attention_heads * hd + 2 * cfg.hidden_size * cfg.num_key_value_heads * hd
                     + cfg.num_attention_heads * hd * cfg.hidden_size + 3 * cfg.hidden_size * cfg.intermediate_size)
        n_ne = per_layer * cfg.num_hidden_layers
        pr = Prompter(lambda t: list(dtok.encode(t, add_special_tokens=False)), [dtok.bos_token_id] if dtok.bos_token_id is not None else [], [0], False, 4096)
        fl = [decoder_flops(cfg, n_ne, len(pr.prefix(s)), [len(pr.suffix(q)) for q in qd]) for s, _, qd in reqs]
        out["decoders"][name] = {"flops": statistics.mean(fl), "nonembedding_params": n_ne}
        print(f"[flops] {name}: {statistics.mean(fl) / 1e12:.2f} TFLOP/contract ({n_ne / 1e9:.2f}B non-embedding parameters)", flush=True)
    # utilization from the measured medians
    s11 = f"{H}-{{}}-contractnli-t8-strength-flexsession-stock"
    tp = lambda f: json.loads((RESULTS / "throughput" / f).read_text())
    u = {}
    for size in ("base", "large"):
        e, b = out["encoders"][size]["engine_flops"], out["encoders"][size]["baseline_flops"]
        d = s11.format(size)
        u[f"{size} engine cold, masked"] = e / (p50(d, "S4096_K17", "engine-cold-fill") / 1e3) / 1e12
        u[f"{size} engine cold, split"] = e / (p50(d, "S4096_K17", "engine-cold-fill-split") / 1e3) / 1e12
        u[f"{size} unpadded baseline"] = b / (p50(d, "S4096_K17", "varlen-torch-varlen-batched-full") / 1e3) / 1e12
        t = tp(f"{H}-contractnli-S4096-K17-flexsession-{size}-split.json")["engine"]
        u[f"{size} engine, 64 contracts, split"] = e * t["64"] / 1e12
    for name, ddir in (("SmolLM2-1.7B", f"{H}-decoder-decoder-smol1.7-contractnli-lr3e5-contractnli"), ("SmolLM2-360M", f"{H}-decoder-decoder-smol360-contractnli-lr1e4-contractnli")):
        t = json.loads((RESULTS / "latency" / ddir / "throughput_S4096_K17.json").read_text())
        u[f"{name}, 64 contracts"] = out["decoders"][name]["flops"] * max(t["64"].values()) / 1e12
    out["model_tflops_per_s"] = u
    for k, v in u.items():
        print(f"[flops] {k}: {v:.0f} model TFLOP/s", flush=True)
    write_json(RESULTS / "attention_kernels" / "model_flops.json", out)
