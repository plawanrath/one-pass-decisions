"""The engine against a prefix-cached decoder on document review (ADR-0006, E11): accuracy, latency, cost, cache bytes.

    .venv/bin/python scripts/decoder_compare.py
Accuracy: the selected SmolLM2-1.7B fine-tune (results/decoder_finetune/selected_<task>.txt) against the masked
encoders on the same test questions (paired bootstrap, 10,000 resamples). Latency (H100, bf16, same session): the
decoder served by vLLM with prefix caching (cold: the faster of submitting all K prompts at once or the first alone;
warm: the contract's prefix resident) against the engine (cold with the fused cache fill; resident) and the fastest
encoder baseline. Dollars per million decisions at $3.49/h. Writes results/decoder_compare/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from onepass.analysis import paired_compare, read_scores
from onepass.bench.timing import TimingCell, compare
from onepass.cli import RESULTS, write_json

USD_H = 3.49
HOST = "x86_64-linux-cuda-bf16"
ENCODER_ARMS = {"contractnli": [("MB-b block, equal steps", "base-block-contractnli-steps"), ("MB-l block, equal steps", "large-block-contractnli-e30"),
                                ("MB-l full", "large-full-contractnli"),
                                ("MB-b block, best checkpoint", "base-block-contractnli-bestacc")],
                "cuad": [("MB-b block, one window", "base-block-cuad-steps"), ("MB-b windowed, four windows", "base-windowed-cuad-steps"),
                         ("MB-b full, one window", "base-full-cuad")]}
# (engine label, latency directory, encoder arm whose accuracy that engine serves, cell-name suffix); the large engine
# was timed in a later session (stock weights) with the base engine re-timed there as the anchor; the split-kernel
# engines in a third session, beside SDPA re-timed as the anchor
ENGINES = {"contractnli": [("MB-b", f"{HOST}-base-contractnli-t8-strength-fused", "MB-b block, equal steps", ""),
                           ("MB-l", f"{HOST}-large-contractnli-t8-strength-fused-stock", "MB-l block, equal steps", ""),
                           ("MB-b split", f"{HOST}-base-contractnli-t8-strength-flexsession-stock", "MB-b block, equal steps", "-split"),
                           ("MB-l split", f"{HOST}-large-contractnli-t8-strength-flexsession-stock", "MB-l block, equal steps", "-split")],
           "cuad": [("MB-b", f"{HOST}-base-cuad-t8-strength-fused-stock", "MB-b block, one window", ""),
                    ("MB-b split", f"{HOST}-base-cuad-t8-strength-flexsession-stock", "MB-b block, one window", "-split")]}
LARGE_TP = f"{HOST}-contractnli-S4096-K17-fused-large.json"
# engine throughput (contracts/s by concurrency) per engine label, from results/throughput
ENGINE_TP = {"contractnli": {"MB-b": f"{HOST}-contractnli-S4096-K17-flexsession-base.json", "MB-l": f"{HOST}-contractnli-S4096-K17-flexsession-large.json",
                             "MB-b split": f"{HOST}-contractnli-S4096-K17-flexsession-base-split.json",
                             "MB-l split": f"{HOST}-contractnli-S4096-K17-flexsession-large-split.json"},
             "cuad": {"MB-b": f"{HOST}-cuad-S8192-K41-fused.json", "MB-b split": f"{HOST}-cuad-S8192-K41-flexsession-split.json"}}
BASELINES = ("torch-one-seq-full", "torch-compile-one-seq-full", "varlen-torch-varlen-one-seq-full", "varlen-torch-varlen-batched-full", "torch-naive-batch-full")


def cell(d: Path, S: int, K: int, path: str) -> TimingCell | None:
    f = d / f"cell_S{S}_K{K}_{path}.json"
    return TimingCell.load(f) if f.exists() else None


def usd_per_million(ms: float, K: int) -> float:
    return USD_H * ms / K / 3.6e6 * 1e6


# matched-accuracy pairs (split-kernel engine against the decoder whose accuracy it matches): task, engine label, the
# engine's latency directory and cell, the encoder arm, which decoder ("large": SmolLM2-1.7B, "small": SmolLM2-360M),
# the decoder's cell, the engine's throughput file, the encoder's cache key
_S11 = f"{HOST}-{{}}-t8-strength-flexsession{{}}-stock"
_S15 = f"{HOST}-{{}}-t8-strength-retimesession-stock"  # engine re-timed after the host-overhead reductions
_AK = "x86_64-linux-cuda-{}-contractnli-S4096-K17-N64-profile-cascade-retimesession.json"  # the cascade engine at 64 contracts
# matched-accuracy pairs (the engine against the decoder whose accuracy it matches): task, engine label, the engine's
# latency directory and cell, windowed prefix, engine cell suffix (cascade kernel; the split kernel for the windowed
# engine, to which the cascade kernel falls back), the encoder arm, which decoder ("large": SmolLM2-1.7B, "small":
# SmolLM2-360M), the decoder's cell, the engine's throughput source, the encoder's cache key
PAIRS = (("contractnli", "MB-l", _S15.format("large-contractnli"), "S4096_K17", "", "-cascade", "MB-l block, equal steps", "large", "S4096_K17",
          ("profile", _AK.format("large")), "large"),
         ("contractnli", "MB-b", _S15.format("base-contractnli"), "S4096_K17", "", "-cascade", "MB-b block, best checkpoint", "small", "S4096_K17",
          ("profile", _AK.format("base")), "base"),
         ("cuad", "MB-b 1w", _S15.format("base-cuad"), "S8192_K41", "", "-cascade", "MB-b block, one window", "small", "S8192_K41",
          ("throughput", f"{HOST}-cuad-S8192-K41-retimesession-cascade.json"), "base"),
         ("cuad", "MB-b 4w", f"{HOST}-base-cuad-t8-strength-retimesession-windowed-stock", "S32768_K41", "windowed-", "-split", "MB-b windowed, four windows", "large", "S8192_K41",
          None, "base"))


# the arm actually served: the full-attention fine-tune scored under the block mask (served-masked); the four-window
# engine has no full-attention counterpart that reads four windows, so its mask-trained windowed arm is the served arm
SERVED = {("contractnli", "MB-l"): "large-full-contractnli", ("contractnli", "MB-b"): "base-full-contractnli-lr2e5-bestacc",
          ("cuad", "MB-b 1w"): "base-full-cuad"}


def pairs(out: dict) -> list[dict]:
    st = json.loads((RESULTS / "cache_study" / "x86_64-linux-cuda-bf16" / "summary.json").read_text())
    rows = []
    for task, eng, edir, ecell, wpre, sfx, arm, which, dcell, tp, ekey in PAIRS:
        t = out["tasks"].get(task)
        if not t:
            continue
        dec = t if which == "large" else t.get("small_decoder")
        if not dec or arm not in t["accuracy"] or arm not in dec["accuracy"]:
            continue
        ddir = RESULTS / "latency" / f"{HOST}-decoder-{dec['decoder']}-{task}"
        ld = lambda d, c, p: TimingCell.load(RESULTS / "latency" / d / f"cell_{c}_{p}.json") if (RESULTS / "latency" / d / f"cell_{c}_{p}.json").exists() else None
        ec = ld(edir, ecell, f"engine-{wpre}cold{'' if wpre else '-fill'}{sfx}")
        eh = ld(edir, ecell, f"engine-{wpre}hit{sfx}")
        # the decoder's better serving path: vLLM, or the tight HF path (prefix cached once, suffixes batched against it;
        # the 1.7B model was timed there with its pretrained weights, since latency does not depend on them)
        tight = RESULTS / "latency" / (f"{HOST}-decoder-{dec['decoder']}-{task}-hf-tight" if which == "small" else f"{HOST}-decoder-SmolLM2-1.7B-Instruct-{task}-hf-tight")
        dirs = [("vllm", ddir), ("tight", tight)]
        dcs = [(b, TimingCell.load(d / f"cell_{dcell}_decoder-cold-{m}.json")) for b, d in dirs for m in ("batch", "first") if (d / f"cell_{dcell}_decoder-cold-{m}.json").exists()]
        dws = [(b, TimingCell.load(d / f"cell_{dcell}_decoder-warm.json")) for b, d in dirs if (d / f"cell_{dcell}_decoder-warm.json").exists()]
        if not ec or not eh or not dcs or not dws:
            continue
        (cold_path, dc), (warm_path, dw) = min(dcs, key=lambda x: x[1].p50_ms), min(dws, key=lambda x: x[1].p50_ms)
        K = int(ecell.split("_K")[1])
        dtp = next(iter(dec.get("throughput", {}).values()), None)
        etp = None
        if tp and tp[0] == "throughput" and (RESULTS / "throughput" / tp[1]).exists():
            etp = json.loads((RESULTS / "throughput" / tp[1]).read_text())["engine"]
        elif tp and tp[0] == "profile" and (RESULTS / "attention_kernels" / tp[1]).exists():
            etp = {"64": json.loads((RESULTS / "attention_kernels" / tp[1]).read_text())["budgets"]["65536"]["contracts_per_s"]}
        top = lambda d: (max(d, key=int), d[max(d, key=int)]) if d else (None, None)
        n_e, v_e = top(etp)
        n_d, v_d = top({k: max(v.values()) for k, v in dtp.items()} if dtp else None)
        rows.append({"task": task, "engine": eng, "encoder_arm": arm, "decoder": dec["decoder"], "engine_cell": ecell, "decoder_cell": dcell,
                     "encoder_accuracy": t["accuracy"][arm]["encoder_accuracy"], "decoder_accuracy": dec["decoder_test"]["accuracy"],
                     "encoder_minus_decoder_pp": dec["accuracy"][arm]["encoder_minus_decoder_pp"],
                     "decoder_cold_path": cold_path, "decoder_warm_path": warm_path,
                     "engine_cold_ms": ec.p50_ms, "decoder_cold_ms": dc.p50_ms, "engine_hit_ms": eh.p50_ms, "decoder_warm_ms": dw.p50_ms,
                     "cold": compare(ec, dc), "resident": compare(eh, dw),
                     "usd_per_million_cold": {"engine": usd_per_million(ec.p50_ms, K), "decoder": usd_per_million(dc.p50_ms, K)},
                     "kb_per_token": {"engine": st["analytic"]["encoders"][ekey]["trimmed_bytes_per_token_bf16"] / 1024,
                                      "decoder": dec["decoder_kv_bytes_per_token"] / 1024},
                     "throughput": {"engine": v_e, "engine_N": n_e, "decoder": v_d, "decoder_N": n_d}})
        r = rows[-1]
        sv = SERVED.get((task, eng))
        f = RESULTS / "finetune" / sv / "scores_test_as_block.jsonl" if sv else None
        if f is not None and f.exists():
            dsc = read_scores(RESULTS / "decoder_finetune" / dec["decoder"] / "scores_test.jsonl")
            ssc = read_scores(f)
            q, c = paired_compare(ssc, dsc), paired_compare(ssc, dsc, cluster=True)
            r["served_masked"] = {"run": sv, "accuracy": float(sum(x["pred"] == x["gold"] for x in ssc) / len(ssc)),
                                  "minus_decoder_pp": q["accuracy_pp"], "minus_decoder_pp_cluster": c["accuracy_pp"],
                                  "minus_decoder_aurc": q["aurc"]}
            print(f"[decoder-compare]   served-masked {sv}: minus decoder {q['accuracy_pp']['point']:+.2f} pp "
                  f"[{q['accuracy_pp']['ci_low']:+.2f}, {q['accuracy_pp']['ci_high']:+.2f}] (contract-clustered [{c['accuracy_pp']['ci_low']:+.2f}, {c['accuracy_pp']['ci_high']:+.2f}])", flush=True)
        print(f"[decoder-compare] pair {task} {eng} vs {dec['decoder']}: acc {100 * r['encoder_accuracy']:.1f}/{100 * r['decoder_accuracy']:.1f}; cold "
              f"{ec.p50_ms:.1f}/{dc.p50_ms:.1f} ({r['cold']['speedup']:.1f}x); resident {eh.p50_ms:.1f}/{dw.p50_ms:.1f} ({r['resident']['speedup']:.1f}x); "
              f"throughput {v_e}/{v_d}", flush=True)
    return rows


def small_decoder(task: str, enc_acc: dict) -> dict | None:
    """The selected SmolLM2-360M fine-tune: accuracy against the same encoder arms, and its serving latency and
    throughput (timed in the split-kernel session, one cell)."""
    sel = RESULTS / "decoder_finetune" / f"selected_smol360_{task}.txt"
    if not sel.exists():
        return None
    name = sel.read_text().strip()
    dsum = json.loads((RESULTS / "decoder_finetune" / name / "summary.json").read_text())
    dec = read_scores(RESULTS / "decoder_finetune" / name / "scores_test.jsonl")
    r = {"decoder": name, "decoder_test": dsum["splits"]["test"], "decoder_kv_bytes_per_token": dsum["kv_bytes_per_token_bf16"], "accuracy": {}, "latency": {}}
    for label, arm in ENCODER_ARMS[task]:
        f = RESULTS / "finetune" / arm / "scores_test.jsonl"
        if f.exists():
            cmp = paired_compare(read_scores(f), dec)
            r["accuracy"][label] = {"encoder_minus_decoder_pp": cmp["accuracy_pp"], "encoder_minus_decoder_aurc": cmp["aurc"]}
    ddir = RESULTS / "latency" / f"{HOST}-decoder-{name}-{task}"
    for p in sorted(ddir.glob("cell_S*_K*_decoder-*.json")):
        c = TimingCell.load(p)
        r["latency"][p.stem.split("_", 3)[3]] = {"cell": p.stem.split("_decoder")[0][5:], "p50_ms": c.p50_ms}
    for f in ddir.glob("throughput_S*_K*.json"):
        r.setdefault("throughput", {})[f.stem] = json.loads(f.read_text())
    meta = ddir / "meta.json"
    if meta.exists():
        r["serving_meta"] = json.loads(meta.read_text())
    print(f"[decoder-compare] {task} small decoder {name}: test {100 * r['decoder_test']['accuracy']:.1f}; "
          + ", ".join(f"{k} {v['p50_ms']:.1f} ms" for k, v in r["latency"].items()), flush=True)
    return r


if __name__ == "__main__":
    out = {"tasks": {}}
    for task in ("contractnli", "cuad"):
        sel = RESULTS / "decoder_finetune" / f"selected_{task}.txt"
        if not sel.exists():
            print(f"[decoder-compare] {task}: no selected decoder yet")
            continue
        name = sel.read_text().strip()
        dsum = json.loads((RESULTS / "decoder_finetune" / name / "summary.json").read_text())
        dec = read_scores(RESULTS / "decoder_finetune" / name / "scores_test.jsonl")
        t = {"decoder": name, "decoder_test": dsum["splits"]["test"], "decoder_kv_bytes_per_token": dsum["kv_bytes_per_token_bf16"],
             "accuracy": {}, "latency": {}}
        for label, arm in ENCODER_ARMS[task]:
            f = RESULTS / "finetune" / arm / "scores_test.jsonl"
            if f.exists():
                cmp = paired_compare(read_scores(f), dec)  # encoder minus decoder
                t["accuracy"][label] = {"encoder_accuracy": json.loads((f.parent / "summary.json").read_text())["splits"]["test"]["accuracy"],
                                        "encoder_minus_decoder_pp": cmp["accuracy_pp"], "encoder_minus_decoder_aurc": cmp["aurc"]}
        ddir = RESULTS / "latency" / f"{HOST}-decoder-{name}-{task}"
        cells = sorted({(int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])) for p in ddir.glob("cell_S*_K*_decoder-warm.json")})
        for eng_label, edir_name, arm, sfx in ENGINES[task]:
            edir = RESULTS / "latency" / edir_name
            for S, K in cells:
                dc = {k: cell(ddir, S, K, f"decoder-{k}") for k in ("cold-batch", "cold-first", "warm")}
                ec = {k: cell(edir, S, K, k + sfx) for k in ("engine-cold-fill", "engine-cold", "engine-hit")}
                bc = {b: cell(edir, S, K, b) for b in BASELINES}
                bc = {k: v for k, v in bc.items() if v}
                if not dc["warm"] or not ec["engine-cold-fill"]:
                    continue
                dcold = min((v for k, v in dc.items() if k.startswith("cold") and v), key=lambda c: c.p50_ms)
                fastest = min(bc.values(), key=lambda c: c.p50_ms) if bc else None
                row = {"engine": eng_label, "accuracy_arm": arm, "decoder_cold_ms": dcold.p50_ms, "decoder_cold_mode": dcold.path,
                       "decoder_warm_ms": dc["warm"].p50_ms, "engine_cold_fill_ms": ec["engine-cold-fill"].p50_ms,
                       "engine_hit_ms": ec["engine-hit"].p50_ms if ec["engine-hit"] else None,
                       "fastest_baseline": fastest.path if fastest else None, "fastest_baseline_ms": fastest.p50_ms if fastest else None,
                       "engine_vs_decoder_cold": compare(ec["engine-cold-fill"], dcold),
                       "engine_vs_decoder_warm": compare(ec["engine-hit"], dc["warm"]) if ec["engine-hit"] else None,
                       "usd_per_million": {"decoder_cold": usd_per_million(dcold.p50_ms, K), "decoder_warm": usd_per_million(dc["warm"].p50_ms, K),
                                           "engine_cold": usd_per_million(ec["engine-cold-fill"].p50_ms, K),
                                           "engine_hit": usd_per_million(ec["engine-hit"].p50_ms, K) if ec["engine-hit"] else None}}
                unf = cell(edir, S, K, "engine-cold-fill-unfused")
                if unf and not sfx:
                    row["fusion_saving"] = compare(ec["engine-cold-fill"], unf)
                t["latency"][f"{eng_label}:{S}:{K}"] = row
                print(f"[decoder-compare] {task} {eng_label} S={S} K={K}: decoder cold {dcold.p50_ms:.1f} / warm {dc['warm'].p50_ms:.1f} ms; engine cold+fill "
                      f"{ec['engine-cold-fill'].p50_ms:.1f} / hit {row['engine_hit_ms']:.1f} ms", flush=True)
        lt = RESULTS / "throughput" / LARGE_TP
        if task == "contractnli" and lt.exists():
            t["throughput_large_engine"] = json.loads(lt.read_text())
        for f in ddir.glob("throughput_S*_K*.json"):
            t.setdefault("throughput", {})[f.stem] = json.loads(f.read_text())
        t["engine_throughput"] = {k: json.loads((RESULTS / "throughput" / f).read_text())["engine"]
                                  for k, f in ENGINE_TP[task].items() if (RESULTS / "throughput" / f).exists()}
        t["small_decoder"] = small_decoder(task, t["accuracy"])
        meta = ddir / "meta.json"
        if meta.exists():
            t["serving_meta"] = json.loads(meta.read_text())
        out["tasks"][task] = t
        for label, v in t["accuracy"].items():
            print(f"[decoder-compare] {task} {label}: encoder {100 * v['encoder_accuracy']:.1f} vs decoder {100 * t['decoder_test']['accuracy']:.1f} "
                  f"(diff {v['encoder_minus_decoder_pp']['point']:+.1f} pp)", flush=True)
    out["pairs"] = pairs(out)
    write_json(RESULTS / "decoder_compare" / "summary.json", out)
