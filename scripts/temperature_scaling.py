"""Does temperature scaling fix the mask-trained arm's confidence? One temperature per run, fitted on the contracts dev
pool by minimizing the negative log-likelihood of the gold option, applied to the test pool; calibration (ECE with
fifteen bins, mean confidence, share of errors at confidence >= 0.9) and ranking (AURC, Kendall tau between the
confidence orders before and after) are reported before and after.

    .venv/bin/python scripts/temperature_scaling.py [--threads 16] [--skip-rescore]
Mask-trained runs whose checkpoints are on this host are re-scored on dev and test (fp32, CPU, block mask, the
evaluation batching of onepass/train/finetune.py) so the option logits keep full precision: the stored score files round
probabilities to 1e-6, which leaves a saturated arm a few hundred distinct confidences. The decisions are checked
against the stored test scores (written on the H100 in bf16). The full-attention and one-question-per-sequence arms are
scaled from their stored probabilities (log probabilities as logits; they have few ties). Writes
results/temperature_scaling/summary.json (and the re-scored logits under results/temperature_scaling/logits/).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import kendalltau

from onepass.cli import REPO, RESULTS, pool_dir, write_json
from onepass.metrics import aurc, ece
from onepass.model.encoder import OnePassEncoder
from onepass.model.layout import TokenizerAdapter
from onepass.tasks.pools import load_pool
from onepass.train.finetune import TrainConfig, build_sequences, collate, make_batches

FT = RESULTS / "finetune"
OUT = RESULTS / "temperature_scaling"
# (result name, checkpoint) of mask-trained runs; runs whose checkpoint is missing are skipped
MASK_TRAINED = [
    ("base-block-contractnli-steps-s1", "models/_from_h100_round16/modernbert-base-block-contractnli-steps-s1"),
    ("base-block-contractnli-steps-s2", "models/_from_h100_round16/modernbert-base-block-contractnli-steps-s2"),
    ("large-block-contractnli-e30", "models/_from_h100/modernbert-large-block-contractnli-e30"),
    ("large-block-contractnli-e30-s1r", "models/_from_h100_large_seeds/modernbert-large-block-contractnli-e30-s1r"),
    ("large-block-contractnli-e30-s2r", "models/_from_h100_large_seeds/modernbert-large-block-contractnli-e30-s2r"),
]
STORED = {
    "ModernBERT-base full": ["base-full-contractnli-lr2e5-bestacc", "base-full-contractnli-lr2e5-bestacc-s1", "base-full-contractnli-lr2e5-bestacc-s2"],
    "ModernBERT-base unpacked": ["base-block-contractnli-unpacked", "base-block-contractnli-unpacked-s1", "base-block-contractnli-unpacked-s2"],
    "ModernBERT-base mask-trained (stored)": ["base-block-contractnli-steps", "base-block-contractnli-steps-s1", "base-block-contractnli-steps-s2"],
    "ModernBERT-large full": ["large-full-contractnli", "large-full-contractnli-s1r", "large-full-contractnli-s2r"],
}


def rescore(model: OnePassEncoder, tok: TokenizerAdapter, records, cfg: TrainConfig) -> list[dict]:
    """evaluate() of onepass/train/finetune.py, keeping each question's raw option logits (fp32)."""
    model.decouple_layers = cfg.decouple_layers
    model.eval()
    seqs = build_sequences(records, tok, cfg, cfg.mode)
    rows: dict[int, dict] = {}
    for batch in make_batches(seqs, cfg.tokens_per_batch, None, 256):
        pb, _ = collate(batch, tok.pad_id, cfg.mode, cfg.pad_multiple)
        with torch.no_grad():
            logits = model(pb).float()
        for b, s in enumerate(batch):
            for k, (ri, g) in enumerate(zip(s.record_idx, s.gold)):
                n = len(records[ri].question.options)
                z = logits[b, k, :n]
                rows[ri] = {"idx": ri, "gold": int(g), "logits": [float(x) for x in z], "group_id": records[ri].group_id}
    return [rows[i] for i in range(len(records))]


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = z / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def pad_logits(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(n, Nmax) logits with -inf in absent option slots, and the gold indices."""
    n = max(len(r["logits"]) for r in rows)
    Z = np.full((len(rows), n), -np.inf)
    for i, r in enumerate(rows):
        Z[i, : len(r["logits"])] = r["logits"]
    return Z, np.array([r["gold"] for r in rows])


def nll(Z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(Z, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None))))


def fit_temperature(Z: np.ndarray, y: np.ndarray) -> float:
    """Grid over log T, then a finer grid around the best point."""
    grid = np.linspace(np.log(0.05), np.log(50.0), 401)
    best = grid[int(np.argmin([nll(Z, y, np.exp(g)) for g in grid]))]
    fine = np.linspace(best - 0.02, best + 0.02, 81)
    return float(np.exp(fine[int(np.argmin([nll(Z, y, np.exp(g)) for g in fine]))]))


def stats(Z: np.ndarray, y: np.ndarray, T: float) -> dict:
    p = softmax(Z, T)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    wrong = (pred != y).astype(int)
    return {"accuracy": float(1 - wrong.mean()), "ece": float(ece(conf, wrong)), "mean_conf": float(conf.mean()),
            "errors_at_conf_ge_0.9": float(((conf >= 0.9) & (wrong == 1)).sum() / max(1, wrong.sum())),
            "aurc": float(aurc(conf, wrong)), "distinct_conf": int(len(np.unique(conf))), "_conf": conf}


def analyse(name: str, Zd, yd, Zt, yt, source: str) -> dict:
    T = fit_temperature(Zd, yd)
    before, after = stats(Zt, yt, 1.0), stats(Zt, yt, T)
    tau = float(kendalltau(before.pop("_conf"), after.pop("_conf")).statistic)
    out = {"run": name, "source": source, "temperature": T, "dev_nll_before": nll(Zd, yd, 1.0), "dev_nll_after": nll(Zd, yd, T),
           "test_before": before, "test_after": after, "kendall_tau_before_after": tau}
    print(f"[temp] {name:38s} T={T:6.3f} acc {100 * before['accuracy']:.1f}  ECE {100 * before['ece']:.1f} -> {100 * after['ece']:.1f}  "
          f"err>=.9 {100 * before['errors_at_conf_ge_0.9']:.0f} -> {100 * after['errors_at_conf_ge_0.9']:.0f}  "
          f"AURC {1000 * before['aurc']:.1f} -> {1000 * after['aurc']:.1f}  tau {tau:.3f}  distinct {before['distinct_conf']} -> {after['distinct_conf']}", flush=True)
    return out


def stored(run: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (FT / run / f"scores_{split}.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["idx"])
    return pad_logits([{"gold": r["gold"], "logits": list(np.log(np.clip(np.array(r["probs"], dtype=float), 1e-12, None)))} for r in rows])


def group_mean(items: list[dict]) -> dict:
    keys = ("accuracy", "ece", "mean_conf", "errors_at_conf_ge_0.9", "aurc")
    return {"n_runs": len(items), "temperature": float(np.mean([x["temperature"] for x in items])),
            "kendall_tau": float(np.mean([x["kendall_tau_before_after"] for x in items])),
            **{f"{w}_{k}": float(np.mean([x[f"test_{w}"][k] for x in items])) for w in ("before", "after") for k in keys}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip-rescore", action="store_true", help="reuse logits already written under results/temperature_scaling/logits/")
    ap.add_argument("--only", nargs="*", default=[], help="re-score only these mask-trained runs")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pools = {s: load_pool(pool_dir("contractnli"), s) for s in ("dev", "test")}
    out = {"task": "contractnli", "runs": [], "groups": {}}
    rescored: dict[str, list[dict]] = {"ModernBERT-base mask-trained (re-scored)": [], "ModernBERT-large mask-trained (re-scored)": []}
    for run, ck in MASK_TRAINED:
        if a.only and run not in a.only:
            continue
        lf = OUT / "logits" / f"{run}.json"
        if not (REPO / ck).exists() and not lf.exists():
            print(f"[temp] {run}: no checkpoint on this host, skipped", flush=True)
            continue
        if lf.exists() and a.skip_rescore:
            L = json.loads(lf.read_text())
        else:
            conf = json.loads((FT / run / "summary.json").read_text())["train"]["config"]
            cfg = TrainConfig(**{k: v for k, v in conf.items() if k in TrainConfig.__dataclass_fields__})
            cfg.device, cfg.autocast_bf16 = "cpu", False
            model = OnePassEncoder.load(REPO / ck).float()
            tok = TokenizerAdapter(OnePassEncoder.load_tokenizer(REPO / ck))
            L = {s: rescore(model, tok, pools[s], cfg) for s in ("dev", "test")}
            write_json(lf, L)
        ref = sorted((json.loads(x) for x in (FT / run / "scores_test.jsonl").read_text().splitlines() if x.strip()), key=lambda r: r["idx"])
        same = sum(int(int(np.argmax(r["logits"])) == q["pred"]) for r, q in zip(L["test"], ref))
        Zd, yd = pad_logits(L["dev"])
        Zt, yt = pad_logits(L["test"])
        res = analyse(run, Zd, yd, Zt, yt, "re-scored fp32 logits")
        res["agreement_with_stored_test"] = {"same_decisions": same, "n": len(ref)}
        print(f"[temp] {run}: {same}/{len(ref)} test decisions equal the stored bf16 scores", flush=True)
        out["runs"].append(res)
        rescored["ModernBERT-base mask-trained (re-scored)" if run.startswith("base") else "ModernBERT-large mask-trained (re-scored)"].append(res)
    for g, items in rescored.items():
        if items:
            out["groups"][g] = group_mean(items)
    for g, runs in STORED.items():
        items = []
        for run in runs:
            if not (FT / run / "scores_test.jsonl").exists():
                continue
            Zd, yd = stored(run, "dev")
            Zt, yt = stored(run, "test")
            items.append(analyse(run, Zd, yd, Zt, yt, "stored probabilities (rounded to 1e-6)"))
        out["runs"] += items
        if items:
            out["groups"][g] = group_mean(items)
    print("[temp] group means (test):", flush=True)
    for g, v in out["groups"].items():
        print(f"[temp]   {g:42s} T {v['temperature']:.2f}  ECE {100 * v['before_ece']:.1f} -> {100 * v['after_ece']:.1f}  "
              f"err>=.9 {100 * v['before_errors_at_conf_ge_0.9']:.0f} -> {100 * v['after_errors_at_conf_ge_0.9']:.0f}  "
              f"AURC {1000 * v['before_aurc']:.1f} -> {1000 * v['after_aurc']:.1f}  tau {v['kendall_tau']:.3f}", flush=True)
    write_json(OUT / "summary.json", out)
