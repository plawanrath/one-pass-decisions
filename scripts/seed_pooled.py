"""One price per encoder: the served-masked, mask-trained and adapted arms against full attention, pooled over every
run of each arm, with a hierarchical bootstrap that resamples runs and documents together (10,000 resamples). Also the
calibration of each arm (ECE with fifteen bins, mean confidence minus accuracy, share of errors made at confidence 0.9
or above), since AURC alone measures ranking, not confidence.

    .venv/bin/python scripts/seed_pooled.py
Reads results/finetune/<run>/scores_test{,_as_block}.jsonl; writes results/seed_pooled/summary.json and the
per-run confidence histograms the paper's calibration figure draws (results/seed_pooled/histograms.json).
A run is one fine-tune; a seed retrained in a later session is a separate run (run-to-run variance is part of the price).
"""
from __future__ import annotations

import json

import numpy as np

from onepass.cli import RESULTS, write_json
from onepass.metrics import ece

FT = RESULTS / "finetune"
B = 10_000
# (task, encoder) -> arm -> runs; "served-masked" reads the full runs' scores_test_as_block.jsonl
GROUPS = {
    ("contracts", "ModernBERT-base"): {
        "full": ["base-full-contractnli-lr2e5-bestacc", "base-full-contractnli-lr2e5-bestacc-s1", "base-full-contractnli-lr2e5-bestacc-s2",
                 "base-full-contractnli-lr2e5-bestacc-s1r", "base-full-contractnli-lr2e5-bestacc-s2r"],
        "mask-trained": ["base-block-contractnli-steps", "base-block-contractnli-steps-s1", "base-block-contractnli-steps-s2"],
        # adapted seeds 1 and 2 start from the retrained full seeds s1r and s2r (whose weights are kept)
        "adapted": ["base-block-contractnli-adapt2", "base-block-contractnli-adapt2-s1", "base-block-contractnli-adapt2-s2"],
        # the control for packing: the block mask with one question per sequence, the full arm's recipe
        "unpacked": ["base-block-contractnli-unpacked", "base-block-contractnli-unpacked-s1", "base-block-contractnli-unpacked-s2"],
        # the mask-trained arm under the full arm's exact protocol: 16 questions per sequence at the full arm's 5,951 steps,
        # lr 2e-5, best checkpoint on dev accuracy (the symmetric protocol; round 19)
        "mask-trained-symmetric": ["base-block-contractnli-pack16-steps", "base-block-contractnli-pack16-steps-s1", "base-block-contractnli-pack16-steps-s2"],
        # the mask arm at its own learning rate (3e-5) with the best of ten checkpoints on dev accuracy
        "mask-trained-own-lr": ["base-block-contractnli-bestacc", "base-block-contractnli-bestacc-s1", "base-block-contractnli-bestacc-s2"]},
    ("contracts", "ModernBERT-large"): {
        "full": ["large-full-contractnli", "large-full-contractnli-s1", "large-full-contractnli-s2", "large-full-contractnli-s1r", "large-full-contractnli-s2r"],
        "mask-trained": ["large-block-contractnli-e30", "large-block-contractnli-e30-s1r", "large-block-contractnli-e30-s2r",
                         "large-block-contractnli-e30-s3r", "large-block-contractnli-e30-s4r"],
        "adapted": ["large-block-contractnli-adapt2", "large-block-contractnli-adapt2-s1", "large-block-contractnli-adapt2-s2"]},
    ("contracts", "EuroBERT-610M"): {
        "full": ["eurobert-large-full-contractnli-lr1e5", "eurobert-large-full-contractnli-lr1e5-s1", "eurobert-large-full-contractnli-lr1e5-s2",
                 "eurobert-large-full-contractnli-lr1e5-r", "eurobert-large-full-contractnli-lr1e5-s1r", "eurobert-large-full-contractnli-lr1e5-s2r"],
        "mask-trained": ["eurobert-large-block-contractnli-steps", "eurobert-large-block-contractnli-steps-s1", "eurobert-large-block-contractnli-steps-s2"],
        "adapted": ["eurobert-large-block-contractnli-adapt2", "eurobert-large-block-contractnli-adapt2-s1", "eurobert-large-block-contractnli-adapt2-s2"]},
    ("clauses", "ModernBERT-base"): {
        "full": ["base-full-cuad", "base-full-cuad-s1", "base-full-cuad-s2", "base-full-cuad-s2r"],
        "mask-trained": ["base-block-cuad-steps", "base-block-cuad-steps-s1"],
        # adapted seed 1 starts from full seed 1, seed 2 from the retrained full seed s2r
        "adapted": ["base-block-cuad-adapt2", "base-block-cuad-adapt2-s1", "base-block-cuad-adapt2-s2"]},
    ("stories", "ModernBERT-base"): {
        "full": ["base-full-quality", "base-full-quality-s1", "base-full-quality-s2"],
        "mask-trained": ["base-block-quality-steps", "base-block-quality-steps-s1", "base-block-quality-steps-s2"],
        "adapted": ["base-block-quality-adapt2", "base-block-quality-adapt2-s1", "base-block-quality-adapt2-s2"]},
    ("stories", "ModernBERT-large"): {
        "full": ["large-full-quality", "large-full-quality-s1", "large-full-quality-s2"],
        "adapted": ["large-block-quality-adapt2", "large-block-quality-adapt2-s1", "large-block-quality-adapt2-s2"]},
    # last, so the groups above keep their bootstrap draws (one shared generator)
    # matched size against SmolLM3-3B: seed 0 at lr 1e-5 and seed 1 at lr 5e-6 (round 20), adapted from each; seed 1 at
    # 1e-5 did not converge (73.1%) and is excluded, as registered before the 5e-6 run; seed 2 at 5e-6 is included
    # whatever its accuracy, as registered before it ran
    ("contracts", "EuroBERT-2.1B"): {
        "full": ["eurobert-xl-full-contractnli-lr1e5", "eurobert-xl-full-contractnli-lr5e6-s1", "eurobert-xl-full-contractnli-lr5e6-s2"],
        "adapted": ["eurobert-xl-block-contractnli-adapt2", "eurobert-xl-block-contractnli-adapt2-lr5e6-s1", "eurobert-xl-block-contractnli-adapt2-lr5e6-s2"]}
}


def load(run: str, as_block: bool = False) -> list[dict] | None:
    f = FT / run / ("scores_test_as_block.jsonl" if as_block else "scores_test.jsonl")
    if not f.exists():
        return None
    return sorted((json.loads(line) for line in f.read_text().splitlines() if line.strip()), key=lambda r: r["idx"])


def arms_of(spec: dict) -> dict[str, list[list[dict]]]:
    out = {"full": [r for r in (load(x) for x in spec["full"]) if r],
           "served-masked": [r for r in (load(x, True) for x in spec["full"]) if r]}
    for arm in ("mask-trained", "adapted", "unpacked", "mask-trained-symmetric", "mask-trained-own-lr"):
        out[arm] = [r for r in (load(x) for x in spec.get(arm, [])) if r]
    return out


def aurc_rep(conf: np.ndarray, wrong: np.ndarray, reps: np.ndarray) -> float:
    """AURC of the items repeated `reps` times (a bootstrap resample as integer weights)."""
    keep = reps > 0
    c, w, n = conf[keep], wrong[keep], reps[keep]
    o = np.argsort(-c, kind="stable")
    w, n = w[o], n[o]
    # each item contributes n points along the coverage axis, all at the same cumulative risk step
    idx = np.repeat(np.arange(len(w)), n)
    cw = np.cumsum(w[idx])
    return float(np.mean(cw / np.arange(1, len(idx) + 1)))


def calib(rows: list[dict]) -> dict:
    conf = np.array([r["conf"] for r in rows])
    wrong = np.array([int(r["pred"] != r["gold"]) for r in rows])
    return {"accuracy": float(1 - wrong.mean()), "mean_conf": float(conf.mean()), "overconfidence": float(conf.mean() - (1 - wrong.mean())),
            "ece": float(ece(conf, wrong)), "errors_at_conf_ge_0.9": float(((conf >= 0.9) & (wrong == 1)).sum() / max(1, wrong.sum())),
            "n": len(rows)}


def pooled(arms: dict, a: str, b: str, rng: np.random.Generator) -> dict | None:
    """Mean over runs of arm a minus mean over runs of arm b, resampling runs (each arm separately; the served-masked
    arm uses the same run draws as the full arm, since they are the same weights) and documents (shared)."""
    if not arms[a] or not arms[b]:
        return None
    ref = arms[b][0]
    groups = sorted({r["group_id"] for r in ref})
    gi = {g: i for i, g in enumerate(groups)}
    item_g = np.array([gi[r["group_id"]] for r in ref])
    G = len(groups)

    def mats(runs):
        C = np.array([[int(r["pred"] == r["gold"]) for r in rows] for rows in runs])
        P = np.array([[r["conf"] for r in rows] for rows in runs])
        return C, P

    Ca, Pa = mats(arms[a])
    Cb, Pb = mats(arms[b])
    paired = {a, b} == {"full", "served-masked"}
    stats = {"accuracy_pp": [], "aurc": [], "ece": [], "overconfidence": []}

    def metrics(C, P, reps):
        out = {"accuracy_pp": [], "aurc": [], "ece": [], "overconfidence": []}
        for c, p in zip(C, P):
            acc = (c * reps).sum() / reps.sum()
            out["accuracy_pp"].append(100 * acc)
            out["aurc"].append(aurc_rep(p, 1 - c, reps))
            idx = np.repeat(np.arange(len(c)), reps)
            out["ece"].append(ece(p[idx], 1 - c[idx]))
            out["overconfidence"].append((p * reps).sum() / reps.sum() - acc)
        return out

    ones = np.ones(len(item_g), dtype=int)
    pa, pb = metrics(Ca, Pa, ones), metrics(Cb, Pb, ones)
    point = {k: float(np.mean(pa[k]) - np.mean(pb[k])) for k in stats}
    for _ in range(B):
        greps = rng.multinomial(G, np.full(G, 1 / G))
        reps = greps[item_g]
        ra = rng.integers(0, len(Ca), len(Ca))
        rb = ra if paired and len(Ca) == len(Cb) else rng.integers(0, len(Cb), len(Cb))
        ma, mb = metrics(Ca[ra], Pa[ra], reps), metrics(Cb[rb], Pb[rb], reps)
        for k in stats:
            stats[k].append(np.mean(ma[k]) - np.mean(mb[k]))
    return {"n_runs": [len(Ca), len(Cb)], "n_documents": G, "n_items": len(item_g),
            **{k: {"point": point[k], "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in stats.items()}}


def histograms(rows: list[dict]) -> dict:
    edges = np.linspace(0, 1, 21)
    conf = np.array([r["conf"] for r in rows])
    wrong = np.array([r["pred"] != r["gold"] for r in rows])
    return {"edges": edges.tolist(), "correct": np.histogram(conf[~wrong], edges)[0].tolist(), "wrong": np.histogram(conf[wrong], edges)[0].tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out, hist = {"resamples": B, "groups": []}, {}
    for (task, enc), spec in GROUPS.items():
        arms = arms_of(spec)
        if not arms["full"]:
            print(f"[pooled] {task} {enc}: no full runs, skipped")
            continue
        g = {"task": task, "encoder": enc, "runs": spec, "calibration": {}, "vs_full": {}, "served_vs_mask_trained": None}
        for arm, runs in arms.items():
            if runs:
                cs = [calib(r) for r in runs]
                g["calibration"][arm] = {k: float(np.mean([c[k] for c in cs])) for k in cs[0]} | {"per_run": cs}
                hist[f"{task}|{enc}|{arm}"] = [histograms(r) for r in runs]
        for arm in ("served-masked", "mask-trained", "adapted", "unpacked"):
            g["vs_full"][arm] = pooled(arms, arm, "full", rng)
        # an arm added later draws from its own generator, so the arms above keep their recorded intervals
        g["vs_full"]["mask-trained-symmetric"] = pooled(arms, "mask-trained-symmetric", "full", np.random.default_rng(1))
        g["vs_full"]["mask-trained-own-lr"] = pooled(arms, "mask-trained-own-lr", "full", np.random.default_rng(2))
        g["served_vs_mask_trained"] = pooled(arms, "served-masked", "mask-trained", rng)
        g["unpacked_vs_mask_trained"] = pooled(arms, "unpacked", "mask-trained", rng)
        out["groups"].append(g)
        for arm, v in g["vs_full"].items():
            if v:
                print(f"[pooled] {task} {enc} {arm} - full: acc {v['accuracy_pp']['point']:+.2f} [{v['accuracy_pp']['ci'][0]:+.2f}, {v['accuracy_pp']['ci'][1]:+.2f}]"
                      f" aurc {1000 * v['aurc']['point']:+.1f} ece {100 * v['ece']['point']:+.1f} overconf {100 * v['overconfidence']['point']:+.1f} (runs {v['n_runs']})", flush=True)
    write_json(RESULTS / "seed_pooled" / "summary.json", out)
    write_json(RESULTS / "seed_pooled" / "histograms.json", hist)
