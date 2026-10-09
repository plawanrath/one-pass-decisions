"""Build and lock the decision pools for the routing, guardrail and contract-review tasks from public corpora.

    .venv/bin/python scripts/lock_pools.py --task routing
    .venv/bin/python scripts/lock_pools.py --task guardrail
    .venv/bin/python scripts/lock_pools.py --task contractnli   # after unpacking contract-nli.zip into data/raw/contractnli
    .venv/bin/python scripts/lock_pools.py --task cuad          # after unpacking CUAD_v1.zip into data/raw/cuad
    .venv/bin/python scripts/lock_pools.py --task ecthr         # downloads LexGLUE ECtHR A and B at the pinned revision
    .venv/bin/python scripts/lock_pools.py --task quality       # after unpacking QuALITY.v1.0.1.zip into data/raw/quality
    .venv/bin/python scripts/lock_pools.py --task race          # downloads RACE at the pinned revision (intermediate training only)

Routing keeps the corpus's official train/validation/test splits (validation -> dev + cal, test -> test) so its
numbers are comparable with the literature. Guardrail pools are group-split over turns from two corpora. Trace-grading
pools are locked by scripts/silver_labels.py once trajectories exist. Locked pools are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from onepass.cli import DATA, SEED, pool_dir, write_json
from onepass.tasks import contractnli, cuad, ecthr, guardrail, quality, race, routing
from onepass.tasks.pools import PoolSpec, lock_pools


def routing_pools(layout: str, relock: bool = False) -> None:
    """dev = the official validation split; cal and test = stratified halves (by gold option and out-of-scope flag) of
    the official test split. The official validation split has 3.8% out-of-scope utterances against 18.2% in the
    official test split, so a threshold set on it does not transfer: the calibration pool must be exchangeable
    with the pool it is judged on. The train split is untouched, so fine-tunes are re-scored, not retrained."""
    out = pool_dir("routing" if layout == "domain" else "routing150")
    if relock and (out / "POOL.json").exists():
        superseded = out.with_name(out.name + "_superseded_" + datetime.now().strftime("%Y%m%d%H%M"))
        out.rename(superseded)
        print("superseded pools moved to", superseded)
    rows_tr, names = routing.load_rows("train")
    rows_va, _ = routing.load_rows("validation")
    rows_te, _ = routing.load_rows("test")
    train = routing.records_from_rows(rows_tr, names, "train", layout)
    dev = routing.records_from_rows(rows_va, names, "validation", layout)
    test_all = routing.records_from_rows(rows_te, names, "test", layout)
    rng = np.random.default_rng(SEED)
    cal, test = [], []
    strata: dict[tuple, list] = {}
    for r in test_all:
        strata.setdefault((r.gold, bool(r.meta.get("oos"))), []).append(r)
    for key in sorted(strata, key=str):
        members = strata[key]
        idx = rng.permutation(len(members))
        half = len(members) // 2
        cal += [members[i] for i in idx[:half]]
        test += [members[i] for i in idx[half:]]
    spec = PoolSpec(out.name, SEED, len(dev), len(cal), len(test), len(train), f"{routing.DATASET[0]}:{routing.DATASET[1]}",
                    f"train and validation official (validation is dev); official test split into cal/test halves stratified by "
                    f"gold option and out-of-scope flag with seed {SEED}, so that cal and test are exchangeable; layout={layout}")
    m = lock_pools([], spec, out, presplit={"train": train, "dev": dev, "cal": cal, "test": test})
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def contractnli_pools() -> None:
    """Official train and dev; the official test contracts split by document into cal and test halves (seeded), so
    the calibration and test splits are exchangeable."""
    out = pool_dir("contractnli")
    raw = DATA / "raw" / "contractnli" / "contract-nli"
    train, dev, test_all = (contractnli.load_split(raw, s) for s in ("train", "dev", "test"))
    groups = sorted({r.group_id for r in test_all})
    rng = np.random.default_rng(SEED)
    cal_groups = {groups[i] for i in rng.permutation(len(groups))[: len(groups) // 2]}
    cal = [r for r in test_all if r.group_id in cal_groups]
    test = [r for r in test_all if r.group_id not in cal_groups]
    spec = PoolSpec("contractnli", SEED, len(dev), len(cal), len(test), len(train), f"{contractnli.URL} sha256:{contractnli.ZIP_SHA256}",
                    f"train and dev official; official test split by contract into cal/test halves with seed {SEED}; 17 hypotheses per contract")
    m = lock_pools([], spec, out, presplit={"train": train, "dev": dev, "cal": cal, "test": test})
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def cuad_pools() -> None:
    """Contracts split 70/10/10/10 into train/dev/cal/test with the seed; all 41 categories of a contract stay together."""
    out = pool_dir("cuad")
    recs = cuad.load_all(DATA / "raw" / "cuad" / "CUAD_v1")
    groups = sorted({r.group_id for r in recs})
    perm = np.random.default_rng(SEED).permutation(len(groups))
    n = len(groups)
    cut = [int(round(f * n)) for f in (0.7, 0.8, 0.9)]
    part = {}
    for i, gi in enumerate(perm):
        part[groups[gi]] = "train" if i < cut[0] else "dev" if i < cut[1] else "cal" if i < cut[2] else "test"
    split = {k: [r for r in recs if part[r.group_id] == k] for k in ("train", "dev", "cal", "test")}
    spec = PoolSpec("cuad", SEED, len(split["dev"]), len(split["cal"]), len(split["test"]), len(split["train"]),
                    f"{cuad.URL} sha256:{cuad.ZIP_SHA256}",
                    f"{n} contracts split by contract 70/10/10/10 with seed {SEED}; 41 categories per contract; gold yes when CUAD marks a span")
    m = lock_pools([], spec, out, presplit=split)
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def ecthr_pools() -> None:
    """Official train and validation (dev); the official test cases split by case into cal and test halves (seeded),
    so the calibration and test splits are exchangeable."""
    out = pool_dir("ecthr")
    train, dev, test_all = (ecthr.load_split(s) for s in ("train", "validation", "test"))
    groups = sorted({r.group_id for r in test_all})
    rng = np.random.default_rng(SEED)
    cal_groups = {groups[i] for i in rng.permutation(len(groups))[: len(groups) // 2]}
    cal = [r for r in test_all if r.group_id in cal_groups]
    test = [r for r in test_all if r.group_id not in cal_groups]
    spec = PoolSpec("ecthr", SEED, len(dev), len(cal), len(test), len(train), f"{ecthr.DATASET[0]}@{ecthr.DATASET[1]} ecthr_a+ecthr_b",
                    f"train and validation official (validation is dev); official test split by case into cal/test halves with seed {SEED}; "
                    f"10 articles per case, three-way (not alleged / alleged, no violation / violation)")
    m = lock_pools([], spec, out, presplit={"train": train, "dev": dev, "cal": cal, "test": test})
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def quality_pools() -> None:
    """The official test split has no public labels. The official train articles split by article into train and dev
    (dev: 15 of 150 articles, seeded); the official dev articles split by article into cal and test halves (seeded)."""
    out = pool_dir("quality")
    raw = DATA / "raw" / "quality"
    train_all, dev_all = quality.load_split(raw, "train"), quality.load_split(raw, "dev")
    rng = np.random.default_rng(SEED)
    tg = sorted({r.group_id for r in train_all})
    dev_groups = {tg[i] for i in rng.permutation(len(tg))[:15]}
    dg = sorted({r.group_id for r in dev_all})
    cal_groups = {dg[i] for i in rng.permutation(len(dg))[: len(dg) // 2]}
    split = {"train": [r for r in train_all if r.group_id not in dev_groups], "dev": [r for r in train_all if r.group_id in dev_groups],
             "cal": [r for r in dev_all if r.group_id in cal_groups], "test": [r for r in dev_all if r.group_id not in cal_groups]}
    spec = PoolSpec("quality", SEED, len(split["dev"]), len(split["cal"]), len(split["test"]), len(split["train"]), f"{quality.URL} sha256:{quality.ZIP_SHA256}",
                    f"official train split by article into train and dev (15 articles); official dev split by article into cal/test halves with seed {SEED}; "
                    f"every question of an article (both writers) is one request")
    m = lock_pools([], spec, out, presplit=split)
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def race_pools() -> None:
    """Intermediate training only: the official train split; official validation as dev; no calibration or test use."""
    out = pool_dir("race")
    train, dev = race.load_split("train"), race.load_split("validation")
    spec = PoolSpec("race", SEED, len(dev), 0, 0, len(train), f"{race.DATASET[0]}@{race.DATASET[1]} all",
                    "official train and validation (dev); intermediate training before QuALITY, never scored as a result")
    m = lock_pools([], spec, out, presplit={"train": train, "dev": dev, "cal": [], "test": []})
    print(out, {k: v["n"] for k, v in m["splits"].items()})


def guardrail_pools() -> None:
    out = pool_dir("guardrail")
    recs = guardrail.load_all()
    n_groups = len({r.group_id for r in recs})
    # sizes in records: dev 8%, cal 22%, test 30%, train the rest; groups are system prompts (SPML) or single turns
    spec = PoolSpec("guardrail", SEED, int(0.08 * len(recs)), int(0.22 * len(recs)), int(0.30 * len(recs)), None,
                    " + ".join(guardrail.CORPORA.values()), f"{n_groups} groups; SPML grouped by system prompt (unseen policies at test)")
    m = lock_pools(recs, spec, out)
    print(out, {k: v["n"] for k, v in m["splits"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["routing", "routing150", "guardrail", "contractnli", "cuad", "ecthr", "quality", "race"])
    ap.add_argument("--relock", action="store_true", help="routing only: supersede an existing lock (records the reason in the manifest notes)")
    a = ap.parse_args()
    if a.task == "routing":
        routing_pools("domain", a.relock)
    elif a.task == "contractnli":
        contractnli_pools()
    elif a.task == "cuad":
        cuad_pools()
    elif a.task == "ecthr":
        ecthr_pools()
    elif a.task == "quality":
        quality_pools()
    elif a.task == "race":
        race_pools()
    elif a.task == "routing150":
        routing_pools("intent150", a.relock)
    else:
        guardrail_pools()
