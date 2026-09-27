"""Locked pools: deterministic, group-aware splits with a manifest (name, n, seed, sha256) written once.

A pool directory holds `{train,dev,cal,test}.jsonl` plus `POOL.json`. Splits are made on `group_id` (a trajectory or
conversation), so no group spans two splits. `lock_pools` refuses to overwrite an existing manifest: locked means
locked; delete the directory on purpose to regenerate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..schema import DecisionRecord, file_sha256, read_jsonl, write_jsonl

SPLITS = ("train", "dev", "cal", "test")


@dataclass
class PoolSpec:
    task: str
    seed: int
    n_dev: int
    n_cal: int
    n_test: int
    n_train: int | None = None  # None: everything left over
    source: str = ""  # corpus id and revision
    notes: str = ""


class PoolLockedError(RuntimeError):
    pass


def _group_split(records: list[DecisionRecord], spec: PoolSpec) -> dict[str, list[DecisionRecord]]:
    rng = np.random.default_rng(spec.seed)
    groups: dict[str, list[DecisionRecord]] = {}
    for r in records:
        groups.setdefault(r.group_id or r.state_hash, []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    out = {s: [] for s in SPLITS}
    want = {"dev": spec.n_dev, "cal": spec.n_cal, "test": spec.n_test}
    for k in keys:
        placed = False
        for s in ("test", "cal", "dev"):  # fill the evaluation splits first so their sizes are exact-ish
            if len(out[s]) < want[s]:
                out[s].extend(groups[k])
                placed = True
                break
        if not placed:
            if spec.n_train is None or len(out["train"]) < spec.n_train:
                out["train"].extend(groups[k])
    return out


def lock_pools(records: list[DecisionRecord], spec: PoolSpec, out_dir: str | Path, presplit: dict[str, list[DecisionRecord]] | None = None) -> dict:
    """Write the four splits and the manifest. `presplit` lets a corpus with an official test split keep it."""
    out = Path(out_dir)
    if (out / "POOL.json").exists():
        raise PoolLockedError(f"{out / 'POOL.json'} exists; pools are locked")
    splits = presplit if presplit is not None else _group_split(records, spec)
    for s in SPLITS:
        splits.setdefault(s, [])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"spec": asdict(spec), "splits": {}}
    for s in SPLITS:
        p = out / f"{s}.jsonl"
        n = write_jsonl(p, splits[s])
        manifest["splits"][s] = {"n": n, "sha256": file_sha256(p), "n_groups": len({r.group_id or r.state_hash for r in splits[s]})}
    # cross-split group leakage check
    seen: dict[str, str] = {}
    for s in SPLITS:
        for r in splits[s]:
            g = r.group_id or r.state_hash
            if g in seen and seen[g] != s:
                raise RuntimeError(f"group {g} appears in {seen[g]} and {s}")
            seen[g] = s
    manifest["kinds"] = sorted({r.question.kind for s in SPLITS for r in splits[s]})
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest["splits"], sort_keys=True).encode()).hexdigest()
    (out / "POOL.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def pool_manifest(pool_dir: str | Path) -> dict:
    return json.loads((Path(pool_dir) / "POOL.json").read_text())


def load_pool(pool_dir: str | Path, split: str, verify: bool = True) -> list[DecisionRecord]:
    d = Path(pool_dir)
    p = d / f"{split}.jsonl"
    if verify:
        m = pool_manifest(d)
        if file_sha256(p) != m["splits"][split]["sha256"]:
            raise RuntimeError(f"{p} does not match its locked sha256")
    return list(read_jsonl(p))
