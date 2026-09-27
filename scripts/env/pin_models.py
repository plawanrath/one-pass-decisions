"""Resolve and download pinned model revisions into the HF cache; write models.lock.txt.

Usage: python scripts/env/pin_models.py [--from-lock] [--revision REPO@SHA ...] [--no-download]
Without --from-lock, the current main revision of each repo in models.txt is resolved and pinned.
With --from-lock (new machine), exactly the revisions already in models.lock.txt are downloaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
LOCK = HERE / "models.lock.txt"
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"]


def _lines(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", action="append", default=[], help="REPO@SHA override")
    ap.add_argument("--from-lock", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    pinned = dict(l.split("@", 1) for l in _lines(LOCK)) if args.from_lock else {}
    pinned.update(dict(r.split("@", 1) for r in args.revision))
    repos = list(pinned) if args.from_lock else _lines(HERE / "models.txt")
    api = HfApi()
    out = ["# Pinned upstream model revisions (HF commit hashes). Regenerate with scripts/env/pin_models.py.",
           "# Format: <hf_repo>@<revision>"]
    for repo in repos:
        try:
            sha = pinned.get(repo) or api.model_info(repo).sha
            if not args.no_download:
                snapshot_download(repo, revision=sha, allow_patterns=PATTERNS)
            out.append(f"{repo}@{sha}")
            print(f"[pin] {repo}@{sha}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            out.append(f"# {repo}  UNRESOLVED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            print(f"[pin] FAILED {repo}: {e}", file=sys.stderr)
    LOCK.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
