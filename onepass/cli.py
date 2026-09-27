"""Shared CLI plumbing for the experiment scripts: paths, seeds, host tag, model registry, JSON writing."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"
MODELS = REPO / "models"  # gitignored: fine-tuned weights
SEED = 20260927

ENCODERS = {"base": ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
            "large": ("answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13")}
SYSTEM2 = ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit", "1db835463eb0f953019af1048077b90403b70424")
DECODERS = {"smol360": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
            "smol1.7": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674")}
GLINER = ("fastino/GLiNER2.5-Decide", "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6")
TASKS = ("routing", "guardrail", "tracegrade")


def host_tag() -> str:
    """A short, non-identifying host label: chip family + OS, e.g. 'arm64-darwin' or 'x86_64-linux'."""
    return f"{platform.machine().lower()}-{platform.system().lower()}"


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    return p


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def pool_dir(task: str) -> Path:
    return DATA / "pools" / task


def model_dir(size: str, mode: str, task: str) -> Path:
    return MODELS / f"modernbert-{size}-{mode}-{task}"


def env_threads(default: int = 8) -> int:
    return int(os.environ.get("ONEPASS_THREADS", default))
