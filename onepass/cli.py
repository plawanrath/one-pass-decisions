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
            "large": ("answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"),
            # the second encoder family: sizes are named eurobert-<size> in model and result directories
            "eurobert-base": ("EuroBERT/EuroBERT-210m", "39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e"),
            "eurobert-large": ("EuroBERT/EuroBERT-610m", "d9af784ed20db6c2096e335ec6a67dd4a219924c"),
            "eurobert-xl": ("EuroBERT/EuroBERT-2.1B", "81245a4d71f43452badf5e04458e4ddb831ff109")}
STATE_SIDE = {"tracegrade": "tail"}  # the step under review ends an agent transcript; other tasks keep the head


STATE_TOKENS = {"contractnli": 4096,  # whole contracts (90th percentile about 4,200 tokens); other tasks fit in 1,024
                "cuad": 8190,  # one encoder window ([CLS] + 8,190 + [SEP] = 8,192 positions); the windowed arm reads more
                "ecthr": 4096,  # case facts: median about 1,300 tokens, 90th percentile about 5,100
                "quality": 8190}  # whole stories in one window (median about 6,800 tokens, 90th percentile about 7,900)
WINDOW_TOKENS = 8192  # layout tokens per window of a windowed state (the encoders' pretrained position range)
WINDOWED_STATE_TOKENS = {"cuad": 4 * 8192}  # four windows: first evidence inside for 98% of CUAD's positive answers


def state_side(task: str) -> str:
    return STATE_SIDE.get(task, "head")


def state_tokens(task: str, mode: str = "block") -> int:
    """The state budget a task is trained, scored and served with (in mode "windowed": layout tokens, whole windows)."""
    if mode == "windowed":
        return WINDOWED_STATE_TOKENS.get(task, WINDOW_TOKENS)
    return STATE_TOKENS.get(task, 1024)


SYSTEM2 = ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-8bit", "1db835463eb0f953019af1048077b90403b70424")
DECODERS = {"smol360": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
            "smol1.7": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"),
            "smol3": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1")}  # the stronger decoder baseline
GLINER = ("fastino/GLiNER2.5-Decide", "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6")
TASKS = ("routing", "guardrail", "tracegrade", "contractnli", "cuad", "ecthr", "quality")


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
    """models/modernbert-<size>-<mode>-<task> for ModernBERT; models/eurobert-<size>-<mode>-<task> for EuroBERT."""
    return MODELS / (f"{size}-{mode}-{task}" if size.startswith("eurobert-") else f"modernbert-{size}-{mode}-{task}")


def env_threads(default: int = 8) -> int:
    return int(os.environ.get("ONEPASS_THREADS", default))


def for_mode(records, task: str, mode: str):
    """Records as a model trained under `mode` sees them: in chunked mode the trace-grading states are re-rendered as
    append-only chunks (task, then one chunk per step) from the captured trajectories; other tasks are unchanged (a
    guardrail system prompt is one chunk)."""
    if mode != "chunked" or task != "tracegrade":
        return records
    from .tasks.tracegrade import as_chunked
    from .traces.harness import load_trajectories

    return as_chunked(records, load_trajectories(RESULTS / "traces" / "agent"))

