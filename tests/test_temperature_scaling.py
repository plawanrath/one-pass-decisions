"""The temperature fit of scripts/temperature_scaling.py recovers a known temperature on synthetic logits."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("temperature_scaling", Path(__file__).resolve().parent.parent / "scripts" / "temperature_scaling.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def test_fit_recovers_known_temperature():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20000, 3)) * 2.0  # well-calibrated logits at T = 1
    p = ts.softmax(Z)
    y = np.array([rng.choice(3, p=row) for row in p])
    T = ts.fit_temperature(Z * 3.0, y)  # logits inflated 3x: the fit should divide by about 3
    assert abs(T - 3.0) < 0.15


def test_scaling_keeps_decisions_and_binary_order():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(500, 2))
    y = rng.integers(0, 2, 500)
    a, b = ts.stats(Z, y, 1.0), ts.stats(Z, y, 2.5)
    assert a["accuracy"] == b["accuracy"]
    assert abs(a["aurc"] - b["aurc"]) < 1e-12  # two options: max-probability order is invariant to one temperature
