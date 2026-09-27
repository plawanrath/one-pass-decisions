"""The kernel's floating-point primitives, written once so that NumPy and the generated C compute identical bits.

Every routine here is a transliteration of the C the emitter writes: same algorithm, same constants, same order of
operations, float32 throughout. Nothing calls libm, because a libm result may differ between platforms and would make
bit-exactness untestable. Reductions are sequential (`cumsum` in NumPy, a `for` loop in C) for the same
reason: pairwise summation would not match a scalar loop.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32
MAGIC = F32(12582912.0)  # 1.5 * 2^23: adding and subtracting it rounds a float32 to an integer, half to even
LOG2E = F32(1.4426950408889634)
LN2_HI, LN2_LO = F32(0.693145751953125), F32(1.428606765330187e-06)  # ln 2 split so that k*LN2_HI is exact
EXP_C = [F32(c) for c in (1.0, 1.0, 0.5, 0.16666667, 0.041666668, 0.008333334, 0.0013888889)]
ERF_P = F32(0.3275911)
ERF_A = [F32(c) for c in (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)]
SQRT1_2 = F32(0.70710678118654752)


def round_to_int(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest integer, halves to even, without depending on the current rounding mode."""
    x = x.astype(F32, copy=False)
    return (x + MAGIC) - MAGIC


def expf(x: np.ndarray) -> np.ndarray:
    """exp for float32: 2^k * poly(r), x = k*ln2 + r. Degree-6 Taylor on |r| <= ln2/2 (relative error < 2e-7)."""
    x = x.astype(F32, copy=False)
    k = round_to_int(x * LOG2E)
    r = (x - k * LN2_HI) - k * LN2_LO
    p = EXP_C[6]
    for c in (EXP_C[5], EXP_C[4], EXP_C[3], EXP_C[2], EXP_C[1], EXP_C[0]):
        p = p * r + c
    ki = np.clip(k.astype(np.int32), -126, 127) + 127
    two_k = (ki.astype(np.uint32) << np.uint32(23)).view(F32) if ki.dtype == np.int32 else ki
    out = p * two_k
    return np.where(x < F32(-87.0), F32(0.0), out).astype(F32)


def erff(x: np.ndarray) -> np.ndarray:
    """Abramowitz and Stegun 7.1.26, odd-extended to negative arguments. The formula's own error is 1.5e-7; evaluating
    it in float32 costs a few more ulps, so the kernel's erf is accurate to about 5e-7, far inside the gate's budget."""
    x = x.astype(F32, copy=False)
    s = np.where(x < F32(0.0), F32(-1.0), F32(1.0)).astype(F32)
    a = np.abs(x).astype(F32)
    t = (F32(1.0) / (F32(1.0) + ERF_P * a)).astype(F32)
    poly = ERF_A[4]
    for c in (ERF_A[3], ERF_A[2], ERF_A[1], ERF_A[0]):
        poly = poly * t + c
    return (s * (F32(1.0) - poly * t * expf(-(a * a)))).astype(F32)


def gelu(x: np.ndarray) -> np.ndarray:
    x = x.astype(F32, copy=False)
    return (F32(0.5) * x * (F32(1.0) + erff(x * SQRT1_2))).astype(F32)


LOG_C = [F32(2.0 / (2 * i + 1)) for i in range(6)]  # 2*(s + s^3/3 + ... + s^11/11), s = (m-1)/(m+1)
LN2 = F32(0.6931471805599453)


def logf(x: np.ndarray) -> np.ndarray:
    """log for float32 on positive arguments: split off the exponent, then an odd series in (m-1)/(m+1) on m in [1, 2)."""
    x = x.astype(F32, copy=False)
    bits = x.view(np.uint32)
    e = (bits >> np.uint32(23)).astype(np.int32) - np.int32(127)
    m = ((bits & np.uint32(0x7FFFFF)) | np.uint32(127 << 23)).view(F32)
    big = m > F32(1.4142135623730951)          # keep the series argument small and symmetric
    m = np.where(big, m * F32(0.5), m).astype(F32)
    e = np.where(big, e + np.int32(1), e)
    t = ((m - F32(1.0)) / (m + F32(1.0))).astype(F32)
    t2 = (t * t).astype(F32)
    poly = LOG_C[5]
    for c in (LOG_C[4], LOG_C[3], LOG_C[2], LOG_C[1], LOG_C[0]):
        poly = poly * t2 + c
    return (e.astype(F32) * LN2 + t * poly).astype(F32)


def seq_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Left-to-right summation in float32, the order a scalar C loop uses."""
    return np.cumsum(x.astype(F32, copy=False), axis=axis, dtype=F32).take(-1, axis=axis)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: F32 = F32(1e-5)) -> np.ndarray:
    """mean and biased variance by sequential summation; 1/sqrt is correctly rounded, so it is portable."""
    x = x.astype(F32, copy=False)
    d = F32(x.shape[-1])
    mean = (seq_sum(x)[..., None] / d).astype(F32)
    dx = (x - mean).astype(F32)
    var = (seq_sum(dx * dx)[..., None] / d).astype(F32)
    inv = (F32(1.0) / np.sqrt(var + eps)).astype(F32)
    return (dx * inv * weight + bias).astype(F32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(F32, copy=False)
    m = x.max(axis=axis, keepdims=True)
    e = expf(x - m)
    return (e / seq_sum(e, axis=axis)[..., None]).astype(F32)


def quantize_activation(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 levels and one float32 scale per token. The scale of a token depends on that token only, so
    trailing padding cannot change any result and the kernel may stop at the end of the payload."""
    x = x.astype(F32, copy=False)
    amax = np.abs(x).max(axis=-1, keepdims=True).astype(F32)
    scale = (np.maximum(amax, F32(1e-8)) / F32(127.0)).astype(F32)
    q = np.clip(round_to_int(x / scale), F32(-127.0), F32(127.0))
    return q.astype(np.int32), scale


def dot_groups(xq: np.ndarray, levels: np.ndarray, scales: np.ndarray, group: int) -> np.ndarray:
    """sum over groups of (exact int32 dot product) * (group scale), accumulated in float32 in group order.
    xq (..., in) int32; levels (out, in) int32; scales (out, n_groups) float32."""
    out = np.zeros(xq.shape[:-1] + (levels.shape[0],), dtype=F32)
    for g in range(levels.shape[1] // group):
        sl = slice(g * group, (g + 1) * group)
        acc = np.matmul(xq[..., sl], levels[:, sl].T)  # int32, exact
        out = (out + acc.astype(F32) * scales[:, g]).astype(F32)
    return out


def dot_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Contract the last axis of both arguments by sequential accumulation, broadcasting the leading axes
    (`matmul` would reassociate the sum and no longer match a scalar C loop)."""
    a, b = a.astype(F32, copy=False), b.astype(F32, copy=False)
    out = np.zeros(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), dtype=F32)
    for i in range(a.shape[-1]):
        out = (out + a[..., i] * b[..., i]).astype(F32)
    return out
