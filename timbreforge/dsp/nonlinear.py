"""Controlled nonlinear processing (harmonic generation) with oversampling.

Only mild, *level-compensated* nonlinearity belongs in a character processor.
The rules implemented here:

* Nonlinear stages run oversampled (default 4x) with polyphase FIR
  interpolation, so generated harmonics above Nyquist are filtered out instead
  of folding back as aliasing.
* Compensation is measured after the shaper and is only ever applied as
  attenuation.  A saturation stage therefore cannot turn into a loudness boost.
* A DC blocker follows asymmetric shaping, because asymmetric curves shift the
  waveform's mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .biquad import high_pass
from .meter import integrated_loudness, rms_db


# ---------------------------------------------------------------------------
# shaper curves  (each maps x -> y, monotonic, unity-ish around 0)
# ---------------------------------------------------------------------------


def _shape_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(drive * x)


def _shape_arctan(x: np.ndarray, drive: float) -> np.ndarray:
    return (2.0 / math.pi) * np.arctan(drive * math.pi / 2.0 * x)


def _shape_soft(x: np.ndarray, drive: float) -> np.ndarray:
    """Cubic soft clip: unity slope at 0, saturating towards +/-1.5."""
    t = np.clip(drive * x, -1.5, 1.5)
    return t - (t ** 3) / 3.0 * (2.0 / 3.0) * 1.5 / 1.5


def _shape_asym(x: np.ndarray, drive: float) -> np.ndarray:
    """Asymmetric soft saturation (even + odd harmonics), DC blocked afterwards."""
    pos = np.tanh(drive * x)
    neg = np.tanh(drive * 0.72 * x)
    return np.where(x >= 0, pos, neg)


SHAPERS = {
    "tanh": _shape_tanh,
    "arctan": _shape_arctan,
    "soft": _shape_soft,
    "asym": _shape_asym,
}


@dataclass
class SaturationParams:
    enabled: bool = False
    shaper: str = "tanh"
    drive_db: float = 0.0        # 0 dB -> unity gain at small signal
    mix: float = 0.0             # parallel amount, 0..1
    oversample: int = 4
    compensate: str = "rms"      # "rms" | "lufs" | "none"  (attenuation only)
    dc_block: bool = True

    @staticmethod
    def from_dict(d: dict) -> "SaturationParams":
        p = SaturationParams()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                setattr(p, k, v if isinstance(cur, bool) else type(cur)(v))
        return p

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("enabled", "shaper", "drive_db", "mix", "oversample", "compensate", "dc_block")}

    def validate(self) -> list[str]:
        out = []
        if self.mix < 0 or self.mix > 1:
            out.append("saturation mix must be between 0 and 1")
        if self.drive_db > 18:
            out.append("saturation drive is limited to +18 dB")
        if self.shaper not in SHAPERS:
            out.append(f"unknown shaper '{self.shaper}'")
        return out


def _oversampled_chunk(x: np.ndarray, factor: int, fn) -> np.ndarray:
    """Oversample one chunk, shape it, and come back to the input rate."""
    pad = 512
    n = x.shape[0]
    if n <= pad * 2 + 1:
        head = x[:pad][::-1] if n > pad else np.zeros((0, x.shape[1]))
        tail = x[-pad:][::-1] if n > pad else np.zeros((0, x.shape[1]))
    else:
        head = x[:pad][::-1]
        tail = x[-pad:][::-1]
    ext = np.concatenate([head, x, tail], axis=0)
    up = signal.resample_poly(ext, factor, 1, axis=0, window=("kaiser", 8.6))
    shaped = fn(up)
    del up
    down = signal.resample_poly(shaped, 1, factor, axis=0, window=("kaiser", 8.6))
    del shaped
    # resample_poly preserves length, so the reflected padding returns at the
    # same offset and is simply trimmed away again
    return down[pad:pad + n]


def _oversampled(x: np.ndarray, factor: int, fn, chunk: int = 1 << 17) -> np.ndarray:
    """Apply a per-sample nonlinearity at an oversampled rate, length preserved.

    Chunked so a long file never needs a full oversampled copy in memory.
    """
    factor = int(max(1, factor))
    n = x.shape[0]
    if factor == 1:
        return fn(x)
    out = np.empty_like(x)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        out[start:stop] = _oversampled_chunk(x[start:stop], factor, fn)
    return out


def saturate(x: np.ndarray, sr: int, p: SaturationParams) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not p.enabled or p.mix <= 0.0:
        return x, {"bypassed": True}

    drive = 10 ** (p.drive_db / 20.0)
    fn = SHAPERS.get(p.shaper, _shape_tanh)
    wet = _oversampled(x, p.oversample, lambda v: fn(v, drive))

    if p.dc_block:
        b, a = high_pass(18.0, 0.707, sr)
        wet = signal.lfilter(b, a, wet, axis=0)

    y = x * (1.0 - p.mix) + wet * p.mix

    comp_db = 0.0
    if p.compensate != "none":
        if p.compensate == "lufs":
            base = integrated_loudness(x, sr)
            now = integrated_loudness(y, sr)
        else:
            base = rms_db(x)
            now = rms_db(y)
        comp_db = min(0.0, base - now)  # attenuation only
        if comp_db < 0:
            y = y * (10 ** (comp_db / 20.0))

    return y, {
        "bypassed": False,
        "shaper": p.shaper,
        "drive_db": p.drive_db,
        "mix": p.mix,
        "compensation_db": comp_db,
        "oversample": p.oversample,
    }


# ---------------------------------------------------------------------------
# alias / harmonic measurement helper (analysis side)
# ---------------------------------------------------------------------------


def harmonic_profile_db(x: np.ndarray, sr: int, f0: float = 1000.0) -> dict[str, float]:
    """Rough 60 Hz harmonic-series ratio of a test tone; for the dev tool only."""
    n = x.shape[0]
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x[:, 0] * w))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    total = float(np.sum(spec ** 2))
    if total <= 0:
        return {}
    out = {}
    for h in range(1, 8):
        f = f0 * h
        if f >= sr / 2:
            break
        band = (freqs >= f * 0.98) & (freqs <= f * 1.02)
        out[f"h{h}"] = float(10 * math.log10(max(np.sum(spec[band] ** 2), 1e-18) / total))
    return out
