"""Mid/side width control with a mono-compatibility guard.

Width is expressed as a change to the side signal only, so the mid (centre) image
and the mono sum are untouched by construction:

    M = (L + R) / 2      S = (L - R) / 2
    S' = S * k           -> mono sum (L+R)/2 == M is bit-identical

Polarity/phase rotation is deliberately *not* offered: rotating phase between
channels decorrelates the image but produces a hollow, phasey result when the
mix is folded to mono, and the analysis needs the stereo question answered
without damaging compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meter import channel_correlation, mono_fold_change, side_mid_db


@dataclass
class StereoParams:
    enabled: bool = False
    width: float = 1.0           # 1.0 = unchanged, 0.0 = mono, 2.0 = double side
    mono_below_hz: float = 0.0   # optional: keep low end centred (0 = off)
    # "width" only -- no phase rotation, no independent channel processing

    @staticmethod
    def from_dict(d: dict) -> "StereoParams":
        p = StereoParams()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                setattr(p, k, v if isinstance(cur, bool) else type(cur)(v))
        return p

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "width": self.width,
                "mono_below_hz": self.mono_below_hz}

    def validate(self) -> list[str]:
        out = []
        if not (0.0 <= self.width <= 2.0):
            out.append("stereo width must be between 0.0 and 2.0")
        if self.mono_below_hz and not (20 <= self.mono_below_hz <= 400):
            out.append("mono-below frequency must be between 20 and 400 Hz")
        return out


def _ms(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (block[:, 0] + block[:, 1]), 0.5 * (block[:, 0] - block[:, 1])


def _lr(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.stack([m + s, m - s], axis=1)


def set_width(x: np.ndarray, p: StereoParams, sr: int) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not p.enabled or x.shape[1] < 2 or abs(p.width - 1.0) < 1e-4:
        return x, {"bypassed": True}
    m, s = _ms(x)
    if p.mono_below_hz > 0:
        from .biquad import low_pass
        b, a = low_pass(p.mono_below_hz, 0.707, sr)
        import scipy.signal as sig
        low = sig.lfilter(b, a, s)
        s = (s - low) + low * min(p.width, 1.0)
    else:
        s = s * p.width
    y = _lr(m, s)
    rep = {
        "bypassed": False,
        "width": p.width,
        "correlation_before": channel_correlation(x),
        "correlation_after": channel_correlation(y),
        "side_mid_before_db": side_mid_db(x),
        "side_mid_after_db": side_mid_db(y),
        "mono_fold_before_db": mono_fold_change(x),
        "mono_fold_after_db": mono_fold_change(y),
    }
    return y, rep
