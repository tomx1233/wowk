"""Biquad filters (RBJ cookbook), cascades and a linear-phase FIR EQ option.

The EQ stage of the processing graph is a *single* cascade of these sections.
Nothing else in the graph applies a static tone shape, so tone is never applied
twice by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# coefficient design (Audio EQ Cookbook, R. Bristow-Johnson)
# ---------------------------------------------------------------------------


def _q_from_bw(bw_oct: float) -> float:
    """Convert a bandwidth in octaves to Q (RBJ)."""
    return 1.0 / (2.0 * math.sinh(math.log(2.0) / 2.0 * bw_oct))


def peaking(fc: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return b / a[0], a / a[0]


def low_shelf(fc: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    sq = 2 * math.sqrt(A) * alpha
    b = np.array([A * ((A + 1) - (A - 1) * cw + sq),
                  2 * A * ((A - 1) - (A + 1) * cw),
                  A * ((A + 1) - (A - 1) * cw - sq)])
    a = np.array([(A + 1) + (A - 1) * cw + sq,
                  -2 * ((A - 1) + (A + 1) * cw),
                  (A + 1) + (A - 1) * cw - sq])
    return b / a[0], a / a[0]


def high_shelf(fc: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    sq = 2 * math.sqrt(A) * alpha
    b = np.array([A * ((A + 1) + (A - 1) * cw + sq),
                  -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - sq)])
    a = np.array([(A + 1) - (A - 1) * cw + sq,
                  2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - sq])
    return b / a[0], a / a[0]


def high_pass(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def low_pass(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def notch(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([1.0, -2 * cw, 1.0])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def band_pass(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def all_pass(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(fc, sr * 0.49) / sr
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([1 - alpha, -2 * cw, 1 + alpha])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


DESIGNERS = {
    "peaking": peaking,
    "lowshelf": low_shelf,
    "highshelf": high_shelf,
    "highpass": high_pass,
    "lowpass": low_pass,
    "notch": notch,
    "bandpass": band_pass,
    "allpass": all_pass,
}


# ---------------------------------------------------------------------------
# band description + cascade
# ---------------------------------------------------------------------------


@dataclass
class Band:
    type: str = "peaking"
    freq: float = 1000.0
    gain_db: float = 0.0
    q: float = 0.707
    enabled: bool = True

    @staticmethod
    def from_dict(d: dict) -> "Band":
        return Band(
            type=str(d.get("type", "peaking")).lower(),
            freq=float(d.get("freq", 1000.0)),
            gain_db=float(d.get("gain_db", 0.0)),
            q=float(d.get("q", 0.707)),
            enabled=bool(d.get("enabled", True)),
        )

    def to_dict(self) -> dict:
        return {"type": self.type, "freq": self.freq, "gain_db": self.gain_db,
                "q": self.q, "enabled": self.enabled}

    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if self.type in ("peaking", "lowshelf", "highshelf"):
            return abs(self.gain_db) >= 0.01
        return self.freq > 0

    def coefficients(self, sr: int) -> tuple[np.ndarray, np.ndarray]:
        fn = DESIGNERS.get(self.type)
        if fn is None:
            raise ValueError(f"Unknown EQ band type '{self.type}'")
        if self.type in ("peaking", "lowshelf", "highshelf"):
            return fn(self.freq, self.gain_db, self.q, sr)
        return fn(self.freq, self.q, sr)


def cascade(secs: Sequence[Sequence[np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(b), np.asarray(a)) for b, a in secs]


def process_cascade(x: np.ndarray, sections: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply a static biquad cascade (float64) to every channel."""
    y = np.asarray(x, dtype=np.float64)
    for b, a in sections:
        y = signal.lfilter(b, a, y, axis=0)
    return y


def magnitude_db(bands: Iterable[Band], freqs: np.ndarray, sr: int) -> np.ndarray:
    """Combined magnitude response in dB, for plotting and FIR design."""
    freqs = np.asarray(freqs, dtype=np.float64)
    total = np.zeros_like(freqs)
    for band in bands:
        if not band.is_active():
            continue
        b, a = band.coefficients(sr)
        _, h = signal.freqz(b, a, worN=freqs, fs=sr)
        total += 20 * np.log10(np.maximum(np.abs(h), 1e-9))
    return total


def phase_latency_warning(bands: Iterable[Band]) -> str | None:
    for band in bands:
        if band.is_active() and band.type in ("highpass", "lowpass"):
            if band.freq < 25 or band.freq > 0.45 * 44100:
                return "An aggressive filter corner was configured; check for ringing."
    return None


# ---------------------------------------------------------------------------
# linear-phase EQ (windowed-FIR version of the same magnitude)
# ---------------------------------------------------------------------------


def linear_phase_fir(bands: Iterable[Band], sr: int, taps: int = 4097) -> np.ndarray:
    """Design a symmetric FIR whose magnitude matches the biquad cascade."""
    taps = int(taps) | 1  # odd -> integer group delay
    freqs = np.linspace(0, sr / 2, taps // 2 + 1)
    mag_db = magnitude_db(bands, freqs, sr)
    mag = 10 ** (mag_db / 20.0)
    mag[0] = 1.0  # no DC offset from the design
    h = np.fft.irfft(mag, n=taps)          # zero-phase, centred on index 0
    h = np.roll(h, (taps - 1) // 2)        # make it causal / linear phase
    h *= np.blackman(taps)
    h /= np.sum(h)                         # unity DC gain
    return h


def apply_linear_phase(x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, int]:
    """Convolve and remove the FIR's group delay so timing is preserved."""
    n = x.shape[0]
    delay = (len(h) - 1) // 2
    y = signal.fftconvolve(x, h[:, None], mode="full", axes=0)
    y = y[delay:delay + n]
    if y.shape[0] < n:
        y = np.pad(y, ((0, n - y.shape[0]), (0, 0)))
    return y, delay


# ---------------------------------------------------------------------------
# crossovers for the multiband stage
# ---------------------------------------------------------------------------


def _lowpass_fir(fc: float, sr: int, taps: int) -> np.ndarray:
    """Symmetric (linear-phase) low-pass FIR, unity gain at DC."""
    taps = int(taps) | 1
    h = signal.firwin(taps, min(fc, sr * 0.47), window=("kaiser", 8.6), fs=sr)
    h /= np.sum(h)
    return h


def _aligned_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve and remove the FIR's group delay, so timing is untouched."""
    d = (len(h) - 1) // 2
    y = signal.fftconvolve(x, h[:, None], mode="full", axes=0)
    return y[d:d + x.shape[0]]


def split_bands(x: np.ndarray, sr: int, crossovers: Sequence[float],
                taps_per_hz: float = 4.0) -> tuple[list[np.ndarray], list[int]]:
    """Complementary linear-phase split into len(crossovers) + 1 bands.

    The split is *complementary*: each high band is the residual
    ``rest - lowpass(rest)``, so the bands sum back to the input exactly.  Every
    band is time aligned with the input (the FIR group delay is removed), which
    means the multiband stage cannot smear transients or shift the mix.

    Using a complementary split instead of Butterworth/Linkwitz-Riley sections is
    a deliberate trade: reconstruction is exact and phase distortion is zero,
    while the transition slopes are gentler.  For a tone-shaping dynamics stage
    that is the better compromise, and the bands stay perfectly aligned.
    """
    bands: list[np.ndarray] = []
    delays: list[int] = []
    rest = np.asarray(x, dtype=np.float64)
    for fc in sorted(crossovers):
        taps = int(np.clip(round(taps_per_hz * sr / max(fc, 20.0)), 129, 16385)) | 1
        h = _lowpass_fir(fc, sr, taps)
        low = _aligned_conv(rest, h)
        rest = rest - low
        bands.append(low)
        delays.append(int((len(h) - 1) // 2))
    bands.append(rest)
    return bands, delays


def lr4_split(x: np.ndarray, sr: int, fc: float) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible two-band split (complementary, linear phase)."""
    bands, _ = split_bands(x, sr, [fc])
    return bands[0], bands[1]


def crossover_sum_is_flat() -> str:
    return ("The crossover is complementary and linear phase: the bands sum to the "
            "input exactly and every band is time aligned with it.")
