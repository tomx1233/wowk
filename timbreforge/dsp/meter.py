"""Objective measurement: levels, true peak, LUFS, envelope, stereo, spectrum.

Everything here is read-only measurement.  Nothing in this module is used to
make the output louder; the loudness figures exist so that preview/export can be
compared *at matched level* and so the reference study can describe what it
actually observed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from scipy import signal

EPS = 1e-20


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def db(x: float | np.ndarray, floor: float = -200.0) -> float | np.ndarray:
    """Amplitude ratio -> dB, with a floor instead of -inf."""
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(x), 10 ** (floor / 20.0))), floor)


def undb(d: float) -> float:
    return float(10.0 ** (d / 20.0))


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float64, copy=False)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float64, copy=False)
    return x.mean(axis=1)


# ---------------------------------------------------------------------------
# peak / true peak
# ---------------------------------------------------------------------------


def _chunked_true_peak(x: np.ndarray, factor: int = 4, chunk: int = 1 << 18) -> float:
    peak = 0.0
    n = x.shape[0]
    if n == 0:
        return 0.0
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        seg = x[start:stop]
        # guard edges so the interpolation filter never sees an artificial step
        pad = 48
        left = x[max(0, start - pad):start]
        right = x[stop:min(n, stop + pad)]
        ext = np.concatenate([left, seg, right])
        if ext.shape[0] < 8:
            peak = max(peak, float(np.max(np.abs(seg))))
            continue
        up = signal.resample_poly(ext, factor, 1, window=("kaiser", 8.6))
        lo = int(math.ceil(len(left) * factor))
        hi = lo + int(len(seg) * factor)
        peak = max(peak, float(np.max(np.abs(up[lo:hi]))))
    return peak


def true_peak(x: np.ndarray, factor: int = 4) -> float:
    """Inter-sample peak estimate (oversampled), linear amplitude."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return _chunked_true_peak(x, factor)
    return max(_chunked_true_peak(x[:, c], factor) for c in range(x.shape[1]))


def sample_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x))) if x.size else 0.0


# ---------------------------------------------------------------------------
# RMS / envelope
# ---------------------------------------------------------------------------


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def rms_db(x: np.ndarray, floor: float = -120.0) -> float:
    v = rms(x)
    return float(max(20.0 * math.log10(max(v, 1e-12)), floor))


def envelope_db(x: np.ndarray, sr: int, window_ms: float = 50.0,
                hop_ms: float | None = None) -> np.ndarray:
    """Short-term RMS envelope in dBFS."""
    mono = to_mono(x)
    win = max(1, int(round(sr * window_ms / 1000.0)))
    hop = max(1, int(round(sr * (hop_ms if hop_ms is not None else window_ms / 2) / 1000.0)))
    if mono.shape[0] < win:
        return np.array([rms_db(mono)])
    n = 1 + (mono.shape[0] - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx]
    p = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(p, 1e-12))


def envelope_range_db(x: np.ndarray, sr: int, window_ms: float = 50.0) -> float:
    env = envelope_db(x, sr, window_ms)
    if env.size < 2:
        return 0.0
    p95, p5 = np.percentile(env, 95), np.percentile(env, 5)
    return float(p95 - p5)


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 loudness
# ---------------------------------------------------------------------------

# BS.1770 K-weighting, designed from the standard's analog prototypes so the
# coefficients are correct at any sample rate (not only 48 kHz).
_SHELF_G = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_FC = 1681.974450955533
_HP_Q = 0.5003270373238773
_HP_FC = 38.13547087602444


def k_weighting_coefficients(sr: int) -> tuple[np.ndarray, np.ndarray]:
    # --- stage 1: high shelf (RBJ form, as used by the standard) -----------
    A = 10 ** (_SHELF_G / 40.0)
    w0 = 2 * math.pi * _SHELF_FC / sr
    alpha = math.sin(w0) / (2 * _SHELF_Q)
    cw = math.cos(w0)
    sqA = math.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * sqA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * sqA * alpha
    shelf_b = np.array([b0, b1, b2]) / a0
    shelf_a = np.array([1.0, a1 / a0, a2 / a0])

    # --- stage 2: RLB high-pass -------------------------------------------
    w0 = 2 * math.pi * _HP_FC / sr
    alpha = math.sin(w0) / (2 * _HP_Q)
    cw = math.cos(w0)
    b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    hp_b, hp_a = b / a[0], a / a[0]
    return (shelf_b, shelf_a), (hp_b, hp_a)


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply K-weighting to a (frames,) or (frames, channels) array."""
    x = np.asarray(x, dtype=np.float64)
    (b1, a1), (b2, a2) = k_weighting_coefficients(sr)
    if x.ndim == 1:
        y = signal.lfilter(b1, a1, x)
        return signal.lfilter(b2, a2, y)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        y = signal.lfilter(b1, a1, x[:, c])
        out[:, c] = signal.lfilter(b2, a2, y)
    return out


_CHANNEL_WEIGHTS = {1: [1.0], 2: [1.0, 1.0], 3: [1.0, 1.0, 1.0],
                    4: [1.0, 1.0, 1.41, 1.41], 5: [1.0, 1.0, 1.0, 1.41, 1.41]}


def _loudness_from_blocks(block_powers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # block_powers: (blocks, channels) mean-square; weights: (channels,)
    return -0.691 + 10.0 * np.log10(np.maximum(block_powers @ weights, EPS))


def loudness_blocks(x: np.ndarray, sr: int, window_ms: float = 400.0,
                    hop_ms: float = 100.0) -> np.ndarray:
    """Ungated block loudness values (LKFS)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = k_weight(x, sr)
    weights = np.array(_CHANNEL_WEIGHTS.get(x.shape[1], [1.0] * x.shape[1]))
    win = max(1, int(round(sr * window_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.shape[0] < win:
        p = np.mean(y * y, axis=0)[None, :]
        return _loudness_from_blocks(p, weights)
    n = 1 + (x.shape[0] - win) // hop
    # cumulative sums keep this O(n) instead of allocating every block
    cs = np.cumsum(y * y, axis=0)
    cs = np.vstack([np.zeros((1, y.shape[1])), cs])
    starts = np.arange(n) * hop
    stops = starts + win
    p = (cs[stops] - cs[starts]) / win
    return _loudness_from_blocks(p, weights)


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness, LUFS (BS.1770-4). -inf-ish floor at -70."""
    blocks = loudness_blocks(x, sr)
    if blocks.size == 0:
        return -70.0
    abs_gate = blocks >= -70.0
    if not abs_gate.any():
        return -70.0
    gated = blocks[abs_gate]
    rel_gate = gated.mean() - 10.0
    keep = blocks >= rel_gate
    if not keep.any():
        return float(gated.mean())
    return float(10.0 * math.log10(np.mean(10.0 ** (blocks[keep] / 10.0))))


def loudness_range(x: np.ndarray, sr: int) -> float:
    """EBU R128 LRA from 3 s short-term blocks."""
    st = loudness_blocks(x, sr, window_ms=3000.0, hop_ms=1000.0)
    st = st[np.isfinite(st)]
    if st.size < 2:
        return 0.0
    gate_abs = st >= -70.0
    if not gate_abs.any():
        return 0.0
    s = st[gate_abs]
    rel = s.mean() - 20.0
    s = s[s >= rel]
    if s.size < 2:
        return 0.0
    return float(np.percentile(s, 95) - np.percentile(s, 10))


# ---------------------------------------------------------------------------
# stereo
# ---------------------------------------------------------------------------


def channel_correlation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] < 2:
        return 1.0
    L, R = x[:, 0], x[:, 1]
    lc = L - L.mean()
    rc = R - R.mean()
    denom = math.sqrt(float(np.dot(lc, lc)) * float(np.dot(rc, rc)))
    if denom <= 1e-15:
        return 1.0
    return float(np.dot(lc, rc) / denom)


def side_mid_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] < 2:
        return -120.0
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1])
    rm, rs = rms(m), rms(s)
    return float(20.0 * math.log10(max(rs, 1e-12))) - float(20.0 * math.log10(max(rm, 1e-12)))


def mono_fold_change(x: np.ndarray) -> float:
    """Level change (dB) when folded to mono with an (L+R)/2 downmix.

    0 dB means the mono fold-down keeps the same level as the stereo signal;
    negative values mean the fold-down loses level (phase cancellation).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] < 2:
        return 0.0
    m = x.mean(axis=1)
    return rms_db(m) - rms_db(x)


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------


def power_spectrum(x: np.ndarray, sr: int, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Welch long-term average power spectrum of the mono downmix."""
    mono = to_mono(x)
    if mono.shape[0] < n_fft:
        n_fft = int(2 ** max(8, int(math.log2(max(mono.shape[0], 256)))))
    f, p = signal.welch(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2,
                        window="hann", detrend=False, scaling="density")
    return f, p


def spectral_centroid(x: np.ndarray, sr: int, n_fft: int = 4096) -> float:
    """Energy-weighted mean frequency of the long-term spectrum (Hz)."""
    f, p = power_spectrum(x, sr, n_fft)
    total = float(np.sum(p))
    if total <= 0:
        return 0.0
    return float(np.sum(f * p) / total)


def spectral_flatness(x: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    """Mean spectral flatness (0 = tonal, 1 = noise-like), 20 Hz..20 kHz."""
    mono = to_mono(x)
    f, t, Z = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2,
                          window="hann", boundary=None, padded=False)
    mag = np.abs(Z)
    band = (f >= 20) & (f <= min(20000, sr / 2))
    mag = mag[band] + 1e-12
    flat = np.exp(np.mean(np.log(mag), axis=0)) / np.mean(mag, axis=0)
    return float(np.mean(flat))


def spectral_rolloff(x: np.ndarray, sr: int, fraction: float = 0.85) -> float:
    f, p = power_spectrum(x, sr)
    c = np.cumsum(p)
    if c[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(c, c[-1] * fraction))
    return float(f[min(idx, len(f) - 1)])


#: Standard analysis bands (Hz edges).  Deliberately broad: a narrow band table
#: encourages over-reading a spectral ratio as an EQ curve.
ANALYSIS_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub 20-60", 20, 60),
    ("bass 60-120", 60, 120),
    ("lowmid 120-300", 120, 300),
    ("mid 300-1k", 300, 1000),
    ("uppermid 1-3k", 1000, 3000),
    ("presence 3-6k", 3000, 6000),
    ("treble 6-10k", 6000, 10000),
    ("air 10-16k", 10000, 16000),
    ("top 16k+", 16000, 24000),
)


def band_energies_db(x: np.ndarray, sr: int, bands=ANALYSIS_BANDS,
                     n_fft: int = 8192) -> dict[str, float]:
    f, p = power_spectrum(x, sr, n_fft)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    out: dict[str, float] = {}
    for name, lo, hi in bands:
        sel = (f >= lo) & (f < min(hi, sr / 2 + 1))
        if not sel.any():
            out[name] = -120.0
            continue
        out[name] = float(10 * math.log10(max(np.sum(p[sel]) * df, 1e-12)))
    return out


def third_octave(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """1/3-octave band levels in dB, 20 Hz..20 kHz."""
    f, p = power_spectrum(x, sr, n_fft=16384)
    df = f[1] - f[0]
    centers = 1000.0 * 2.0 ** (np.arange(-17, 14) / 3.0)  # 25 Hz .. 16 kHz
    levels = []
    for fc in centers:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        sel = (f >= lo) & (f < hi)
        levels.append(10 * math.log10(max(np.sum(p[sel]) * df, 1e-12)) if sel.any() else -120.0)
    return centers, np.array(levels)


# ---------------------------------------------------------------------------
# aggregate stats
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    frames: int = 0
    peak_db: float = -120.0
    true_peak_db: float = -120.0
    rms_db: float = -120.0
    lufs: float = -70.0
    lra: float = 0.0
    crest_db: float = 0.0
    envelope_range_db: float = 0.0
    clipped_fraction: float = 0.0
    dc_offset: float = 0.0
    centroid_hz: float = 0.0
    flatness: float = 0.0
    rolloff_hz: float = 0.0
    correlation: float = 1.0
    side_mid_db: float = -120.0
    mono_fold_db: float = 0.0
    bands: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"Duration      {self.duration:.3f} s   ({self.frames} frames @ {self.sample_rate} Hz, {self.channels} ch)",
            f"Sample peak   {self.peak_db:+.2f} dBFS     true peak {self.true_peak_db:+.2f} dBTP",
            f"RMS           {self.rms_db:+.2f} dBFS     crest {self.crest_db:.2f} dB",
            f"Loudness      {self.lufs:.2f} LUFS      LRA {self.lra:.2f} LU",
            f"Envelope      {self.envelope_range_db:.2f} dB (50 ms, p95-p5)",
            f"Spectrum      centroid {self.centroid_hz:.0f} Hz, flatness {self.flatness:.3f}, rolloff85 {self.rolloff_hz:.0f} Hz",
            f"Stereo        corr {self.correlation:.3f}, S/M {self.side_mid_db:+.2f} dB, mono fold {self.mono_fold_db:+.2f} dB",
            f"Clipped       {self.clipped_fraction * 100:.4f} % of samples at/above 0.999 FS",
        ]


def measure(x: np.ndarray, sr: int, *, true_peak_estimate: bool = True,
            bands: bool = True) -> Stats:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    s = Stats()
    s.sample_rate = int(sr)
    s.channels = int(x.shape[1])
    s.frames = int(x.shape[0])
    s.duration = s.frames / float(sr) if sr else 0.0
    s.peak_db = float(db(sample_peak(x), -120.0))
    s.true_peak_db = float(db(true_peak(x), -120.0)) if true_peak_estimate else s.peak_db
    s.rms_db = rms_db(x)
    s.lufs = integrated_loudness(x, sr)
    s.lra = loudness_range(x, sr)
    s.crest_db = s.true_peak_db - s.rms_db
    s.envelope_range_db = envelope_range_db(x, sr)
    s.clipped_fraction = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    s.dc_offset = float(np.mean(x))
    s.centroid_hz = spectral_centroid(x, sr)
    s.flatness = spectral_flatness(x, sr)
    s.rolloff_hz = spectral_rolloff(x, sr)
    if x.shape[1] >= 2:
        s.correlation = channel_correlation(x)
        s.side_mid_db = side_mid_db(x)
        s.mono_fold_db = mono_fold_change(x)
    if bands:
        s.bands = band_energies_db(x, sr)
    if not np.isfinite(x).all():
        s.warnings.append("non-finite samples present")
    return s


def compare_levels(a: Stats, b: Stats) -> dict[str, float]:
    """Gain-invariant comparison between two measurements."""
    return {
        "lufs_delta": b.lufs - a.lufs,
        "rms_delta": b.rms_db - a.rms_db,
        "true_peak_delta": b.true_peak_db - a.true_peak_db,
        "crest_delta": b.crest_db - a.crest_db,
        "envelope_range_delta": b.envelope_range_db - a.envelope_range_db,
        "centroid_ratio": (b.centroid_hz + 1e-9) / (a.centroid_hz + 1e-9),
        "correlation_delta": b.correlation - a.correlation,
        "side_mid_delta": b.side_mid_db - a.side_mid_db,
        "flatness_delta": b.flatness - a.flatness,
    }


def band_delta_db(a: dict[str, float], b: dict[str, float],
                  bands: Iterable[str] | None = None) -> dict[str, float]:
    keys = list(a.keys()) if bands is None else list(bands)
    return {k: b.get(k, -120.0) - a.get(k, -120.0) for k in keys}


def level_match_gain_db(ref_lufs: float, other_lufs: float,
                        allow_boost: bool = False) -> float:
    """Gain to apply to ``other`` so it sits at ``ref`` loudness.

    With ``allow_boost=False`` the result is never positive, i.e. matching can
    only attenuate.
    """
    g = float(ref_lufs - other_lufs)
    return g if allow_boost else min(0.0, g)
