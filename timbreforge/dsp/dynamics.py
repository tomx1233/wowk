"""Dynamics: feed-forward compressor, multiband, dynamic EQ, transient shaper and
an offline true-peak limiter.

Rules baked into this module
----------------------------
* All compressor stages are *feed-forward* with an explicit threshold/ratio and
  an explicit, usually negative, makeup gain.  No stage silently raises level.
* Detection is done at a control rate (about 2 kHz) and smoothed with proper
  attack/release ballistics; the resulting gain curve is interpolated to audio
  rate so there is no zipper noise and no lookahead latency.
* The multiband stage splits with a complementary linear-phase FIR crossover:
  the bands sum back to the input sample for sample, so the stage cannot smear
  transients or shift the mix.
* The limiter works offline with a look-ahead window.  Because the gain curve is
  computed over the whole file, the signal itself is never delayed: output length
  always equals input length, so timing is preserved exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .biquad import high_pass, low_shelf, peaking, split_bands
from .meter import db, true_peak


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _control_step(sr: int, ctrl_hz: float = 2000.0) -> int:
    return max(1, int(round(sr / min(ctrl_hz, sr))))


def _db_to_lin(d: np.ndarray | float) -> np.ndarray:
    return 10.0 ** (np.asarray(d, dtype=np.float64) / 20.0)


def _coef(ms: float, fs: float) -> float:
    if ms <= 0:
        return 1.0
    return float(1.0 - math.exp(-1.0 / (ms * 1e-3 * fs)))


def _block_reduce(x: np.ndarray, step: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Reduce audio-rate signal to control-rate start indices and values."""
    n = x.shape[0]
    starts = np.arange(0, n, step)
    stops = np.minimum(starts + step, n)
    vals = np.empty(starts.size, dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, stops)):
        seg = x[s:e]
        vals[i] = np.max(seg) if mode == "max" else float(np.mean(seg))
    return starts, vals


def _smooth_gr(target_db: np.ndarray, attack_ms: float, release_ms: float,
               fs_ctrl: float) -> np.ndarray:
    """Ballistic smoother for gain reduction (dB, value <= 0)."""
    att = _coef(attack_ms, fs_ctrl)
    rel = _coef(release_ms, fs_ctrl)
    out = np.empty_like(target_db)
    g = 0.0
    for i in range(target_db.size):
        t = target_db[i]
        g += ((t - g) * att) if t < g else ((t - g) * rel)
        out[i] = g
    return out


def _curve_to_audio(starts: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Interpolate a control-rate curve to audio rate, sample aligned."""
    if values.size == 1:
        return np.full(n, values[0])
    grid = np.arange(n, dtype=np.float64)
    src = starts.astype(np.float64)
    return np.interp(grid, src, values, left=values[0], right=values[-1])


# ---------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorParams:
    enabled: bool = False
    threshold_db: float = -18.0
    ratio: float = 2.0
    attack_ms: float = 15.0
    release_ms: float = 180.0
    knee_db: float = 8.0
    detector: str = "rms"          # "rms" | "peak"
    window_ms: float = 12.0        # RMS integration window
    stereo: str = "linked"         # "linked" | "independent"
    mix: float = 1.0               # parallel blend (1.0 = fully compressed)
    sidechain_hp_hz: float = 0.0   # 0 = off
    makeup_db: float = 0.0         # negative or zero for the shipped profile

    @staticmethod
    def from_dict(d: dict) -> "CompressorParams":
        p = CompressorParams()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                setattr(p, k, type(cur)(v) if cur is not None and not isinstance(cur, bool) else v)
        return p

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "threshold_db": self.threshold_db,
            "ratio": self.ratio, "attack_ms": self.attack_ms,
            "release_ms": self.release_ms, "knee_db": self.knee_db,
            "detector": self.detector, "window_ms": self.window_ms,
            "stereo": self.stereo, "mix": self.mix,
            "sidechain_hp_hz": self.sidechain_hp_hz,
            "makeup_db": self.makeup_db,
        }

    def validate(self) -> list[str]:
        errs = []
        if not (0.0 <= self.mix <= 1.0):
            errs.append("compressor mix must be between 0 and 1")
        if self.ratio < 1.0:
            errs.append("compressor ratio must be >= 1")
        if self.attack_ms < 0 or self.release_ms <= 0:
            errs.append("compressor attack/release must be non-negative, release > 0")
        return errs


def _gain_computer_db(level_db: np.ndarray, p: CompressorParams) -> np.ndarray:
    t, k = p.threshold_db, max(p.knee_db, 0.0)
    r = max(p.ratio, 1.0)
    slope = 1.0 - 1.0 / r
    down = np.zeros_like(level_db)
    if k <= 0:
        over = np.maximum(level_db - t, 0.0)
        return -over * slope
    lower, upper = t - k / 2.0, t + k / 2.0
    below = level_db <= lower
    above = level_db >= upper
    mid = ~(below | above)
    down[mid] = -(slope * (level_db[mid] - lower) ** 2) / (2.0 * k)
    down[above] = -slope * (level_db[above] - t)
    return down


def compress(x: np.ndarray, sr: int, p: CompressorParams) -> tuple[np.ndarray, dict]:
    """Feed-forward compressor.  Returns processed audio and a report dict."""
    x = _as2d(x)
    if not p.enabled:
        return x, {"max_reduction_db": 0.0, "bypassed": True}

    step = _control_step(sr)
    fs_ctrl = sr / float(step)

    def run(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """sig: (n, 1).  Returns processed channel and control-rate gain curve."""
        if p.detector == "peak":
            det = np.abs(sig).max(axis=1)
            starts, vals = _block_reduce(det, step, "max")
        else:
            sq = np.mean(sig * sig, axis=1)
            starts, vals = _block_reduce(sq, step, "mean")
        if p.detector != "peak":
            # one-pole power smoothing at the control rate
            a = _coef(p.window_ms, fs_ctrl)
            sm = np.empty_like(vals)
            acc = vals[0] if vals.size else 0.0
            for i, v in enumerate(vals):
                acc += (v - acc) * a
                sm[i] = acc
            vals = sm
        level_db = 20.0 * np.log10(np.maximum(np.sqrt(vals) if p.detector != "peak" else vals,
                                              1e-9))
        target = _gain_computer_db(level_db, p)
        curve = _smooth_gr(target, p.attack_ms, p.release_ms, fs_ctrl)
        return curve, starts

    if x.shape[1] == 2 and p.stereo == "linked":
        det_src = x
        if p.sidechain_hp_hz > 0:
            b, a = high_pass(p.sidechain_hp_hz, 0.707, sr)
            det_src = signal.lfilter(b, a, x, axis=0)
        curve, starts = run(np.mean(det_src, axis=1, keepdims=True))
        g_audio_db = _curve_to_audio(starts, curve, x.shape[0])
        lin = _db_to_lin(g_audio_db)
        del g_audio_db
        wet = np.empty_like(x)
        np.multiply(x, lin[:, None], out=wet)
        del lin
        gr_curve = curve
    else:
        wet = np.empty_like(x)
        curves = []
        for c in range(x.shape[1]):
            curve, starts = run(x[:, c:c + 1])
            curves.append(curve)
            wet[:, c] = x[:, c] * _db_to_lin(_curve_to_audio(starts, curve, x.shape[0]))
        gr_curve = np.min(np.vstack(curves), axis=0) if curves else np.zeros(1)

    if p.mix >= 0.999:
        y = wet
    else:
        y = np.empty_like(x)
        np.multiply(x, 1.0 - p.mix, out=y)
        y += wet * p.mix
        del wet
    if abs(p.makeup_db) > 1e-9:
        y *= _db_to_lin(p.makeup_db)
    report = {
        "max_reduction_db": float(min(gr_curve.min(), 0.0)) if gr_curve.size else 0.0,
        "typical_reduction_db": float(np.percentile(gr_curve, 50)) if gr_curve.size else 0.0,
        "bypassed": False,
    }
    return y, report


# ---------------------------------------------------------------------------
# multiband
# ---------------------------------------------------------------------------


@dataclass
class MultibandParams:
    enabled: bool = False
    crossovers_hz: list[float] = field(default_factory=lambda: [180.0, 1800.0])
    #: linear-phase FIR length as taps per hertz of crossover frequency
    taps_per_hz: float = 4.0
    bands: list[CompressorParams] = field(default_factory=lambda: [CompressorParams(),
                                                                  CompressorParams(),
                                                                  CompressorParams()])

    @staticmethod
    def from_dict(d: dict) -> "MultibandParams":
        xo = [float(v) for v in d.get("crossovers_hz", [180.0, 1800.0])]
        bands = [CompressorParams.from_dict(b) for b in d.get("bands", [])]
        while len(bands) < len(xo) + 1:
            bands.append(CompressorParams())
        return MultibandParams(enabled=bool(d.get("enabled", False)),
                               crossovers_hz=xo,
                               taps_per_hz=float(d.get("taps_per_hz", 4.0)),
                               bands=bands[: len(xo) + 1])

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "crossovers_hz": list(self.crossovers_hz),
                "taps_per_hz": self.taps_per_hz,
                "bands": [b.to_dict() for b in self.bands]}

    def validate(self) -> list[str]:
        errs = []
        xo = sorted(self.crossovers_hz)
        if any(f <= 20 or f >= 20000 for f in xo):
            errs.append("multiband crossovers must be between 20 Hz and 20 kHz")
        errs.extend(f"MB band {i + 1}: {e}" for i, b in enumerate(self.bands) for e in b.validate())
        return errs


def multiband(x: np.ndarray, sr: int, p: MultibandParams) -> tuple[np.ndarray, dict]:
    """Band-wise dynamics around a complementary linear-phase crossover.

    With neutral band settings the stage is transparent (the bands sum back to
    the input sample for sample), so it cannot damage timing or phase.
    """
    x = _as2d(x)
    if not p.enabled:
        return x, {"bands": [], "bypassed": True}
    xo = sorted(f for f in p.crossovers_hz if 20 < f < sr * 0.45)
    if not xo:
        return x, {"bands": [], "bypassed": True}
    parts, delays = split_bands(x, sr, xo, p.taps_per_hz)
    out = np.zeros_like(x)
    reports = []
    for i, part in enumerate(parts):
        params = p.bands[i] if i < len(p.bands) else CompressorParams()
        processed, rep = compress(part, sr, params)
        out += processed
        reports.append(rep)
        del processed, part
    return out, {"bands": reports, "bypassed": False, "crossovers_hz": xo,
                 "crossover": "complementary linear-phase FIR (zero net delay)",
                 "fir_group_delays_removed": delays}


# ---------------------------------------------------------------------------
# dynamic EQ
# ---------------------------------------------------------------------------


@dataclass
class DynamicEqBand:
    enabled: bool = False
    freq: float = 3000.0
    q: float = 1.0
    gain_db: float = -3.0        # sign of the target shape
    threshold_db: float = -30.0
    ratio: float = 2.0
    attack_ms: float = 15.0
    release_ms: float = 200.0
    direction: str = "above"     # "above" -> cut when loud, "below" -> boost when quiet
    max_reduction_db: float = 12.0
    stereo: str = "linked"

    @staticmethod
    def from_dict(d: dict) -> "DynamicEqBand":
        b = DynamicEqBand()
        for k, v in (d or {}).items():
            if hasattr(b, k) and v is not None:
                cur = getattr(b, k)
                setattr(b, k, v if isinstance(cur, bool) else type(cur)(v))
        return b

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("enabled", "freq", "q", "gain_db", "threshold_db", "ratio",
                 "attack_ms", "release_ms", "direction", "max_reduction_db", "stereo")}

    def validate(self) -> list[str]:
        errs = []
        if self.ratio < 1:
            errs.append("dynamic EQ ratio must be >= 1")
        if self.direction not in ("above", "below"):
            errs.append("dynamic EQ direction must be 'above' or 'below'")
        if abs(self.gain_db) > 18:
            errs.append("dynamic EQ shape gain is limited to +/-18 dB")
        return errs


@dataclass
class DynamicEqParams:
    enabled: bool = False
    bands: list[DynamicEqBand] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "DynamicEqParams":
        return DynamicEqParams(enabled=bool(d.get("enabled", False)),
                               bands=[DynamicEqBand.from_dict(b) for b in d.get("bands", [])])

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "bands": [b.to_dict() for b in self.bands]}

    def validate(self) -> list[str]:
        out = []
        for i, b in enumerate(self.bands):
            out.extend(f"dynamic EQ band {i + 1}: {e}" for e in b.validate())
        return out


def dynamic_eq(x: np.ndarray, sr: int, p: DynamicEqParams) -> tuple[np.ndarray, dict]:
    """Level-dependent peaking.

    Implementation note: each band uses a fixed peaking filter at its maximum
    shape gain and modulates how much of that filtered path is blended back.
    That is a smooth (zipper-free) approximation of modulating a filter's gain,
    and it cannot go unstable because the filter coefficients never change.
    """
    x = _as2d(x)
    if not p.enabled or not p.bands:
        return x, {"bands": [], "bypassed": True}
    y = x.copy()
    step = _control_step(sr)
    fs_ctrl = sr / float(step)
    reports = []
    for band in p.bands:
        if not band.enabled or abs(band.gain_db) < 0.05:
            continue
        # static detector band-pass (mono sum), output at audio rate
        b, a = peaking(band.freq, band.gain_db, band.q, sr)
        full = signal.lfilter(b, a, y, axis=0)
        det_b, det_a = _detector_bandpass(band.freq, band.q, sr)
        det = signal.lfilter(det_b, det_a, y.mean(axis=1), axis=0)
        starts, vals = _block_reduce(det * det, step, "mean")
        lvl_db = 20 * np.log10(np.maximum(np.sqrt(vals), 1e-9))
        over = (lvl_db - band.threshold_db) if band.direction == "above" \
            else (band.threshold_db - lvl_db)
        over = np.maximum(over, 0.0)
        amount = over * (1.0 - 1.0 / max(band.ratio, 1.0))
        amount = np.clip(amount, 0.0, max(band.max_reduction_db, 0.0)) / max(abs(band.gain_db), 1e-9)
        amount = np.clip(amount, 0.0, 1.0)
        amount = _smooth_gr(-amount, band.attack_ms, band.release_ms, fs_ctrl) * -1.0
        wet_curve = np.clip(_curve_to_audio(starts, amount, x.shape[0]), 0.0, 1.0)
        y += (full - y) * wet_curve[:, None]
        del full
        reports.append({"freq": band.freq, "peak_amount": float(wet_curve.max())})
    return y, {"bands": reports, "bypassed": False}


def _detector_bandpass(fc: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * min(max(fc, 20.0), sr * 0.45) / sr
    alpha = math.sin(w0) / (2 * max(q, 0.1))
    cw = math.cos(w0)
    bb = np.array([alpha, 0.0, -alpha])
    aa = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return bb / aa[0], aa / aa[0]


# ---------------------------------------------------------------------------
# transient shaper
# ---------------------------------------------------------------------------


@dataclass
class TransientParams:
    enabled: bool = False
    attack_db: float = 0.0        # + emphasises attacks, - softens them
    sustain_db: float = 0.0       # + lifts tails, - tightens them
    fast_ms: float = 2.0
    slow_ms: float = 40.0
    sensitivity_db: float = 6.0

    @staticmethod
    def from_dict(d: dict) -> "TransientParams":
        t = TransientParams()
        for k, v in (d or {}).items():
            if hasattr(t, k) and v is not None:
                cur = getattr(t, k)
                setattr(t, k, v if isinstance(cur, bool) else type(cur)(v))
        return t

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("enabled", "attack_db", "sustain_db", "fast_ms", "slow_ms", "sensitivity_db")}

    def validate(self) -> list[str]:
        e = []
        if abs(self.attack_db) > 12 or abs(self.sustain_db) > 12:
            e.append("transient shaping is limited to +/-12 dB")
        return e


def transient_shape(x: np.ndarray, sr: int, p: TransientParams) -> tuple[np.ndarray, dict]:
    x = _as2d(x)
    if not p.enabled or (abs(p.attack_db) < 0.05 and abs(p.sustain_db) < 0.05):
        return x, {"bypassed": True}
    mono = np.abs(x).max(axis=1)
    fast = _onepole(mono, p.fast_ms, sr)
    slow = _onepole(mono, p.slow_ms, sr)
    diff_db = 20 * np.log10(np.maximum(fast, 1e-9)) - 20 * np.log10(np.maximum(slow, 1e-9))
    sens = max(p.sensitivity_db, 0.5)
    att = np.clip(diff_db / sens, 0.0, 1.0) * p.attack_db
    sus = np.clip(-diff_db / sens, 0.0, 1.0) * p.sustain_db
    gain_db = att + sus
    gain_db = _onepole(gain_db, 12.0, sr)
    y = x * _db_to_lin(gain_db)[:, None]
    return y, {"bypassed": False, "max_boost_db": float(gain_db.max()),
               "max_cut_db": float(gain_db.min())}


def _onepole(v: np.ndarray, tau_ms: float, sr: int) -> np.ndarray:
    a = _coef(tau_ms, sr)
    return signal.lfilter([a], [1.0, a - 1.0], v)


# ---------------------------------------------------------------------------
# true-peak limiter
# ---------------------------------------------------------------------------


@dataclass
class LimiterParams:
    enabled: bool = True
    ceiling_db: float = -1.0
    lookahead_ms: float = 6.0
    release_ms: float = 140.0
    oversample: int = 4

    @staticmethod
    def from_dict(d: dict) -> "LimiterParams":
        p = LimiterParams()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                setattr(p, k, type(getattr(p, k))(v))
        return p

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "ceiling_db": self.ceiling_db,
                "lookahead_ms": self.lookahead_ms, "release_ms": self.release_ms,
                "oversample": self.oversample}

    def validate(self) -> list[str]:
        e = []
        if self.ceiling_db > -0.1:
            e.append("true-peak ceiling must be at or below -0.1 dBTP")
        return e


def _intersample_envelope(x: np.ndarray, oversample: int,
                          chunk: int = 1 << 18) -> np.ndarray:
    """Per-input-sample peak including inter-sample peaks.

    Processed in chunks so long files never allocate a full oversampled copy
    (a 2.5 minute stereo file would otherwise need hundreds of megabytes).
    """
    env = np.abs(x).max(axis=1).astype(np.float32, copy=False)
    factor = int(oversample)
    if factor <= 1:
        return env
    n = x.shape[0]
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        seg = x[start:stop]
        up = signal.resample_poly(seg, factor, 1, axis=0, window=("kaiser", 8.6))
        m = min(up.shape[0] // factor, seg.shape[0])
        up = up[: m * factor].reshape(m, factor, seg.shape[1])
        isp = np.abs(up).max(axis=(1, 2)).astype(np.float32)
        env[start:start + m] = np.maximum(env[start:start + m], isp)
    return env


def _rolling_max(v: np.ndarray, half: int) -> np.ndarray:
    if half <= 0:
        return v
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(v, size=2 * half + 1, mode="nearest")


def limit(x: np.ndarray, sr: int, p: LimiterParams) -> tuple[np.ndarray, dict]:
    """Look-ahead true-peak limiter with zero output latency (length preserved)."""
    x = _as2d(x)
    if not p.enabled:
        return x, {"bypassed": True}
    ceiling = 10 ** (p.ceiling_db / 20.0)
    env = _intersample_envelope(x, int(max(1, p.oversample)))
    peak = float(env.max()) if env.size else 0.0
    if peak <= ceiling:
        return x, {"bypassed": True, "peak_ceiling_db": p.ceiling_db}

    half = max(1, int(round(sr * p.lookahead_ms / 1000.0)))
    env_hold = _rolling_max(env.astype(np.float32, copy=False), half)
    target_db = np.minimum(0.0, p.ceiling_db - db(env_hold, -160.0)).astype(np.float32)
    del env_hold

    # release as a slope in dB/sample; attack is instant but the rolling max
    # already gives a smooth ramp into each peak.  All in place to keep the
    # memory footprint sane on long files.
    rel_db_per_sample = np.float32(60.0 / max(1.0, p.release_ms * 1e-3 * sr))
    ramp = np.arange(target_db.size, dtype=np.float32) * rel_db_per_sample
    target_db += ramp
    curve = np.maximum.accumulate(target_db)
    curve -= ramp
    del ramp, target_db
    np.minimum(curve, 0.0, out=curve)

    # soften the corners slightly, then re-clamp so the ceiling still holds
    smooth_ms = max(0.4, p.lookahead_ms / 4.0)
    win = max(3, int(round(sr * smooth_ms / 1000.0)) | 1)
    kernel = np.hanning(win).astype(np.float32)
    kernel /= kernel.sum()
    curve = np.convolve(curve, kernel, mode="same").astype(np.float32, copy=False)
    np.minimum(curve, 0.0, out=curve)

    y = x * _db_to_lin(curve.astype(np.float64))[:, None]

    # verification passes: guarantee the encoded ceiling, reduce gain if needed
    for _ in range(3):
        measured = db(true_peak(y, factor=4), -160.0)
        if measured <= p.ceiling_db + 0.02:
            break
        y = y * (10 ** ((p.ceiling_db - measured) / 20.0))
    return y, {
        "bypassed": False,
        "max_reduction_db": float(curve.min()),
        "input_true_peak_db": float(db(peak, -160.0)),
        "output_true_peak_db": float(db(true_peak(y), -160.0)),
        "ceiling_db": p.ceiling_db,
    }


# ---------------------------------------------------------------------------
# gain
# ---------------------------------------------------------------------------


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-9:
        return x
    return x * (10.0 ** (gain_db / 20.0))


def measure_true_peak_db(x: np.ndarray) -> float:
    return float(db(true_peak(x), -160.0))
