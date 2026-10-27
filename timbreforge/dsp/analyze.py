"""Reference-pair analysis.

This module exists to answer, with numbers, the questions the previous attempt
got wrong:

* Are the two files the same recording?  (duration alone proves nothing)
* Is there a real timing offset, and is there drift?  (no reused fixed offset:
  the lag is measured per segment with normalised correlation)
* Is the pitch identical?  (fine tuning estimated from spectral peaks, not from
  chroma alone)
* What changes are *gain invariant*, and which of them depend on level?
  (spectra are compared level-matched and level-stratified, so a static EQ curve
  is not confused with a nonlinear/dynamic process)
* What cannot be reproduced while keeping the quality rules?  (peak/crest
  figures of a crushed reference are described, not targeted)

Nothing here is used by the application's render path; the app only needs the
profile JSON that this produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .audio import load_audio, AudioData
from .meter import (
    ANALYSIS_BANDS, Stats, band_energies_db, channel_correlation, db,
    envelope_db, envelope_range_db, integrated_loudness, loudness_range, measure,
    power_spectrum, rms_db, side_mid_db, spectral_centroid, spectral_flatness,
    to_mono, true_peak,
)

EPS = 1e-20


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def _prepare_alignment_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Band-limit to 150-4000 Hz so noise/codec differences do not dominate."""
    mono = to_mono(x)
    sos = signal.butter(3, [150.0 / (sr / 2), min(4000.0 / (sr / 2), 0.95)],
                        btype="band", output="sos")
    y = signal.sosfilt(sos, mono)
    y = y - y.mean()
    return y


def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation for lags in [-max_lag, max_lag].

    Returns (lags, r) where r is a genuine Pearson-style correlation of the
    overlapping region, so its magnitude is directly interpretable in [0, 1].
    """
    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]
    nfft = 1 << int(math.ceil(math.log2(2 * n + 2 * max_lag + 2)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(A * np.conj(B), nfft)
    # cc[k] = sum a[n] * b[n-k]  (lag k means b is delayed by k relative to a)
    pos = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)

    # local energies for the overlapping part at each lag
    ca = np.concatenate([[0.0], np.cumsum(a * a)])
    cb = np.concatenate([[0.0], np.cumsum(b * b)])
    ea = np.empty_like(pos, dtype=np.float64)
    eb = np.empty_like(pos, dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag >= 0:
            # overlap: a[lag:n], b[0:n-lag]
            ea[i] = ca[n] - ca[lag]
            eb[i] = cb[n - lag]
            # cc indexing for positive lag: sum a[n+lag]*b[n] == cc[-lag]
        else:
            k = -lag
            ea[i] = ca[n - k]
            eb[i] = cb[n] - cb[k]
    r = pos / np.sqrt(np.maximum(ea * eb, EPS))
    return lags, r


def onset_envelope(x: np.ndarray, sr: int, hop: int = 256,
                   n_fft: int = 1024) -> np.ndarray:
    """Spectral-flux onset strength envelope, z-scored.

    Correlating onsets instead of waveforms gives sharp, unambiguous peaks on
    musical material, where a waveform correlation is broad and can lock onto
    the beat period instead of the true offset.
    """
    mono = to_mono(x)
    if mono.shape[0] < n_fft * 2:
        n_fft = max(64, int(2 ** int(math.log2(max(mono.shape[0] // 2, 64)))))
    f, t, Z = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          window="hann", boundary=None, padded=False)
    mag = np.abs(Z)
    flux = np.diff(mag, axis=1, prepend=mag[:, :1])
    env = np.maximum(flux, 0.0).sum(axis=0)
    env = env - env.mean()
    std = float(env.std())
    return env / std if std > 1e-9 else env


def _xcorr_envelope(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two envelopes, overlapping region only."""
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    nfft = 1 << int(math.ceil(math.log2(2 * n + 2 * max_lag + 2)))
    cc = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    pos = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    ca = np.concatenate([[0.0], np.cumsum(a * a)])
    cb = np.concatenate([[0.0], np.cumsum(b * b)])
    ea = np.empty_like(pos)
    eb = np.empty_like(pos)
    for i, lag in enumerate(lags):
        if lag >= 0:
            ea[i] = ca[n] - ca[lag]
            eb[i] = cb[n - lag]
        else:
            k = -lag
            ea[i] = ca[n - k]
            eb[i] = cb[n] - cb[k]
    return lags, pos / np.sqrt(np.maximum(ea * eb, EPS))


def estimate_lag(orig: np.ndarray, ref: np.ndarray, sr: int,
                 max_lag_ms: float = 400.0) -> dict:
    """Estimate the delay of ``ref`` relative to ``orig``, in samples.

    The waveform correlation is always maximised over the whole requested range
    (a coarse envelope estimate is reported alongside as an independent check).
    Restricting a *single* pass to a narrow window is avoided on purpose: on
    repetitive material a narrow window can lock onto the beat period.
    """
    a = _prepare_alignment_mono(orig, sr)
    b = _prepare_alignment_mono(ref, sr)
    max_lag = int(sr * max_lag_ms / 1000.0)
    max_lag = int(np.clip(max_lag, 1, max(1, len(a) // 4)))
    lags, r = normalized_xcorr(a, b, max_lag)
    rabs = np.abs(r)
    k = int(np.argmax(rabs))
    lag = -int(lags[k])
    conf = float(r[k])
    rr = rabs
    if 0 < k < len(rr) - 1:
        y0, y1, y2 = rr[k - 1], rr[k], rr[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag_frac = lag + float(np.clip(delta, -1.0, 1.0))

    # independent cross-check on onset envelopes
    hop = 256
    env_a = onset_envelope(orig, sr, hop)
    env_b = onset_envelope(ref, sr, hop)
    elags, ec = _xcorr_envelope(env_a, env_b, max(1, max_lag // hop))
    ke = int(np.argmax(np.abs(ec)))
    onset_lag = -int(elags[ke]) * hop
    onset_peak = float(np.abs(ec[ke]))
    guard_e = 2
    mask_e = np.abs(elags - elags[ke]) > guard_e
    second_e = float(np.max(np.abs(ec[mask_e]))) if mask_e.any() else 0.0
    onset_ratio = second_e / max(onset_peak, 1e-9)

    # ambiguity: another lag is essentially as good
    guard = max(2, int(sr * 0.002))
    mask = np.abs(lags - lags[k]) > guard
    second = float(np.max(rabs[mask])) if mask.any() else 0.0
    peak_ratio = second / max(float(rabs[k]), 1e-9)
    ambiguous = bool(conf < 0.6 or (peak_ratio > 0.97 and conf < 0.99))
    return {
        "lag_samples": lag,
        "lag_samples_fractional": lag_frac,
        "lag_ms": lag_frac * 1000.0 / sr,
        "correlation": conf,
        "onset_lag_samples": onset_lag,
        "onset_correlation": onset_peak,
        "onset_peak_ratio": onset_ratio,
        "onset_agrees": abs(onset_lag - lag) <= max(hop, int(0.002 * sr)),
        "second_peak_ratio": peak_ratio,
        "max_lag_ms_searched": max_lag_ms,
        "ambiguous": ambiguous,
        "note": ("a second lag correlates almost as strongly; verify the offset before "
                 "trusting the spectral alignment" if ambiguous
                 else "single dominant alignment"),
    }


def segment_lags(orig: np.ndarray, ref: np.ndarray, sr: int, segments: int = 8,
                 max_lag_ms: float = 30.0) -> dict:
    """Per-segment lag: distinguishes a constant offset from real drift."""
    n = min(orig.shape[0], ref.shape[0])
    seg = n // segments
    out = []
    for i in range(segments):
        s = i * seg
        e = s + seg
        a = _prepare_alignment_mono(orig[s:e], sr)
        b = _prepare_alignment_mono(ref[s:e], sr)
        max_lag = int(sr * max_lag_ms / 1000.0)
        lags, r = normalized_xcorr(a, b, max_lag)
        k = int(np.argmax(np.abs(r)))
        out.append({"start_s": s / sr, "lag_samples": -int(lags[k]),
                    "correlation": float(r[k])})
    lags_arr = np.array([o["lag_samples"] for o in out], dtype=float)
    times = np.array([o["start_s"] * sr for o in out], dtype=float)
    slope = 0.0
    if len(lags_arr) > 2 and np.ptp(times) > 0:
        slope = float(np.polyfit(times, lags_arr, 1)[0])
    return {
        "segments": out,
        "lag_span_samples": float(np.ptp(lags_arr)),
        "slope_samples_per_sample": slope,
        "drift_ppm": slope * 1e6,
        "consistent_offset": float(np.ptp(lags_arr)) <= 2.0,
    }


def _log_spectrum(x: np.ndarray, sr: int, f_lo: float = 60.0, f_hi: float = 3000.0,
                  points: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Long-term power spectrum sampled on a log-frequency grid (dB)."""
    f, p = power_spectrum(x, sr, n_fft=16384)
    grid = np.geomspace(f_lo, min(f_hi, sr / 2 * 0.98), points)
    # interpolate the PSD on a log axis, then smooth slightly (1/12 octave)
    vals = np.interp(np.log(grid), np.log(np.maximum(f, 1e-6)),
                     np.log10(np.maximum(p, 1e-30)))
    smooth = np.convolve(vals, np.ones(5) / 5.0, mode="same")
    return grid, smooth * 10.0


def fine_tuning_ratio(orig: np.ndarray, ref: np.ndarray, sr: int,
                      f_lo: float = 60.0, f_hi: float = 3000.0,
                      max_cents: float = 200.0) -> dict:
    """Estimate the pitch ratio between two files from their spectra.

    The reference's log-spectrum is rescaled by a candidate ratio and compared
    with the original's; the ratio that maximises the correlation wins.  This
    measures tuning directly (not chroma), so a real pitch shift or a
    time-stretch shows up as a ratio away from 1.0, while identical tuning
    returns ~1.000 with a high correlation.
    """
    grid, a = _log_spectrum(orig, sr, f_lo, f_hi)
    _, b = _log_spectrum(ref, sr, f_lo, f_hi)
    a = a - a.mean()
    b = b - b.mean()

    lo = 1.0 - max_cents / 1200.0 / 1.0
    ratios = 1.0 + np.linspace(-max_cents, max_cents, 401) / 1200.0
    scores = np.empty(ratios.size)
    log_grid = np.log(grid)
    for i, r in enumerate(ratios):
        # a feature at frequency f in the original appears at f*r in the reference
        shifted = np.interp(log_grid - np.log(r), log_grid, b)
        shifted = shifted - shifted.mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(shifted * shifted))
        scores[i] = float(np.sum(a * shifted) / denom) if denom > 0 else 0.0

    k = int(np.argmax(scores))
    # parabolic refinement around the best ratio
    if 0 < k < scores.size - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        step = ratios[1] - ratios[0]
        best = float(ratios[k] + np.clip(delta, -1, 1) * step)
    else:
        best = float(ratios[k])
    cents = 1200.0 * math.log2(best)
    return {
        "ratio": best,
        "cents": cents,
        "correlation_at_best": float(scores[k]),
        "correlation_at_unity": float(scores[np.argmin(np.abs(ratios - 1.0))]),
        "pairs": int(grid.size),
        "confidence": float(np.clip((scores[k] - 0.5) * 2.0, 0.0, 1.0)),
        "searched_cents": float(max_cents),
    }


# ---------------------------------------------------------------------------
# level-stratified spectral comparison
# ---------------------------------------------------------------------------


@dataclass
class FrameSpectra:
    freqs: np.ndarray
    power: np.ndarray      # (frames, bins)
    levels_db: np.ndarray  # (frames,)


def frame_spectra(x: np.ndarray, sr: int, n_fft: int = 4096,
                  hop: int = 2048) -> FrameSpectra:
    mono = to_mono(x)
    f, t, Z = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          window="hann", boundary=None, padded=False)
    # store as (frames, bins) - scipy returns (bins, frames)
    pw = np.abs(Z).T ** 2
    # frame level from the time-domain block (comparable across files)
    win = np.hanning(n_fft)
    idx = (t * sr).astype(int)
    levels = []
    for i0 in idx:
        seg = mono[i0:i0 + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        levels.append(20 * math.log10(max(np.sqrt(np.mean((seg * win) ** 2)), 1e-9)))
    return FrameSpectra(freqs=f, power=pw, levels_db=np.array(levels))


def _align_by_lag(x: np.ndarray, lag: int) -> np.ndarray:
    """Advance ``x`` by ``lag`` samples (positive = move content earlier).

    ``lag`` is the measured delay of the reference relative to the original, so
    advancing the reference by that amount lines the two files up.  Samples that
    would be revealed at the end are zero-filled; nothing is stretched.
    """
    if lag == 0:
        return x
    if lag > 0:
        return np.concatenate(
            [x[lag:], np.zeros((min(lag, x.shape[0]),) + x.shape[1:], dtype=x.dtype)],
            axis=0)[: x.shape[0]]
    k = -lag
    pad = np.zeros((min(k, x.shape[0]),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([pad, x], axis=0)[: x.shape[0]]


def level_stratified_transfer(orig: np.ndarray, ref: np.ndarray, sr: int,
                              lag: int = 0, n_fft: int = 4096,
                              edge_trim: int = 2) -> dict:
    """Per-level spectral transfer, in dB.

    For each loudness quartile of the *original* the mean band energies of both
    files are compared.  Two columns are produced:

    ``bands``
        Absolute band ratio ref/orig - includes the overall gain difference.
    ``relative``
        The same ratio with the mean across bands removed.  This is the shape of
        the transfer and is independent of overall gain, so a broadband
        compressor (which only shifts everything by one gain value) leaves it
        alone while a static filter gives a fixed pattern.

    ``level_dependence_db`` is the loudest-minus-quietest difference of
    ``relative``: near zero means the change is static, a systematic offset means
    the process is dynamic or nonlinear (dynamic EQ, band-dependent compression,
    saturation that only bites when loud).
    """
    ref_s = _align_by_lag(ref, lag)
    n = min(orig.shape[0], ref_s.shape[0])
    fo = frame_spectra(orig[:n], sr, n_fft)
    fr = frame_spectra(ref_s[:n], sr, n_fft)
    m = min(fo.power.shape[0], fr.power.shape[0]) - edge_trim
    if m < 8:
        return {"ok": False, "reason": "too little audio for stratified analysis"}
    po = fo.power[:m]
    pr = fr.power[:m]
    lv = fo.levels_db[:m]
    freqs = fo.freqs
    bands = [(40, 80), (80, 160), (160, 320), (320, 640), (640, 1280),
             (1280, 2560), (2560, 5120), (5120, 10240), (10240, 18000)]
    keys = [f"{lo}-{hi}" for lo, hi in bands]
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in bands]
    to = po.sum(axis=1)
    tr = pr.sum(axis=1)
    # ignore frames that are essentially silence
    usable = (to > 1e-7) & (tr > 1e-7)

    q = np.percentile(lv, [0, 25, 50, 75, 100])
    edges = [q[0] - 100, q[1], q[2], q[3], q[4] + 100]
    labels = ["quietest 25%", "low-mid", "high-mid", "loudest 25%"]
    out: dict = {"ok": True, "band_labels": keys, "level_bins": [],
                 "band_ratio_db": {}, "band_relative_db": {},
                 "level_dependence_db": {}, "max_level_dependence_db": 0.0,
                 "verdict": "unknown"}
    for i, label in enumerate(labels):
        sel = (lv >= edges[i]) & (lv <= edges[i + 1]) & usable
        if sel.sum() < 4:
            sel = (lv >= edges[i]) & (lv <= edges[i + 1])
        if sel.sum() < 2:
            continue
        # content-weighted band ratios for this loudness range
        ratios = []
        for msk in masks:
            a = po[sel][:, msk].sum()
            b = pr[sel][:, msk].sum()
            ratios.append(10 * math.log10(max(b, EPS) / max(a, EPS)))
        ratios = np.array(ratios)
        relative = ratios - ratios.mean()
        entry = {"label": label, "frames": int(sel.sum()),
                 "mean_level_db": float(lv[sel].mean()),
                 "bands": {k: float(v) for k, v in zip(keys, ratios)},
                 "relative": {k: float(v) for k, v in zip(keys, relative)},
                 "overall_gain_db": float(ratios.mean())}
        out["level_bins"].append(entry)

    if len(out["level_bins"]) >= 2:
        out["band_ratio_db"] = {k: float(np.mean([e["bands"][k] for e in out["level_bins"]]))
                                for k in keys}
        out["band_relative_db"] = {k: float(np.mean([e["relative"][k]
                                                     for e in out["level_bins"]]))
                                   for k in keys}
        dep = {k: float(out["level_bins"][-1]["relative"][k]
                        - out["level_bins"][0]["relative"][k]) for k in keys}
        out["level_dependence_db"] = dep
        out["max_level_dependence_db"] = float(max(abs(v) for v in dep.values()))

        # frame-by-frame test: does the high-band share change with loudness?
        # (fast dynamic behaviour can average out inside a loudness quartile, so
        # this regression is more sensitive than the quartile table)
        hf_mask = (freqs >= 2500) & (freqs < 11000)
        band_sums = np.stack([po[:, msk].sum(axis=1) for msk in masks], axis=1)
        band_sums_r = np.stack([pr[:, msk].sum(axis=1) for msk in masks], axis=1)
        tot = band_sums.sum(axis=1)
        tot_r = band_sums_r.sum(axis=1)
        hf_idx = [j for j, msk in enumerate(masks) if msk is hf_mask] or None
        ks = keys.index("2560-5120"), keys.index("5120-10240")
        hf_o = band_sums[:, list(ks)].sum(axis=1)
        hf_r = band_sums_r[:, list(ks)].sum(axis=1)
        rel = (10 * np.log10(np.maximum(hf_r, EPS) / np.maximum(tot_r, EPS)) -
               10 * np.log10(np.maximum(hf_o, EPS) / np.maximum(tot, EPS)))
        ok = usable & np.isfinite(rel)
        if ok.sum() > 20:
            lvv, rr = lv[ok], rel[ok]
            if np.std(lvv) > 0.5 and np.std(rr) > 1e-3:
                corr = float(np.corrcoef(lvv, rr)[0, 1])
                slope = float(np.polyfit(lvv, rr, 1)[0])
            else:
                corr, slope = 0.0, 0.0
            out["hf_level_correlation"] = corr
            out["hf_level_slope_db_per_db"] = slope
            dynamic_by_frame = abs(corr) > 0.3 and abs(slope) > 0.2
        else:
            out["hf_level_correlation"] = 0.0
            out["hf_level_slope_db_per_db"] = 0.0
            dynamic_by_frame = False
        # the quartile comparison is the primary criterion; the frame-level
        # regression is reported for context but is content-confounded (loud
        # frames are often brighter simply because of what is played)
        out["level_dependent"] = bool(out["max_level_dependence_db"] > 2.0)
        out["hf_regression_dynamic"] = bool(dynamic_by_frame)
        out["verdict"] = ("level-dependent (dynamic or nonlinear)"
                          if out["level_dependent"] else "mostly static")
    return out


def smooth_curve(freqs: np.ndarray, curve_db: np.ndarray,
                 fraction: float = 1 / 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-octave smoothing of a curve (default 1/3 octave)."""
    logf = np.log2(np.maximum(freqs, 1.0))
    out = np.empty_like(curve_db)
    half = fraction / 2.0
    for i, lf in enumerate(logf):
        sel = (logf >= lf - half) & (logf <= lf + half)
        out[i] = np.mean(curve_db[sel]) if sel.any() else curve_db[i]
    return freqs, out


def apparent_transfer_curve(orig: np.ndarray, ref: np.ndarray, sr: int,
                            lag: int = 0, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Octave-smoothed long-term magnitude ratio (dB) after alignment."""
    ref_s = _align_by_lag(ref, lag)
    f_o, p_o = power_spectrum(orig, sr, n_fft)
    f_r, p_r = power_spectrum(ref_s, sr, n_fft)
    n = min(len(p_o), len(p_r))
    f = f_o[:n]
    ratio = 10 * np.log10(np.maximum(p_r[:n], EPS) / np.maximum(p_o[:n], EPS))
    return smooth_curve(f, ratio, 1 / 3.0)


# ---------------------------------------------------------------------------
# modulation and dynamics character
# ---------------------------------------------------------------------------


def envelope_modulation(x: np.ndarray, sr: int, band=(60.0, 6000.0),
                        max_hz: float = 30.0) -> dict:
    """Peaks in the envelope spectrum (rhythm vs. an independent tremolo)."""
    mono = to_mono(x)
    sos = signal.butter(4, [band[0] / (sr / 2), min(band[1] / (sr / 2), 0.95)],
                        btype="band", output="sos")
    y = signal.sosfilt(sos, mono)
    env = np.abs(signal.hilbert(y))
    # decimate the envelope to 200 Hz
    target = 200.0
    step = max(1, int(sr // target))
    env = env[::step]
    fs = sr / step
    env = env - env.mean()
    w = np.hanning(env.size)
    spec = np.abs(np.fft.rfft(env * w)) ** 2
    f = np.fft.rfftfreq(env.size, 1 / fs)
    sel = (f > 0.2) & (f <= max_hz)
    f, spec = f[sel], spec[sel]
    if spec.size < 8:
        return {"ok": False}
    thr = np.percentile(spec, 97)
    peaks = []
    for i in np.where(spec >= thr)[0]:
        lo, hi = max(0, i - 3), min(len(spec), i + 4)
        if spec[i] == spec[lo:hi].max():
            peaks.append({"freq_hz": float(f[i]),
                          "level_db": float(10 * math.log10(max(spec[i], EPS)))})
    peaks.sort(key=lambda p: -p["level_db"])
    return {"ok": True, "peaks": peaks[:8]}


def short_term_level_relation(orig: np.ndarray, ref: np.ndarray, sr: int,
                              lag: int = 0, window_ms: float = 200.0) -> dict:
    """How much the reference's level follows the original's (compression signature)."""
    ref_s = _align_by_lag(ref, lag)
    n = min(orig.shape[0], ref_s.shape[0])
    a = to_mono(orig[:n])
    b = to_mono(ref_s[:n])
    win = int(sr * window_ms / 1000.0)
    hop = max(1, win // 2)
    cnt = 1 + (n - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(cnt)[:, None]
    la = 20 * np.log10(np.maximum(np.sqrt(np.mean(a[idx] ** 2, axis=1)), 1e-9))
    lb = 20 * np.log10(np.maximum(np.sqrt(np.mean(b[idx] ** 2, axis=1)), 1e-9))
    ok = (la > -70) & (lb > -70)
    if ok.sum() < 8:
        return {"ok": False}
    la, lb = la[ok], lb[ok]
    slope = float(np.polyfit(la, lb, 1)[0])
    corr = float(np.corrcoef(la, lb)[0, 1])
    return {
        "ok": True,
        "level_slope": slope,             # < 1 => reference is more compressed
        "implied_ratio": float(1.0 / slope) if slope > 1e-3 else float("inf"),
        "level_correlation": corr,
        "orig_range_db": float(np.percentile(la, 95) - np.percentile(la, 5)),
        "ref_range_db": float(np.percentile(lb, 95) - np.percentile(lb, 5)),
    }


# ---------------------------------------------------------------------------
# top-level pair report
# ---------------------------------------------------------------------------


def analyze_pair(orig_path: str, ref_path: str, *, verbose: bool = False) -> dict:
    orig = load_audio(orig_path)
    ref = load_audio(ref_path)
    sr = orig.sample_rate
    if ref.sample_rate != sr:
        raise ValueError(
            f"Sample rates differ ({orig.sample_rate} vs {ref.sample_rate}); resample "
            "the reference to the original rate before analysis.")

    # same-duration files cannot be shifted far; a wide search only invites a
    # false lock on periodic material
    dur_gap_ms = abs(orig.duration - ref.duration) * 1000.0
    search_ms = 25.0 if dur_gap_ms < 5.0 else min(400.0, dur_gap_ms * 4 + 25.0)
    lag = estimate_lag(orig.samples, ref.samples, sr, max_lag_ms=search_ms)
    seg = segment_lags(orig.samples, ref.samples, sr)
    tuning = fine_tuning_ratio(orig.samples, ref.samples, sr)
    lag_n = int(lag["lag_samples"])
    ref_s = _align_by_lag(ref.samples, lag_n)

    s_o = measure(orig.samples, sr)
    s_r = measure(ref.samples, sr)
    s_r_matched = None
    match_db = s_o.lufs - s_r.lufs
    scale = 10 ** (match_db / 20.0)
    s_r_matched = measure(ref.samples * scale, sr)

    transfer_f, transfer_db = apparent_transfer_curve(orig.samples, ref.samples, sr, lag_n)
    strat = level_stratified_transfer(orig.samples, ref.samples, sr, lag_n)
    mod_o = envelope_modulation(orig.samples, sr)
    mod_r = envelope_modulation(ref.samples, sr)
    rel = short_term_level_relation(orig.samples, ref.samples, sr, lag_n)

    report = {
        "files": {
            "original": orig.info(),
            "reference": ref.info(),
        },
        "geometry": {
            "same_sample_rate": orig.sample_rate == ref.sample_rate,
            "same_channels": orig.channels == ref.channels,
            "frame_delta": ref.frames - orig.frames,
            "duration_delta_s": ref.duration - orig.duration,
            "channels_original": orig.channels,
            "channels_reference": ref.channels,
        },
        "alignment": {**lag, "segments": seg},
        "tuning": tuning,
        "levels_original": s_o.to_dict(),
        "levels_reference": s_r.to_dict(),
        "levels_reference_matched": s_r_matched.to_dict(),
        "level_match_gain_db": float(match_db),
        "band_delta_db_at_match": {
            k: s_r_matched.bands[k] - s_o.bands[k] for k in s_o.bands
        },
        "transfer_curve": {"freqs_hz": transfer_f.tolist(),
                           "db": transfer_db.tolist()},
        "level_stratified": strat,
        "modulation_original": mod_o,
        "modulation_reference": mod_r,
        "level_relation": rel,
        "same_recording_verdict": _same_recording_verdict(orig, ref, lag, seg, tuning),
    }
    return report


def _same_recording_verdict(orig: AudioData, ref: AudioData, lag: dict,
                            seg: dict, tuning: dict) -> dict:
    checks = {
        "duration_matches_within_100ms": abs(orig.duration - ref.duration) < 0.1,
        "strong_normalised_correlation": lag["correlation"] >= 0.55,
        "no_large_drift": abs(seg["drift_ppm"]) < 500.0,
        "tuning_consistent": abs(tuning.get("cents", 0.0)) < 15.0
        or tuning.get("confidence", 0.0) < 0.3,
    }
    return {
        "checks": checks,
        "verdict": "same recording" if sum(checks.values()) >= 3 else "needs review",
        "note": "Duration alone is never used as proof; correlation is measured at "
                "the corrected lag and drift is checked per segment.",
    }


# ---------------------------------------------------------------------------
# candidate scoring (used by tools/compare_methods.py)
# ---------------------------------------------------------------------------


def _band_vector(stats: Stats) -> np.ndarray:
    return np.array([stats.bands[k] for k in stats.bands], dtype=float)


def score_candidate(candidate: np.ndarray, sr: int, orig: np.ndarray,
                    ref_stats: Stats, orig_stats: Stats) -> dict:
    """Gain-invariant similarity score against the reference (lower is better)."""
    cand = measure(candidate, sr)
    if cand.lufs <= -69 or not np.isfinite(cand.lufs):
        return {"valid": False}
    gain = orig_stats.lufs - cand.lufs
    matched = measure(candidate * 10 ** (gain / 20.0), sr)
    ref_b = _band_vector(ref_stats) - ref_stats.lufs
    m_b = _band_vector(matched) - matched.lufs
    o_b = _band_vector(orig_stats) - orig_stats.lufs
    target_delta = ref_b - o_b
    got_delta = m_b - o_b
    return {
        "valid": True,
        "band_rmse_db": float(np.sqrt(np.mean((got_delta - target_delta) ** 2))),
        "centroid_ratio_error": float(abs(math.log2(max(matched.centroid_hz, 1) /
                                                    max(ref_stats.centroid_hz, 1)))),
        "correlation_error": float(abs(matched.correlation - ref_stats.correlation)),
        "crest_error_db": float(abs(matched.crest_db - ref_stats.crest_db)),
        "envelope_range_error_db": float(abs(matched.envelope_range_db
                                             - ref_stats.envelope_range_db)),
        "side_mid_error_db": float(abs(matched.side_mid_db - ref_stats.side_mid_db)),
        "flatness_error": float(abs(matched.flatness - ref_stats.flatness)),
        "stats": matched.to_dict(),
    }
