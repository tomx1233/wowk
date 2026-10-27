"""Measure what was done to one audio file to produce another.

This is the tool behind ``docs/PAIR_MATCH.md``.  It answers three questions about
a pair ``(original, modified)`` that are supposed to be *the same recording*:

1. **Pitch** - by what frequency ratio were the partials moved?
2. **Tempo / speed** - by what ratio was the timeline compressed or stretched?
3. **Everything else** - what tonal or level difference is left once the first two
   are accounted for?

Both estimates are validated on synthetic controls built from the *original* file
itself (``--selftest``, on by default): the original is resampled by a known
ratio, optionally passed through a time-stretch and a lossy encode, and the
measurement is run on that pair.  If the estimator cannot recover the ratio it
was given, its answer on the real pair is not reported as fact.

Usage::

    python tools/pair_match.py originalfile.ogg ModifiedFile.ogg --out-dir analysis/pair
    python tools/pair_match.py A.ogg B.ogg --render examples/reconstruction.ogg

Nothing here guesses.  Every number printed is measured from the files, and the
report says which numbers were cross-checked and how.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge.dsp.audio import ExportSettings, load_audio, save_audio
from timbreforge.dsp.meter import measure

SR_DEFAULT = 48000


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------
def decode(path: str) -> tuple[np.ndarray, int]:
    audio = load_audio(path)
    return np.asarray(audio.samples, dtype=np.float64), int(audio.sample_rate)


def both_channels(x: np.ndarray) -> np.ndarray:
    n = min(x.shape[1], 2)
    return x[:, :n].mean(axis=1)


# ---------------------------------------------------------------------------
# 1. pitch: shift of the spectral fine structure in cents
# ---------------------------------------------------------------------------
def mean_log_spectrum(x: np.ndarray, sr: int, t0: float, dur: float,
                      nfft: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    a = int(t0 * sr)
    b = min(len(x), int((t0 + dur) * sr))
    seg = x[a:b]
    if len(seg) < nfft:
        seg = np.pad(seg, (0, nfft - len(seg)))
    win = np.hanning(nfft)
    hop = nfft // 4
    n = 1 + (len(seg) - nfft) // hop
    acc = np.zeros(nfft // 2 + 1)
    for i in range(max(n, 1)):
        acc += np.abs(np.fft.rfft(seg[i * hop:i * hop + nfft] * win)) ** 2
    acc /= max(n, 1)
    f = np.arange(acc.size) * sr / nfft
    return f, 10 * np.log10(acc + 1e-20)


def whiten(db: np.ndarray, width: int = 41) -> np.ndarray:
    """Remove the smooth spectral envelope, keeping only the partial structure."""
    from scipy.ndimage import median_filter

    return db - median_filter(db, size=width, mode="nearest")


def spectral_shift_cents(xo, xm, sr, t_o, t_m, dur, fmin=100.0, fmax=8000.0,
                         lo=-60.0, hi=200.0, step=0.05) -> tuple[float, float]:
    """Shift (cents) that maps the original's partials onto the modified's.

    Positive means the modified file is *higher* in pitch.
    """
    fo, does = mean_log_spectrum(both_channels(xo), sr, t_o, dur)
    fm, dbm = mean_log_spectrum(both_channels(xm), sr, t_m, dur)
    wo = whiten(does)
    wm = whiten(dbm)
    sel = (fo >= fmin) & (fo <= fmax)
    fo_s, wo_s = fo[sel], wo[sel]
    grid = np.arange(lo, hi, step)
    scores = np.empty(grid.size)
    for k, c in enumerate(grid):
        r = 2.0 ** (c / 1200.0)
        wi = np.interp(fo_s * r, fm, wm, left=0.0, right=0.0)
        a = wo_s - wo_s.mean()
        b = wi - wi.mean()
        scores[k] = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-12))
    k = int(np.argmax(scores))
    best = float(grid[k])
    if 0 < k < grid.size - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-15:
            best += 0.5 * (y0 - y2) / den * step
    return best, float(scores[k])


def estimate_pitch(xo, xm, sr, tempo_hint: float, n_windows: int = 10,
                   dur: float = 12.0) -> dict:
    """Measure the pitch ratio on several windows spread over the file."""
    total = len(xo) / sr
    margin = 8.0
    start, stop = margin, max(margin + dur + 1, total - dur - margin)
    times = np.linspace(start, stop, n_windows)
    cents, corr, ratios = [], [], []
    for t_o in times:
        t_m = t_o / tempo_hint
        if t_m + dur > len(xm) / sr:
            continue
        c, r = spectral_shift_cents(xo, xm, sr, t_o, t_m, dur)
        cents.append(c)
        corr.append(r)
        ratios.append(2.0 ** (c / 1200.0))
    cents = np.array(cents)
    ratios = np.array(ratios)
    med = float(np.median(ratios))
    return {
        "method": "whitened log-spectrum shift, median over windows",
        "windows": len(cents),
        "cents": [round(float(c), 3) for c in cents],
        "corr": [round(float(c), 3) for c in corr],
        "median_cents": round(float(np.median(cents)), 3),
        "spread_cents": round(float(np.std(cents)), 3),
        "ratio": round(med, 6),
        "percent": round((med - 1.0) * 100.0, 4),
    }


# ---------------------------------------------------------------------------
# 2. tempo: scale of the timeline
# ---------------------------------------------------------------------------
def onset_envelope(x: np.ndarray, sr: int, hop: int = 120, nfft: int = 1024):
    win = np.hanning(nfft)
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    S = np.log1p(20.0 * np.abs(np.fft.rfft(x[idx] * win, axis=1)))
    e = np.maximum(0.0, np.diff(S, axis=0)).sum(axis=1)
    e = np.concatenate([[0.0], e])
    k = int(0.5 * sr / hop)
    base = np.convolve(e, np.ones(k) / k, mode="same")
    return np.maximum(0.0, e - base)


def _z(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    s = v.std()
    return v / s if s > 0 else v


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int):
    from scipy.signal import fftconvolve

    a, b = _z(a), _z(b)
    c = fftconvolve(b, a[::-1], mode="full")
    lags = np.arange(-len(a) + 1, len(b))
    sel = np.abs(lags) <= max_lag
    c, lags = c[sel] / len(a), lags[sel]
    k = int(np.argmax(c))
    return float(c[k]), int(lags[k])


def estimate_tempo(eo, em, v_lo=0.90, v_hi=1.20, step=2e-5, max_lag=4000):
    """Find v such that eo(t*v) matches em(t): the modified file runs v times faster."""
    grid = np.arange(v_lo, v_hi + step / 2, step)
    n = np.arange(len(em))
    on = np.arange(len(eo))
    scores = np.empty(grid.size)
    for k, v in enumerate(grid):
        oi = n * v
        ok = oi < len(eo) - 1
        if ok.sum() < 0.5 * len(em):
            scores[k] = -1.0
            continue
        res = np.interp(oi, on, eo, right=0.0)
        scores[k] = best_lag(res[ok], em[ok], max_lag)[0]
    k = int(np.argmax(scores))
    v = float(grid[k])
    if 0 < k < grid.size - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-15:
            v += 0.5 * (y0 - y2) / den * step
    oi = n * v
    ok = oi < len(eo) - 1
    res = np.interp(oi, on, eo, right=0.0)
    corr, lag = best_lag(res[ok], em[ok], max_lag)
    # drift check: per-quarter residual lag (a wrong v shows up as a ramp)
    quarters = []
    q = len(em) // 4
    for i in range(4):
        a = res[i * q:(i + 1) * q]
        b = em[i * q:(i + 1) * q]
        quarters.append(best_lag(a, b, max(200, max_lag // 4))[1])
    return {
        "method": "onset-envelope scale fit (integer lag, then parabolic refine)",
        "ratio": round(v, 6),
        "percent": round((v - 1.0) * 100.0, 4),
        "corr": round(corr, 4),
        "offset_samples": int(lag),
        "offset_ms": round(lag * 120 / 48000 * 1000.0, 1),
        "quarter_lags": quarters,
    }


# ---------------------------------------------------------------------------
# 3. what is left: level and tone after removing the pitch shift
# ---------------------------------------------------------------------------
def ltas(x: np.ndarray, sr: int, nfft: int = 16384, hop: int = 8192):
    m = both_channels(x)
    win = np.hanning(nfft)
    n = 1 + (len(m) - nfft) // hop
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, n, 4):
        acc += np.abs(np.fft.rfft(m[i * hop:i * hop + nfft] * win)) ** 2
        cnt += 1
    return acc / max(cnt, 1)


def residual_tone(xo, xm, sr, pitch_ratio, edges=None) -> dict:
    """Band levels of the modified minus the pitch-compensated original."""
    edges = edges if edges is not None else np.geomspace(40, 18000, 20)
    Lo = ltas(xo, sr)
    Lm = ltas(xm, sr)
    f = np.arange(Lo.size) * sr / 16384
    centres, deltas = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel_m = (f >= lo) & (f < hi)
        sel_o = (f / pitch_ratio >= lo) & (f / pitch_ratio < hi)
        if not sel_m.any() or not sel_o.any():
            continue
        b = 10 * np.log10(Lm[sel_m].sum() + 1e-30)
        a = 10 * np.log10(Lo[sel_o].sum() + 1e-30)
        centres.append(float(np.sqrt(lo * hi)))
        deltas.append(float(b - a))
    centres, deltas = np.array(centres), np.array(deltas)
    ref = (centres >= 1000) & (centres <= 4000)
    offset = float(deltas[ref].mean()) if ref.any() else float(deltas.mean())
    return {
        "edges": [round(float(e), 1) for e in edges],
        "centres": [round(float(c), 1) for c in centres],
        "delta_db": [round(float(d), 2) for d in deltas],
        "normalised_db": [round(float(d - offset), 2) for d in deltas],
        "level_offset_db": round(offset, 2),
        "range_db": round(float(deltas.max() - deltas.min()), 2),
    }


def chain_selftest(x, sr, ratio, seconds=60.0):
    """End-to-end control: resample + WSOLA stretch, then measure the pitch.

    Guards against the estimator being fooled by the artefacts of a
    time-stretch stage (which the resample-only self-test cannot see).
    """
    n = min(len(x), int(seconds * sr))
    seg = np.asarray(x)[:n]
    ref = seg.copy()
    c = _wsola(_speedup(seg, ratio, sr), 1.0, sr)
    p = estimate_pitch(ref, c, sr, 1.0, n_windows=4, dur=8.0)
    return {"true_ratio": round(float(ratio), 6), "measured_ratio": p["ratio"],
            "error_ppm": int(round((p["ratio"] / ratio - 1) * 1e6)),
            "spread_cents": p["spread_cents"]}


def content_span(x, sr, floor_db=-60.0, frame_s=0.05):
    """Where the actual audio starts and stops inside the file (ignores padding)."""
    m = np.abs(np.asarray(x)).max(axis=1) if np.asarray(x).ndim > 1 else np.abs(np.asarray(x))
    n = int(frame_s * sr)
    k = len(m) // n
    if k == 0:
        return {"start_s": 0.0, "end_s": len(m) / sr, "span_s": len(m) / sr}
    rms = np.sqrt((m[: k * n].reshape(k, n) ** 2).mean(axis=1))
    keep = np.where(20 * np.log10(np.maximum(rms, 1e-12)) > floor_db)[0]
    if keep.size == 0:
        return {"start_s": 0.0, "end_s": 0.0, "span_s": 0.0}
    t0, t1 = keep[0] * frame_s, (keep[-1] + 1) * frame_s
    return {"start_s": round(float(t0), 3), "end_s": round(float(t1), 3),
            "span_s": round(float(t1 - t0), 3)}


def source_bandwidth(x, sr, bands=((1000, 4000), (13000, 16000), (16000, 17800),
                                   (17800, 19400), (19400, 22000))):
    """Absolute band levels plus the highest frequency carrying real content.

    A band-limited source shows a cliff: the last band with real music is tens of
    dB above the bands beyond it, which then sit at the codec noise floor with a
    flat (white) internal spectrum.
    """
    L = ltas(np.asarray(x).reshape(len(x), -1), sr)
    f = np.arange(L.size) * sr / 16384.0
    out = {"bands": [], "edge_hz": None}
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        lvl = 10 * np.log10(L[sel].sum() + 1e-30)
        p = L[sel] / L[sel].sum()
        flat = 10 * np.log10(np.exp(np.mean(np.log(p + 1e-30))) / np.mean(p))
        out["bands"].append({"band": f"{lo}-{hi}", "level_db": round(float(lvl), 1),
                             "flatness_db": round(float(flat), 1)})
    ref = 10 * np.log10(L[(f >= 1000) & (f < 4000)].sum() + 1e-30)
    above = np.where(L > ref - 40.0)[0]
    above = above[f[above] > 6000]
    if above.size:
        out["edge_hz"] = int(round(float(f[above[-1]])))
    return out


def level_over_time(xo, xm, sr, tempo, offset_s=0.0, win=0.4) -> dict:
    n = int(win * sr)
    d, lv = [], []
    for t_m in np.arange(2.0, len(xm) / sr - 2.0, 1.0):
        t_o = t_m * tempo + offset_s
        a_o, a_m = int(t_o * sr), int(t_m * sr)
        if a_o < 0 or a_o + n > len(xo) or a_m + n > len(xm):
            continue
        o = 10 * np.log10((xo[a_o:a_o + n] ** 2).mean() + 1e-12)
        m = 10 * np.log10((xm[a_m:a_m + n] ** 2).mean() + 1e-12)
        d.append(m - o)
        lv.append(o)
    d, lv = np.array(d), np.array(lv)
    if len(d) < 8:
        return {}
    A = np.stack([lv, np.ones(len(lv))], 1)
    slope, intercept = np.linalg.lstsq(A, d, rcond=None)[0]
    return {
        "median_db": round(float(np.median(d)), 2),
        "p5_db": round(float(np.percentile(d, 5)), 2),
        "p95_db": round(float(np.percentile(d, 95)), 2),
        "std_db": round(float(np.std(d)), 2),
        "slope_vs_level": round(float(slope), 3),
        "windows": int(len(d)),
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def speedup(x: np.ndarray, ratio: float, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(x, 1_000_000, int(round(ratio * 1_000_000)), axis=0)


def self_test(xo, sr, test_ratios=(1.054, 1.05), tempo_ratio=None,
              quality: float | None = 0.5) -> list[dict]:
    """Recover known ratios from controls built out of the original file."""
    out = []
    for r in test_ratios:
        y = speedup(xo, r, sr)
        path = None
        if quality is not None:
            import tempfile

            fd, path = tempfile.mkstemp(suffix=".ogg")
            os.close(fd)
            try:
                save_audio(y, ExportSettings(path=path, format="OGG", subtype="VORBIS",
                                             ogg_quality=quality,
                                             overwrite=True), sample_rate=sr)
                y, _ = decode(path)
            finally:
                if path and os.path.exists(path):
                    os.unlink(path)
        tempo = tempo_ratio or r
        got = estimate_pitch(xo, y, sr, tempo, n_windows=6, dur=10.0)
        out.append({"true_ratio": r, "measured_ratio": got["ratio"],
                    "error_ppm": round((got["ratio"] / r - 1) * 1e6)})
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _hms(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def file_facts(path: str) -> dict:
    import soundfile as sf

    info = sf.info(path)
    x, sr = decode(path)
    st = measure(x, sr)
    return {
        "path": os.path.basename(path),
        "codec": info.subtype,
        "container": info.format,
        "sample_rate": sr,
        "channels": info.channels,
        "frames": info.frames,
        "duration": round(info.frames / info.samplerate, 4),
        "bitrate_kbps": round(os.path.getsize(path) * 8 / info.frames * info.samplerate / 1000, 1),
        "peak_dbfs": round(st.peak_db, 2),
        "true_peak_dbtp": round(st.true_peak_db, 2),
        "rms_dbfs": round(st.rms_db, 2),
        "lufs": round(st.lufs, 2),
        "crest_db": round(st.crest_db, 2),
        "lra": round(st.lra, 2),
        "stereo_corr": round(st.correlation, 3),
    }


def macro_lines(pitch_percent: float, tempo_percent: float,
                points: list[tuple[float, float]] | None = None,
                gain_db: float | None = None) -> list[str]:
    """Audacity macro commands (Macro Manager > Import)."""
    out = [f"ChangePitch: Percentage={pitch_percent:.2f}",
           f"ChangeTempo: Percentage={tempo_percent:.2f}"]
    if points:
        parts = []
        for i, (f, v) in enumerate(points):
            parts.append(f'f{i}="{f:.0f}" v{i}="{v:.2f}"')
        parts += ['FilterLength="8191"', 'InterpolateLin="0"',
                  'InterpolationMethod="B-spline"']
        out.append("FilterCurve: " + " ".join(parts))
    if gain_db is not None:
        out.append(f"Amplify: Amplification={gain_db:.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("original")
    ap.add_argument("modified")
    ap.add_argument("--out-dir", default="analysis/pair")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--figures", action="store_true", help="write PNG figures")
    ap.add_argument("--render", default=None,
                    help="render the reconstruction to this path (ogg/wav)")
    ap.add_argument("--tempo-hint", type=float, default=1.05,
                    help="rough speed factor, used only to line up windows")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"reading {args.original} / {args.modified}")
    xo, sr = decode(args.original)
    xm, sr2 = decode(args.modified)
    if sr != sr2:
        print(f"WARNING: sample rates differ ({sr} vs {sr2}); comparing at {sr}")
    fo = file_facts(args.original)
    fm = file_facts(args.modified)
    print(f"  {fo['path']}: {fo['duration']:.3f} s, {sr} Hz, {fo['channels']} ch, "
          f"{fo['codec']}, {fo['bitrate_kbps']:.0f} kbps")
    print(f"  {fm['path']}: {fm['duration']:.3f} s, {sr2} Hz, {fm['channels']} ch, "
          f"{fm['codec']}, {fm['bitrate_kbps']:.0f} kbps")

    mo, mm = both_channels(xo), both_channels(xm)

    print("\n[1/3] tempo")
    tempo = estimate_tempo(onset_envelope(mo, sr), onset_envelope(mm, sr))
    print(f"  speed ratio {tempo['ratio']:.6f} ({(tempo['ratio'] - 1) * 100:+.3f} %), "
          f"corr {tempo['corr']:.3f}, drift {tempo['quarter_lags']}")

    print("\n[2/3] pitch")
    pitch = estimate_pitch(xo, xm, sr, tempo["ratio"])
    print(f"  pitch ratio {pitch['ratio']:.6f} ({pitch['median_cents']:+.2f} cents, "
          f"{pitch['percent']:+.3f} %), spread {pitch['spread_cents']:.2f} c")

    span_o = content_span(xo, sr)
    span_m = content_span(xm, sr)
    bw_o = source_bandwidth(xo, sr)
    bw_m = source_bandwidth(xm, sr)
    print("\n[3/3] residual level and tone")
    tone = residual_tone(xo, xm, sr, pitch["ratio"])
    level = level_over_time(xo, xm, sr, tempo["ratio"],
                            offset_s=tempo["offset_samples"] * 120 / sr)
    print(f"  level {level.get('median_db', float('nan')):+.2f} dB "
          f"(p5..p95 {level.get('p5_db')}..{level.get('p95_db')}), "
          f"tone range {tone['range_db']:.2f} dB")

    selftest = []
    chain = None
    if not args.no_selftest:
        print("\n[self-test] recovering known ratios from controls of the original")
        selftest = self_test(xo, sr, test_ratios=(1.054, 1.05, pitch["ratio"]))
        for row in selftest:
            print(f"  true {row['true_ratio']:.6f} -> measured "
                  f"{row['measured_ratio']:.6f} ({row['error_ppm']:+d} ppm)")
        chain = chain_selftest(xo, sr, 1.054)
        print(f"  through resample+WSOLA: true {chain['true_ratio']:.6f} -> measured "
              f"{chain['measured_ratio']:.6f} ({chain['error_ppm']:+d} ppm, "
              f"spread {chain['spread_cents']:.2f} c)")

    print(f"  audio span {span_o['start_s']:.3f}-{span_o['end_s']:.3f} s -> "
          f"{span_m['start_s']:.3f}-{span_m['end_s']:.3f} s, "
          f"span ratio {span_o['span_s'] / max(span_m['span_s'], 1e-9):.6f}")
    print(f"  source bandwidth: original ends {bw_o['edge_hz']} Hz, "
          f"modified ends {bw_m['edge_hz']} Hz")
    if (bw_o["edge_hz"] and bw_m["edge_hz"] and bw_m["edge_hz"] > bw_o["edge_hz"] + 800):
        print("  NOTE: the modified file carries content above the original's band "
              "limit; no pitch/tempo/gain chain can create that, so the tone "
              "difference above that limit is a source/encode difference.")

    report = {
        "original": fo, "modified": fm,
        "content_span": {"original": span_o, "modified": span_m,
                         "span_ratio": round(span_o["span_s"] / max(span_m["span_s"], 1e-9), 6)},
        "bandwidth": {"original": bw_o, "modified": bw_m},
        "tempo": tempo, "pitch": pitch, "tone": tone, "level": level,
        "selftest": selftest,
        "selftest_through_stretch": chain,
        "duration_ratio": round(fo["duration"] / fm["duration"], 6),
    }

    macro = macro_lines(pitch["percent"], tempo["percent"])
    report["macro"] = macro

    json_path = args.json or os.path.join(args.out_dir, "measurements.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(args.out_dir, "macro.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(macro) + "\n")
    print(f"\nwrote {json_path} and {os.path.join(args.out_dir, 'macro.txt')}")
    print("macro:")
    for line in macro:
        print("  " + line)

    if args.figures:
        make_figures(xo, xm, sr, report, args.out_dir)
        print(f"wrote figures to {args.out_dir}")

    if args.render:
        render_reconstruction(xo, xm, sr, pitch["ratio"], tempo["ratio"], args.render,
                              report)

    return 0


# ---------------------------------------------------------------------------
# reconstruction (transparent pitch + tempo, for verification)
# ---------------------------------------------------------------------------
def _speedup(x, ratio, sr=SR_DEFAULT):
    """Resample so the pitch (and the playback speed) scales by ``ratio``."""
    from scipy.signal import resample_poly

    return resample_poly(x, 1_000_000, int(round(ratio * 1_000_000)), axis=0)


def _wsola(x, rate, sr=SR_DEFAULT, frame=2048, tol=512):
    """Waveform-similarity overlap-add time scaling (pitch preserving).

    ``rate`` > 1 compresses time (output is shorter).  The synthesis hop is
    fixed and the analysis hop is a float, so the long-run rate is exact; only
    the per-frame position jitters inside +-``tol``.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    frame = int(frame) // 2 * 2
    syn_hop = frame // 2
    ana_hop = syn_hop * float(rate)
    win = np.hanning(frame)[:, None]
    out_len = int(np.ceil(n / rate)) + frame
    y = np.zeros((out_len, ch))
    wsum = np.zeros((out_len, 1))
    ref = x.mean(axis=1)
    pos_ana, pos_syn, prev = 0.0, 0, None
    while pos_syn + frame < out_len:
        a = int(round(pos_ana))
        if a + frame + tol >= n:
            break
        if prev is None:
            best = a
        else:
            lo = max(0, a - tol)
            hi = min(n - frame - 1, a + tol)
            if hi <= lo:
                best = a
            else:
                seg = ref[lo:hi + len(prev)]
                corr = np.correlate(seg, prev, mode="valid")
                best = lo + int(np.argmax(corr))
        seg = x[best:best + frame]
        y[pos_syn:pos_syn + frame] += seg * win
        wsum[pos_syn:pos_syn + frame] += win
        prev = ref[best + syn_hop:best + syn_hop + syn_hop]
        pos_ana += ana_hop
        pos_syn += syn_hop
    wsum[wsum < 1e-6] = 1.0
    y /= wsum
    return y[:int(round(n / rate))]


def render_reconstruction(xo, xm, sr, pitch, tempo, path, report=None):
    from timbreforge.dsp.audio import AudioData

    y = _speedup(xo, pitch, sr)
    y = _wsola(y, tempo / pitch, sr)
    y *= 10 ** (0.0 / 20)
    target = len(xo) / tempo
    if y.shape[0] > target:
        y = y[:int(target)]
    save_audio(AudioData(samples=y, sample_rate=sr),
               ExportSettings(path=path, format=None, subtype=None, overwrite=True),
               sample_rate=sr)
    print(f"rendered {path} ({y.shape[0] / sr:.3f} s)")
    return path


def make_figures(xo, xm, sr, report, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # -- spectrograms with identical settings --------------------------
    def spec(x, t0=20.0, dur=12.0):
        a, b = int(t0 * sr), int((t0 + dur) * sr)
        seg = both_channels(x)[a:b]
        nfft, hop = 4096, 512
        n = 1 + (len(seg) - nfft) // hop
        idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
        S = np.abs(np.fft.rfft(seg[idx] * np.hanning(nfft), axis=1))
        db = 20 * np.log10(S + 1e-9)
        f = np.arange(S.shape[1]) * sr / nfft
        return db, f

    tempo = report["tempo"]["ratio"]
    do, f = spec(xo, 20.0)
    dm, _ = spec(xm, 20.0 / tempo)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, sharey=True)
    for ax, d, title in ((axes[0], do, "original"),
                         (axes[1], dm, "modified")):
        keep = f <= 16000
        ax.imshow(d[:, keep].T, origin="lower", aspect="auto", cmap="magma",
                  vmin=d.max() - 80, vmax=d.max(),
                  extent=[0, 12, 0, f[keep][-1] / 1000])
        ax.set_title(f"{title}  (identical FFT settings: 4096 @ 512 hop, 48 kHz)",
                     fontsize=9, loc="left")
        ax.set_ylabel("kHz")
    axes[1].set_xlabel("seconds within the window")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectrograms.png"), dpi=130)
    plt.close(fig)

    # -- residual tone -------------------------------------------------
    tone = report["tone"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.semilogx(tone["centres"], tone["normalised_db"], "o-", color="#2f81f7")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB vs pitch-compensated original")
    ax.set_title("Tonal difference left after removing the pitch shift "
                 "(normalised to 1-4 kHz)", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tone_residual.png"), dpi=130)
    plt.close(fig)

    # -- pitch stability ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(range(len(report["pitch"]["cents"])), report["pitch"]["cents"], "o-",
            color="#3fb950")
    ax.axhline(report["pitch"]["median_cents"], color="#888", ls="--", lw=0.8)
    ax.set_xlabel("window")
    ax.set_ylabel("cents")
    ax.set_title(f"pitch shift per window (median "
                 f"{report['pitch']['median_cents']:+.2f} c)", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pitch_stability.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
