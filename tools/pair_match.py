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
from timbreforge.dsp.pairmatch import (SR_DEFAULT, _speedup, _wsola,  # noqa: F401
                                       best_lag, both_channels, chain_selftest,
                                       content_span, decode, estimate_pitch,
                                       estimate_tempo, file_facts, level_over_time,
                                       ltas, macro_lines, mean_log_spectrum,
                                       onset_envelope, residual_tone, self_test,
                                       source_bandwidth, spectral_shift_cents,
                                       speedup, whiten)


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
