#!/usr/bin/env python3
"""Analyse a processed/original reference pair and write a report.

Usage
-----
    python tools/analyze_pair.py --original TestSong.ogg --reference NewTest.ogg \
        --out-dir analysis/NewTest_vs_TestSong [--figures]

Writes ``report.txt``, ``report.json``, ``transfer.csv`` and, with ``--figures``,
PNG figures of the spectra, spectrograms and envelope/loudness behaviour.

Everything is measurement only: nothing is copied from the reference and no
processing profile is applied automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from timbreforge.dsp.analyze import analyze_pair
from timbreforge.dsp.audio import load_audio


def _figures(orig_path: str, ref_path: str, report: dict, out_dir: str,
             lag: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping figures: {exc})")
        return []

    from timbreforge.dsp.meter import envelope_db, power_spectrum, to_mono

    orig = load_audio(orig_path)
    ref = load_audio(ref_path)
    sr = orig.sample_rate
    made = []

    # level-match the reference so global gain does not dominate the figures
    gain_db = report["level_match_gain_db"]
    ref_matched = ref.samples * (10 ** (gain_db / 20.0))

    # 1 - long-term spectrum, level matched
    f_o, p_o = power_spectrum(orig.samples, sr)
    f_r, p_r = power_spectrum(ref.samples, sr)
    n = min(len(p_o), len(p_r))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.4]})
    ax[0].semilogx(f_o[:n], 10 * np.log10(np.maximum(p_o[:n], 1e-20)), label="original",
                   color="#4d5766", lw=1)
    ax[0].semilogx(f_r[:n], 10 * np.log10(np.maximum(p_r[:n], 1e-20)),
                   label="reference", color="#e5695b", lw=1)
    ax[0].set_ylabel("PSD (dB/Hz)")
    ax[0].set_title("Long-term spectrum (raw; the reference is louder overall)")
    ax[0].grid(True, which="both", alpha=0.2)
    ax[0].set_xlim(20, sr / 2)
    ax[0].legend()
    tc = report["transfer_curve"]
    ax[1].semilogx(tc["freqs_hz"], tc["db"], color="#57c8d8", lw=1.6)
    ax[1].axhline(0, color="#888", lw=0.8)
    ax[1].set_ylabel("ref/orig (dB)")
    ax[1].set_xlabel("Hz")
    ax[1].set_title("Apparent transfer after alignment (1/3-octave smoothed)")
    ax[1].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_dir, "spectrum_transfer.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    # 2 - spectrograms of representative + held-out passages
    dur = min(orig.duration, ref.duration)
    windows = [("intro 1-9 s", 1.0, min(9.0, dur - 1)),
               ("held-out 62-70 s", 62.0, min(70.0, dur - 1)),
               ("held-out 118-126 s", 118.0, min(126.0, dur - 1))]
    windows = [w for w in windows if w[2] > w[1] + 1]
    fig, axes = plt.subplots(len(windows), 2, figsize=(12, 3.1 * len(windows)))
    if len(windows) == 1:
        axes = axes[None, :]
    for row, (label, a, b) in enumerate(windows):
        for col, (name, sig) in enumerate((("original", orig.samples),
                                           ("reference", ref.samples))):
            seg = sig[int(a * sr):int(b * sr)]
            ax = axes[row, col]
            ax.specgram(to_mono(seg), NFFT=2048, Fs=sr, noverlap=1024,
                        cmap="magma", vmin=-120, vmax=-20)
            ax.set_title(f"{label} - {name}", fontsize=9)
            ax.set_ylim(0, 16000)
            if col == 0:
                ax.set_ylabel("Hz")
    fig.suptitle("Spectrograms, identical settings (dB, -120..-20)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "spectrograms.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    made.append(path)

    # 3 - level behaviour at matched gain
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    idx_o = np.arange(0, orig.frames, max(1, sr // 10))
    env_o = envelope_db(orig.samples, sr, 50.0, 100.0)
    env_r = envelope_db(ref_matched, sr, 50.0, 100.0)
    t_o = np.arange(env_o.size) * 0.1
    t_r = np.arange(env_r.size) * 0.1
    ax[0].plot(t_o, env_o, color="#7f8b9c", lw=0.7, label="original")
    ax[0].plot(t_r, env_r, color="#e5695b", lw=0.7, label="reference (level matched)")
    ax[0].set_ylabel("50 ms envelope (dBFS)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(alpha=0.2)
    ax[0].set_title("Envelope, reference attenuated by "
                    f"{gain_db:.1f} dB to match loudness")
    bw = 20.0
    nb = int(min(env_o.size, env_r.size) // (bw * 10))
    if nb > 3:
        eo = env_o[: nb * int(bw * 10)].reshape(nb, -1).mean(axis=1)
        er = env_r[: nb * int(bw * 10)].reshape(nb, -1).mean(axis=1)
        ax[1].plot(np.arange(nb) * bw, eo, color="#7f8b9c", lw=1.0, label="original")
        ax[1].plot(np.arange(nb) * bw, er, color="#e5695b", lw=1.0, label="reference")
        ax[1].set_ylabel(f"{bw:.0f} s mean level (dBFS)")
        ax[1].set_xlabel("seconds")
        ax[1].grid(alpha=0.2)
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "envelope.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="unprocessed file")
    ap.add_argument("--reference", required=True, help="processed file to investigate")
    ap.add_argument("--out-dir", default="analysis/pair", help="where to write results")
    ap.add_argument("--figures", action="store_true", help="also render PNG figures")
    args = ap.parse_args()

    for p in (args.original, args.reference):
        if not os.path.isfile(p):
            print(f"error: file not found: {p}")
            return 2
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"analysing {os.path.basename(args.original)} vs "
          f"{os.path.basename(args.reference)} ...")
    report = analyze_pair(args.original, args.reference)

    from timbreforge.gui.analysis_dialog import format_report
    text = format_report(report)
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    tc = report["transfer_curve"]
    with open(os.path.join(args.out_dir, "transfer.csv"), "w", encoding="utf-8") as fh:
        fh.write("freq_hz,apparent_transfer_db\n")
        for f, d in zip(tc["freqs_hz"], tc["db"]):
            fh.write(f"{f:.3f},{d:.4f}\n")
    print(text)

    if args.figures:
        lag = int(report["alignment"]["lag_samples"])
        made = _figures(args.original, args.reference, report, args.out_dir, lag)
        for m in made:
            print("figure:", m)

    print(f"\nwrote {args.out_dir}/report.txt, report.json, transfer.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
