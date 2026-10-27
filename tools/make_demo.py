#!/usr/bin/env python3
"""Generate demo material so the app can be tried without any supplied files.

    python tools/make_demo.py --out demo/

Writes:
  demo_loop.wav    a 24 s stereo, moderately dynamic pseudo-musical loop
  demo_quiet.wav   the same material at -26 dBFS (quiet-source test)
  demo_loud.wav    the same material near full scale with sharp transients
  demo_transients.wav  percussion-heavy stress material for transient/limiter tests

These are synthetic; they are not the reference pair and say nothing about it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from timbreforge.dsp.audio import ExportSettings, save_audio


def build(seed: int = 12, seconds: float = 24.0, sr: int = 44100) -> np.ndarray:
    """Two 8-bar sections with different arrangements, fading material in and out."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    beat = 0.5  # 120 bpm

    # chord bed: four notes with slow vibrato-free amplitude motion
    bed = np.zeros(n)
    for f, a in ((110.0, 0.30), (164.8, 0.18), (220.0, 0.16), (329.6, 0.10), (440.0, 0.05)):
        bed += a * np.sin(2 * np.pi * f * t + rng.random() * 6.28)

    # plucked arpeggio with a decaying envelope (transient content)
    arp = np.zeros(n)
    step = int(sr * beat / 2)
    notes = [440.0, 554.4, 659.3, 880.0, 659.3, 554.4]
    for i, start in enumerate(range(0, n, step)):
        f = notes[i % len(notes)]
        ln = min(step, n - start)
        env = np.exp(-np.arange(ln) / (sr * 0.12))
        arp[start:start + ln] += 0.22 * env * np.sin(2 * np.pi * f * np.arange(ln) / sr)

    # kick and hats
    kick = np.zeros(n)
    hat = np.zeros(n)
    for start in range(0, n, int(sr * beat)):
        ln = min(int(sr * 0.18), n - start)
        f = 55 * np.exp(-np.arange(ln) / (sr * 0.05))
        kick[start:start + ln] += 0.45 * np.sin(2 * np.pi * np.cumsum(f) / sr) * \
            np.exp(-np.arange(ln) / (sr * 0.07))
    for start in range(int(sr * beat / 2), n, int(sr * beat)):
        ln = min(int(sr * 0.05), n - start)
        hat[start:start + ln] += 0.12 * rng.standard_normal(ln) * \
            np.exp(-np.arange(ln) / (sr * 0.008))

    # gentle section-level dynamics so the material is not a constant wall
    sw = 0.75 + 0.25 * np.sin(2 * np.pi * 0.08 * t)
    section = np.where(((t // 6.0) % 2) == 0, 1.0, 0.0)          # alternate arrangement
    breakdown = np.where((t % 12.0) < 1.2, 0.35, 1.0)            # quiet bar every 12 s
    arp_gain = 0.35 + 0.65 * section
    hat_gain = 0.4 + 0.6 * section
    left = (bed * sw + arp * arp_gain * breakdown + (kick + hat * hat_gain) * breakdown)
    right = (bed * sw * 0.95 + np.roll(arp, 32) * arp_gain * breakdown
             + (kick + np.roll(hat, 7) * hat_gain) * breakdown)

    stereo = np.stack([left, right], axis=1)
    stereo -= stereo.mean(axis=0)
    return stereo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sr = args.sr
    base = build(sr=sr)

    def norm(x, peak):
        return x / max(np.abs(x).max(), 1e-9) * peak

    variants = {
        "demo_loop.wav": norm(base, 0.72),
        "demo_quiet.wav": norm(base, 0.05),
        "demo_loud.wav": norm(base, 0.99),
    }
    for name, audio in variants.items():
        path = os.path.join(args.out, name)
        save_audio(audio, ExportSettings(path=path, format="WAV", subtype="PCM_24",
                                         overwrite=True), sample_rate=sr)
        print("wrote", path)

    # transient stress: sparse clicks and noise bursts with long silences
    n = int(sr * 12)
    x = np.zeros((n, 2))
    for start in range(0, n, int(sr * 0.7)):
        ln = min(int(sr * 0.02), n - start)
        x[start:start + ln] += (np.random.default_rng(start).standard_normal((ln, 2)) * 0.9 *
                                np.exp(-np.arange(ln) / (sr * 0.004))[:, None])
    path = os.path.join(args.out, "demo_transients.wav")
    save_audio(x, ExportSettings(path=path, format="WAV", subtype="PCM_24", overwrite=True),
               sample_rate=sr)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
