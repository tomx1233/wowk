#!/usr/bin/env python3
"""Compare candidate processing families against a reference pair.

Runs a fixed list of candidate graphs through the real render engine, measures
them at matched loudness, and prints a table so the choice of method is based on
evidence rather than on one guess.  Every candidate keeps pitch and speed at
1.00x; none of them is allowed to raise loudness.

Usage
-----
    python tools/compare_methods.py --original TestSong.ogg --reference NewTest.ogg \
        [--out analysis/methods.md] [--max-seconds 60]

Notes
-----
* The score is gain-invariant: candidates are level-matched to the original
  before every comparison, so a louder chain cannot win on loudness alone.
* The reference's own level statistics are reported next to the candidates to
  show which of its properties are simply unreachable under the quality rules.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from timbreforge.dsp.audio import load_audio
from timbreforge.dsp.biquad import Band
from timbreforge.dsp.dynamics import (CompressorParams, DynamicEqBand,
                                      DynamicEqParams, LimiterParams,
                                      MultibandParams)
from timbreforge.dsp.graph import Profile, render
from timbreforge.dsp.meter import measure
from timbreforge.dsp.nonlinear import SaturationParams
from timbreforge.dsp.stereo import StereoParams


@dataclass
class Candidate:
    key: str
    family: str
    hypothesis: str
    explains: str
    fails: str
    profile: Profile
    notes: str = ""


def _prof(name: str) -> Profile:
    p = Profile(name=name)
    p.loudness_match = True
    p.limiter = LimiterParams(enabled=True, ceiling_db=-1.0)
    return p


def build_candidates() -> list[Candidate]:
    out: list[Candidate] = []

    out.append(Candidate(
        "A_identity", "A. static tone",
        "Reference is only an EQ change plus gain",
        "any static spectral difference",
        "density, crest and stereo changes",
        _prof("EQ only")))

    p = _prof("Low cut + tilt")
    p.eq_bands = [Band("highpass", 32, 0, 0.707), Band("lowshelf", 110, -3.0, 0.707),
                  Band("peaking", 380, -1.5, 0.85), Band("highshelf", 7000, 4.0, 0.707)]
    out.append(Candidate(
        "B_tilt", "A. static tone",
        "Static tilt: less sub-bass, more air",
        "band-energy deltas and centroid",
        "anything level dependent",
        p))

    p = _prof("Tilt + broadband comp")
    p.eq_bands = [Band("highpass", 32, 0, 0.707), Band("lowshelf", 110, -3.0, 0.707),
                  Band("highshelf", 7000, 4.0, 0.707)]
    p.compressor = CompressorParams(enabled=True, threshold_db=-22, ratio=2.5,
                                    attack_ms=12, release_ms=140, knee_db=6, mix=0.8)
    out.append(Candidate(
        "C_tilt_comp", "C. dynamics",
        "Tilt plus a real compressor",
        "envelope range, crest, density",
        "HF-specific behaviour",
        p))

    p = _prof("Multiband dynamics")
    p.eq_bands = [Band("highpass", 32, 0, 0.707), Band("lowshelf", 110, -2.5, 0.707)]
    p.multiband = MultibandParams(enabled=True, crossovers_hz=[200, 2500], bands=[
        CompressorParams(enabled=True, threshold_db=-22, ratio=2.0, attack_ms=25,
                         release_ms=180, mix=0.8),
        CompressorParams(enabled=True, threshold_db=-24, ratio=1.8, attack_ms=15,
                         release_ms=150, mix=0.7),
        CompressorParams(enabled=True, threshold_db=-30, ratio=1.8, attack_ms=5,
                         release_ms=90, mix=0.8)])
    out.append(Candidate(
        "D_multiband", "B. dynamic tone",
        "Band-wise dynamics make the top end denser without flattening everything",
        "level-dependent HF behaviour, flatter envelope",
        "a purely static spectral tilt",
        p))

    p = _prof("Dynamic EQ top")
    p.eq_bands = [Band("highshelf", 7000, 3.0, 0.707), Band("lowshelf", 110, -2.0, 0.707)]
    p.dynamic_eq = DynamicEqParams(enabled=True, bands=[
        DynamicEqBand(enabled=True, freq=6500, q=1.4, gain_db=-5.0, threshold_db=-32,
                      ratio=2.5, attack_ms=3, release_ms=90, max_reduction_db=7)])
    out.append(Candidate(
        "E_dyneq", "B. dynamic tone",
        "High band is controlled only when loud (de-esser style)",
        "level-dependent HF, sibilance control",
        "broadband tightness",
        p))

    p = _prof("Tilt + saturation")
    p.eq_bands = [Band("highshelf", 6000, 2.5, 0.707), Band("lowshelf", 110, -2.0, 0.707)]
    p.saturation = SaturationParams(enabled=True, shaper="tanh", drive_db=9.0, mix=0.35,
                                    oversample=4, compensate="rms")
    out.append(Candidate(
        "F_saturation", "D. nonlinear",
        "HF lift comes from generated harmonics, not EQ",
        "harmonic content, spectral flatness",
        "spectral centroid change without added harmonics",
        p, notes="level compensated by attenuation only"))

    p = _prof("Comp + saturation")
    p.compressor = CompressorParams(enabled=True, threshold_db=-24, ratio=3.0,
                                    attack_ms=8, release_ms=120, knee_db=4, mix=0.9)
    p.saturation = SaturationParams(enabled=True, shaper="tanh", drive_db=12.0, mix=0.5,
                                    oversample=4, compensate="rms")
    out.append(Candidate(
        "G_comp_sat", "D. nonlinear",
        "Density first, then harmonic generation",
        "crest collapse plus harmonic brightness",
        "the harshness and intermodulation of the reference",
        p, notes="worst case for harshness; watch pumping"))

    p = _prof("Stereo only")
    p.stereo = StereoParams(enabled=True, width=1.15)
    out.append(Candidate(
        "H_stereo", "E. stereo",
        "Only width changed",
        "side/mid ratio",
        "correlation drop without a dedicated widener",
        p))

    p = _prof("Tilt + width + band comp")
    p.eq_bands = [Band("highpass", 32, 0, 0.707), Band("lowshelf", 120, -2.5, 0.707),
                  Band("peaking", 1600, 1.5, 0.8), Band("highshelf", 7000, 3.5, 0.707)]
    p.multiband = MultibandParams(enabled=True, crossovers_hz=[180, 2400], bands=[
        CompressorParams(enabled=True, threshold_db=-22, ratio=1.6, attack_ms=25,
                         release_ms=200, mix=0.7),
        CompressorParams(enabled=True, threshold_db=-21, ratio=1.7, attack_ms=18,
                         release_ms=170, mix=0.6),
        CompressorParams(enabled=True, threshold_db=-26, ratio=1.4, attack_ms=10,
                         release_ms=140, mix=0.45)])
    p.compressor = CompressorParams(enabled=True, threshold_db=-20, ratio=1.7,
                                    attack_ms=22, release_ms=210, knee_db=10, mix=0.55)
    p.saturation = SaturationParams(enabled=True, shaper="tanh", drive_db=4.0, mix=0.12,
                                    oversample=4, compensate="rms")
    p.stereo = StereoParams(enabled=True, width=1.10)
    out.append(Candidate(
        "I_combined", "F. combined",
        "Tilt + band dynamics + gentle saturation + width",
        "several deltas at once without loudness maximising",
        "the reference's extreme density (deliberately not targeted)",
        p, notes="this is what the app ships as a starting point"))

    p = _prof("Loudness maximised (rejected)")
    p.loudness_match = False
    p.eq_bands = [Band("highshelf", 7000, 5.0, 0.707)]
    p.compressor = CompressorParams(enabled=True, threshold_db=-30, ratio=8.0,
                                    attack_ms=2, release_ms=60, knee_db=2, mix=1.0,
                                    makeup_db=0.0)
    p.limiter = LimiterParams(enabled=True, ceiling_db=-0.1, lookahead_ms=3.0)
    out.append(Candidate(
        "J_maximised_rejected", "control",
        "Reproduce the reference's loudness directly",
        "crest and density figures",
        "the quality rules; kept only as a demonstration of what is refused",
        p, notes="REFERENCE ONLY - violates the quality rules, never shipped"))
    return out


def score(cand_samples: np.ndarray, sr: int, orig_stats, ref_stats) -> dict:
    st = measure(cand_samples, sr)
    if not np.isfinite(st.lufs) or st.lufs < -69:
        return {"valid": False}
    gain = orig_stats.lufs - st.lufs
    matched = measure(cand_samples * 10 ** (gain / 20.0), sr)
    keys = list(ref_stats.bands)
    o = np.array([orig_stats.bands[k] for k in keys])
    r = np.array([ref_stats.bands[k] for k in keys])
    m = np.array([matched.bands[k] for k in keys])
    target = r - r.mean()
    got = m - m.mean()
    orig_rel = o - o.mean()
    target_delta = target - orig_rel
    got_delta = got - orig_rel
    return {
        "valid": True,
        "band_rmse": float(np.sqrt(np.mean((got_delta - target_delta) ** 2))),
        "centroid": matched.centroid_hz,
        "centroid_err": abs(np.log2(max(matched.centroid_hz, 1) / max(ref_stats.centroid_hz, 1))),
        "crest": matched.crest_db,
        "crest_err": abs(matched.crest_db - ref_stats.crest_db),
        "env_range": matched.envelope_range_db,
        "env_err": abs(matched.envelope_range_db - ref_stats.envelope_range_db),
        "corr": matched.correlation,
        "corr_err": abs(matched.correlation - ref_stats.correlation),
        "sm": matched.side_mid_db,
        "sm_err": abs(matched.side_mid_db - ref_stats.side_mid_db),
        "lufs": st.lufs,
        "tp": st.true_peak_db,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default="analysis/method_comparison.md")
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="analyse at most this many seconds (0 = whole file)")
    args = ap.parse_args()

    for p in (args.original, args.reference):
        if not os.path.isfile(p):
            print(f"error: file not found: {p}")
            return 2

    orig = load_audio(args.original)
    ref = load_audio(args.reference)
    if orig.sample_rate != ref.sample_rate:
        print("error: sample rates differ; resample the reference first")
        return 2
    sr = orig.sample_rate
    n = orig.frames if args.max_seconds <= 0 else min(orig.frames, int(args.max_seconds * sr))
    # align crudely on the strongest correlation before trimming
    from timbreforge.dsp.analyze import estimate_lag
    lag = estimate_lag(orig.samples[: int(min(orig.duration, 20) * sr)],
                       ref.samples[: int(min(ref.duration, 20) * sr)], sr,
                       max_lag_ms=200.0)
    print(f"alignment lag: {lag['lag_samples']} samples "
          f"({lag['lag_ms']:+.2f} ms), r={lag['correlation']:.3f}")
    start = int(sr * 60.0)  # a held-out section, not the intro
    if start + n > min(orig.frames, ref.frames):
        start = max(0, min(orig.frames, ref.frames) - n)
    x = orig.samples[start:start + n]
    lag_n = int(lag["lag_samples"])
    r_start = max(0, start + lag_n)
    r = ref.samples[r_start:r_start + n]
    m = min(x.shape[0], r.shape[0])
    x, r = x[:m], r[:m]

    orig_stats = measure(x, sr)
    ref_stats = measure(r, sr)
    print(f"analysing {m / sr:.1f} s from {start / sr:.1f} s")
    print(f"reference: {ref_stats.lufs:.2f} LUFS, crest {ref_stats.crest_db:.2f} dB "
          f"(the original is {orig_stats.lufs:.2f} LUFS, crest {orig_stats.crest_db:.2f} dB)")

    rows = []
    for cand in build_candidates():
        print(f"  rendering {cand.key} ...", flush=True)
        res = render(x, sr, cand.profile)
        sc = score(res.samples, sr, orig_stats, ref_stats)
        rows.append((cand, sc))

    valid = [(c, s) for c, s in rows if s.get("valid")]
    valid.sort(key=lambda cs: cs[1]["band_rmse"])

    lines = ["# Candidate methods, measured against the reference", "",
             f"Compared on {m / sr:.1f} s starting at {start / sr:.1f} s, "
             f"lag-corrected by {lag_n} samples.", "",
             "All candidates keep pitch and speed at 1.00x. Scores are gain invariant: "
             "each render is attenuated to the original's loudness before comparison, so "
             "a louder chain cannot win.", "",
             "| candidate | family | band RMSE dB | centroid Hz | crest dB | env range dB | "
             "L/R corr | S/M dB | export LUFS | true peak dBTP |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in rows:
        if not s.get("valid"):
            lines.append(f"| {c.key} | {c.family} | invalid | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {c.key} | {c.family} | {s['band_rmse']:.2f} | {s['centroid']:.0f} | "
            f"{s['crest']:.1f} | {s['env_range']:.1f} | {s['corr']:.3f} | "
            f"{s['sm']:+.1f} | {s['lufs']:.1f} | {s['tp']:+.2f} |")
    lines += ["", f"Reference itself: centroid {ref_stats.centroid_hz:.0f} Hz, crest "
              f"{ref_stats.crest_db:.1f} dB, envelope range "
              f"{ref_stats.envelope_range_db:.1f} dB, corr {ref_stats.correlation:.3f}, "
              f"S/M {ref_stats.side_mid_db:+.1f} dB, {ref_stats.lufs:.1f} LUFS.", "",
              "## What each candidate explains and misses", ""]
    for c, s in rows:
        lines += [f"### {c.key} - {c.hypothesis}",
                  f"- explains: {c.explains}",
                  f"- fails to explain: {c.fails}"]
        if c.notes:
            lines.append(f"- note: {c.notes}")
        if s.get("valid"):
            lines.append(f"- measured: band RMSE {s['band_rmse']:.2f} dB, crest error "
                         f"{s['crest_err']:.1f} dB, envelope error {s['env_err']:.1f} dB, "
                         f"correlation error {s['corr_err']:.3f}")
        lines.append("")
    lines += ["## Ranking (best band-shape match first)", ""]
    for i, (c, s) in enumerate(valid, 1):
        lines.append(f"{i}. {c.key} - band RMSE {s['band_rmse']:.2f} dB")
    lines += ["", "The ranking is about spectral shape only. Crest and envelope columns "
              "show why the reference cannot be matched on dynamics: its crest is around "
              "2.6 dB, which requires the loudness maximising that this project refuses. "
              "A candidate is therefore chosen for tonal/textural closeness with natural "
              "dynamics, not for the lowest RMSE at any cost.", ""]

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
