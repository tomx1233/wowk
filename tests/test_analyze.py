"""Tests for the reference-pair analysis, using synthetic pairs with known truth.

The point is to prove that the checks in the report actually detect what they
claim: a known lag, a known pitch error, a known static EQ and a known
compression are each recovered from the measurement alone.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import signal

from timbreforge.dsp.analyze import (analyze_pair, estimate_lag, fine_tuning_ratio,
                                     level_stratified_transfer,
                                     short_term_level_relation)
from timbreforge.dsp.audio import ExportSettings, save_audio
from timbreforge.dsp.graph import Profile, render
from timbreforge.dsp.biquad import Band
from timbreforge.dsp.dynamics import CompressorParams

SR = 44100
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def material(seconds=40.0, sr=SR, seed=11):
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # slowly modulated pseudo-musical material: chords + arpeggio + noise bed
    env = 0.35 + 0.3 * np.sin(2 * np.pi * 0.4 * t) + 0.1 * np.sin(2 * np.pi * 1.7 * t)
    body = sum(a * np.sin(2 * np.pi * f * t) for f, a in
               ((110, 0.25), (165, 0.16), (220, 0.14), (330, 0.1), (660, 0.06)))
    arp = np.zeros(n)
    step = int(sr * 0.25)
    for i, start in enumerate(range(0, n, step)):
        f = [440, 554, 659, 880][i % 4]
        ln = min(step, n - start)
        arp[start:start + ln] += 0.18 * np.exp(-np.arange(ln) / (sr * 0.08)) * \
            np.sin(2 * np.pi * f * np.arange(ln) / sr)
    bed = 0.03 * rng.standard_normal(n)
    left = (body * env + arp + bed)
    right = (body * env * 0.95 + np.roll(arp, 24) + np.roll(bed, 5))
    x = np.stack([left, right], axis=1)
    return x / np.abs(x).max() * 0.7


def write(path, x, sr=SR):
    save_audio(x, ExportSettings(path=path, format="WAV", subtype="PCM_24", overwrite=True),
               sample_rate=sr)


def main():
    tmp = tempfile.mkdtemp()
    x = material()

    # a "reference" with a known lag, a known EQ, a known compressor and gain
    prof = Profile()
    prof.eq_bands = [Band("highshelf", 5000, 6.0, 0.707), Band("lowshelf", 120, -4.0, 0.707)]
    prof.compressor = CompressorParams(enabled=True, threshold_db=-26, ratio=3.0,
                                       attack_ms=25, release_ms=150, knee_db=6)
    prof.loudness_match = False
    rendered = render(x, SR, prof).samples
    lag_true = 64
    padded = np.concatenate([np.zeros((lag_true, 2)), rendered * 1.6], axis=0)[: len(x)]
    orig_path = os.path.join(tmp, "orig.wav")
    ref_path = os.path.join(tmp, "ref.wav")
    write(orig_path, x)
    write(ref_path, padded)

    print("\n[lag estimation]")
    lag = estimate_lag(x, padded, SR, max_lag_ms=50.0)
    check(abs(lag["lag_samples"] - lag_true) <= 2,
          f"known lag recovered ({lag['lag_samples']} vs {lag_true})")
    check(lag["correlation"] > 0.5, f"correlation is high ({lag['correlation']:.3f})")

    print("\n[full report]")
    rep = analyze_pair(orig_path, ref_path)
    check(rep["same_recording_verdict"]["verdict"] == "same recording",
          "pair recognised as the same recording")
    check(abs(rep["geometry"]["frame_delta"]) <= 1, "frame delta reported")
    check(abs(rep["tuning"]["ratio"] - 1.0) < 0.002,
          f"tuning measured as unchanged ({rep['tuning']['ratio']:.5f})")
    check(rep["alignment"]["segments"]["consistent_offset"], "constant offset, no drift")

    print("\n[static vs level-dependent]")
    strat = rep["level_stratified"]
    hf = "5120-10240"
    shape_hf = [b["relative"][hf] for b in strat["level_bins"]]
    check(all(v > 2.0 for v in shape_hf),
          f"high-band lift detected at every level (gain removed, min {min(shape_hf):+.1f} dB)")
    dep = strat["level_dependence_db"]
    check(max(abs(v) for v in dep.values()) < 6.0,
          "with a compressor in the chain the level dependence stays moderate "
          f"(max deviation {max(abs(v) for v in dep.values()):.1f} dB)")
    rel = rep["level_relation"]
    check(rel["ok"] and rel["level_slope"] < 0.6,
          f"compression detected from the level slope ({rel['level_slope']:.2f})")
    check(rel["level_correlation"] > 0.5, "level correlation reported")
    check(rep["levels_reference"]["envelope_range_db"] <
          rep["levels_original"]["envelope_range_db"] - 1.0,
          "envelope-range reduction detected "
          f"({rep['levels_reference']['envelope_range_db']:.1f} vs "
          f"{rep['levels_original']['envelope_range_db']:.1f} dB)")
    match = rep["level_match_gain_db"]
    expected = rep["levels_original"]["lufs"] - rep["levels_reference"]["lufs"]
    check(abs(match - expected) < 0.01,
          f"level-match gain equals the loudness difference ({match:+.2f} dB)")
    low_band = [b["relative"]["80-160"] for b in strat["level_bins"]]
    check(all(v < 0 for v in low_band), "low shelf cut detected at every level")

    print("\n[purely static reference]")
    static = Profile()
    static.eq_bands = [Band("highshelf", 5000, 6.0, 0.707)]
    static.loudness_match = False
    static.limiter.enabled = False
    static_ref = np.concatenate([np.zeros((lag_true, 2)),
                                 render(x, SR, static).samples], axis=0)[: len(x)]
    st_path = os.path.join(tmp, "static.wav")
    write(st_path, static_ref)
    rep_static = analyze_pair(orig_path, st_path)
    dep_s = rep_static["level_stratified"]["level_dependence_db"]
    worst = max(abs(v) for v in dep_s.values())
    check(worst < 1.5,
          f"a static shelf is reported as level-independent (max deviation {worst:.2f} dB)")

    print("\n[a level-dependent process must be detected]")
    from timbreforge.dsp.dynamics import DynamicEqBand, DynamicEqParams
    # dedicated control signal: a 4 kHz tone whose level swings 24 dB, so a
    # level-dependent band can be measured unambiguously
    n = int(SR * 30)
    tt = np.arange(n) / SR
    swing = 10 ** ((12 * np.sin(2 * np.pi * 0.15 * tt)) / 20.0)   # +/-12 dB
    base = (0.05 * np.sin(2 * np.pi * 250 * tt) +
            0.02 * swing * np.sin(2 * np.pi * 4000 * tt))
    ctrl = np.stack([base, base], axis=1)
    ctrl = ctrl / np.abs(ctrl).max() * 0.5
    dyn_prof = Profile()
    dyn_prof.eq_bands = []
    dyn_prof.dynamic_eq = DynamicEqParams(enabled=True, bands=[
        DynamicEqBand(enabled=True, freq=4000, q=1.0, gain_db=-12.0, threshold_db=-38,
                      ratio=4.0, attack_ms=5, release_ms=120, max_reduction_db=12.0)])
    dyn_prof.loudness_match = False
    dyn_prof.limiter.enabled = False
    dyn_ref = render(ctrl, SR, dyn_prof).samples
    ctrl_path = os.path.join(tmp, "ctrl.wav")
    dyn_path = os.path.join(tmp, "dyn.wav")
    write(ctrl_path, ctrl)
    write(dyn_path, dyn_ref)
    rep_dyn = analyze_pair(ctrl_path, dyn_path)
    strat_d = rep_dyn["level_stratified"]
    check(strat_d["max_level_dependence_db"] > 2.0,
          "level-dependent EQ measured as level dependent "
          f"({strat_d['max_level_dependence_db']:.1f} dB)")
    check(rep_dyn["level_stratified"]["level_dependent"] is True,
          "level_dependent flag set")
    check(strat_d["verdict"].startswith("level-dependent"),
          f"verdict says so: '{strat_d['verdict']}'")

    print("\n[pitch-shift detection (must not be fooled)]")
    # the same material resampled -> a genuine ~1 semitone pitch/speed change
    shifted = signal.resample_poly(rendered, 100, 94)[: len(x)]
    shifted = np.concatenate([shifted, np.zeros((len(x) - len(shifted), 2))], axis=0)
    ref2 = os.path.join(tmp, "ref_shifted.wav")
    write(ref2, shifted)
    tun = fine_tuning_ratio(x, shifted, SR)
    check(tun["ratio"] > 1.02, f"pitch ratio detected ({tun['ratio']:.4f})")
    rep2 = analyze_pair(orig_path, ref2)
    check(rep2["same_recording_verdict"]["verdict"] == "needs review",
          "a stretched/shifted pair is flagged for review, not accepted")

    print("\n" + ("ALL ANALYSIS CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
