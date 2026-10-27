"""Invariant tests for the processing graph.

Run with:  python tests/test_dsp.py      (or: pytest -q)
These tests do not need any audio files; they synthesise material.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge.dsp.audio import AudioData, ExportSettings, load_audio, save_audio
from timbreforge.dsp.biquad import Band, magnitude_db
from timbreforge.dsp.dynamics import (CompressorParams, DynamicEqBand,
                                      DynamicEqParams, LimiterParams,
                                      MultibandParams, TransientParams,
                                      limit, true_peak)
from timbreforge.dsp.graph import Cancelled, Profile, preset_gentle_character, render
from timbreforge.dsp.meter import (db, integrated_loudness, measure,
                                   channel_correlation, mono_fold_change,
                                   spectral_centroid, to_mono)
from timbreforge.dsp.nonlinear import SaturationParams
from timbreforge.dsp.stereo import StereoParams

SR = 44100
FAILS: list[str] = []


def check(cond: bool, msg: str):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)


def dominant_freq(x: np.ndarray, sr: int = SR) -> float:
    mono = to_mono(x)
    n = mono.size
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(mono * w))
    k = int(np.argmax(spec))
    if 0 < k < spec.size - 1:
        a, b, c = np.log(spec[k - 1] + 1e-20), np.log(spec[k] + 1e-20), np.log(spec[k + 1] + 1e-20)
        d = 0.5 * (a - c) / (a - 2 * b + c)
        k = k + d
    return k * sr / n


def tone(freq=1000.0, seconds=2.0, amp=0.25, sr=SR, stereo=True) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    s = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([s, s * 0.9], axis=1) if stereo else s


def noise(seconds=2.0, amp=0.1, sr=SR) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(sr * seconds)
    l = rng.standard_normal(n) * amp
    r = rng.standard_normal(n) * amp
    return np.stack([l, r], axis=1)


def full_profile() -> Profile:
    p = Profile(name="test-all")
    p.eq_bands = [Band("highpass", 30, 0, 0.707), Band("peaking", 300, -2.0, 1.0),
                  Band("highshelf", 8000, 3.0, 0.707)]
    p.dynamic_eq = DynamicEqParams(enabled=True, bands=[
        DynamicEqBand(enabled=True, freq=3500, q=1.2, gain_db=-4, threshold_db=-30,
                      ratio=2.0)])
    p.transient = TransientParams(enabled=True, attack_db=1.5, sustain_db=-1.0)
    p.multiband = MultibandParams(enabled=True, crossovers_hz=[200.0, 2000.0], bands=[
        CompressorParams(enabled=True, threshold_db=-24, ratio=2.0, attack_ms=20,
                         release_ms=180, mix=0.6),
        CompressorParams(enabled=True, threshold_db=-22, ratio=1.8, attack_ms=12,
                         release_ms=150, mix=0.5),
        CompressorParams(enabled=True, threshold_db=-20, ratio=1.6, attack_ms=8,
                         release_ms=120, mix=0.4),
    ])
    p.compressor = CompressorParams(enabled=True, threshold_db=-20, ratio=2.0,
                                   attack_ms=18, release_ms=200, knee_db=8, mix=0.7)
    p.saturation = SaturationParams(enabled=True, shaper="tanh", drive_db=6.0, mix=0.25,
                                    oversample=4)
    p.stereo = StereoParams(enabled=True, width=0.85)
    p.limiter = LimiterParams(enabled=True, ceiling_db=-1.0)
    return p


# ---------------------------------------------------------------------------


def test_identity():
    print("\n[identity / bypass]")
    x = noise(1.0)
    r = render(x, SR, Profile())
    check(np.array_equal(r.samples, x), "default profile returns the input unchanged")
    p = Profile(bypass=True)
    p.compressor = CompressorParams(enabled=True, threshold_db=-30, ratio=6)
    r = render(x, SR, p)
    check(np.array_equal(r.samples, x), "global bypass ignores configured stages")


def test_length_and_pitch():
    print("\n[length, pitch and timing]")
    x = tone(1000.0, 2.0)
    f0 = dominant_freq(x)
    for label, prof in (("gentle", preset_gentle_character()), ("all stages", full_profile())):
        r = render(x, SR, prof)
        check(r.samples.shape == x.shape, f"{label}: length preserved exactly")
        f1 = dominant_freq(r.samples)
        cents = 1200 * math.log2(f1 / f0)
        check(abs(cents) < 0.5, f"{label}: pitch unchanged (delta {cents:+.3f} cents)")
        check(r.latency_samples == 0, f"{label}: no reported latency")

    # a click stays at the same sample index
    x = np.zeros((SR, 2))
    click_at = 12345
    x[click_at, :] = 0.5
    r = render(x, SR, preset_gentle_character())
    env = np.abs(r.samples).max(axis=1)
    peak_at = int(np.argmax(env))
    check(abs(peak_at - click_at) <= 2,
          f"impulse position preserved ({peak_at} vs {click_at})")


def test_loudness_policy():
    print("\n[loudness policy]")
    for amp, name in ((0.03, "quiet (-30 dBFS tone)"), (0.5, "loud (-6 dBFS tone)"),
                      (0.95, "near full scale")):
        x = tone(440.0, 1.5, amp=amp)
        st_in = measure(x, SR)
        r = render(x, SR, preset_gentle_character())
        st_out = measure(r.samples, SR)
        check(st_out.lufs <= st_in.lufs + 0.2,
              f"{name}: output not louder than input "
              f"({st_out.lufs:.2f} vs {st_in.lufs:.2f} LUFS)")
        check(st_out.true_peak_db <= -0.85,
              f"{name}: true peak under the ceiling ({st_out.true_peak_db:+.2f} dBTP)")
        check(np.isfinite(r.samples).all(), f"{name}: no NaN/Inf")
        check(st_out.clipped_fraction < 1e-4, f"{name}: no samples pinned at full scale")


def test_multiband_transparency():
    print("\n[multiband crossover]")
    from timbreforge.dsp.biquad import split_bands
    from timbreforge.dsp.dynamics import MultibandParams, multiband
    x = noise(1.0, amp=0.2)
    bands, delays = split_bands(x, SR, [180.0, 2400.0])
    err = np.abs(sum(bands) - x).max()
    check(err < 1e-12, f"bands sum back to the input exactly (max error {err:.2e})")
    neutral = [CompressorParams(enabled=True, threshold_db=0.0, ratio=1.0, mix=1.0)] * 3
    y, _ = multiband(x, SR, MultibandParams(enabled=True, crossovers_hz=[180.0, 2400.0],
                                            bands=neutral))
    check(np.abs(y - x).max() < 1e-12, "neutral multiband settings are transparent")
    # a loud low band must not move the high band in time
    p = Profile()
    p.multiband = MultibandParams(enabled=True, crossovers_hz=[180.0, 2400.0], bands=[
        CompressorParams(enabled=True, threshold_db=-30, ratio=4.0, attack_ms=10,
                         release_ms=120, mix=1.0),
        CompressorParams(), CompressorParams()])
    y2 = render(x, SR, p).samples
    check(y2.shape == x.shape, "multiband keeps the length")


def test_limiter():
    print("\n[true-peak limiter]")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((SR * 2, 2)) * 0.9
    y, rep = limit(x, SR, LimiterParams(enabled=True, ceiling_db=-1.0))
    tp = db(true_peak(y))
    check(tp <= -0.95, f"ceiling respected ({tp:+.2f} dBTP)")
    check(y.shape == x.shape, "limiter preserves length (zero latency)")
    # quieter material passes untouched
    q = tone(300.0, 1.0, amp=0.02)
    y2, rep2 = limit(q, SR, LimiterParams(enabled=True, ceiling_db=-1.0))
    check(rep2.get("bypassed") is True, "limiter bypassed when headroom is available")


def test_nonlinearity_is_level_compensated():
    print("\n[saturation compensation]")
    x = tone(500.0, 1.0, amp=0.3)
    p = Profile()
    p.saturation = SaturationParams(enabled=True, shaper="tanh", drive_db=12, mix=0.6,
                                    compensate="rms")
    p.loudness_match = False
    p.limiter = LimiterParams(enabled=False)
    r = render(x, SR, p)
    gain = db(np.sqrt(np.mean(r.samples ** 2)) / np.sqrt(np.mean(x ** 2)))
    check(gain <= 0.05, f"no loudness gain from saturation ({gain:+.2f} dB)")
    sat = [s for s in r.reports if s["stage"] == "saturation"][0]
    check(sat["compensation_db"] <= 0.0, "compensation is attenuation only")


def test_stereo_and_mono():
    print("\n[stereo width and mono compatibility]")
    x = noise(1.5, amp=0.2)
    base_corr = channel_correlation(x)
    p = Profile()
    p.stereo = StereoParams(enabled=True, width=1.4)
    p.loudness_match = True
    r = render(x, SR, p)
    from timbreforge.dsp.meter import side_mid_db
    check(side_mid_db(r.samples) > side_mid_db(x),
          "widening increases the side/mid ratio")
    check(mono_fold_change(r.samples) < mono_fold_change(x) + 0.5,
          "mono fold-down level is retained (no phase cancellation)")
    check(abs(np.mean(r.samples[:, 0] + r.samples[:, 1]) -
              np.mean(x[:, 0] + x[:, 1])) < 0.05,
          "mid signal (mono sum) is preserved by width control")
    p2 = Profile()
    p2.stereo = StereoParams(enabled=True, width=0.0)
    r2 = render(x, SR, p2)
    check(abs(channel_correlation(r2.samples) - 1.0) < 1e-6,
          "width 0% collapses to a correlated mono image")


def test_eq_accuracy():
    print("\n[static EQ accuracy]")
    x = noise(3.0, amp=0.05)
    p = Profile()
    p.eq_bands = [Band("highshelf", 4000, 6.0, 0.707), Band("peaking", 200, -5.0, 1.0)]
    p.loudness_match = False
    p.limiter = LimiterParams(enabled=False)
    r = render(x, SR, p)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    X = np.abs(np.fft.rfft(to_mono(x))) ** 2
    Y = np.abs(np.fft.rfft(to_mono(r.samples))) ** 2
    measured = 10 * np.log10(np.maximum(Y, 1e-20) / np.maximum(X, 1e-20))
    target = magnitude_db(p.eq_bands, np.maximum(f, 1e-6), SR)
    sel = (f > 40) & (f < 16000)
    # smooth both over ~1/6 octave to ignore noise
    err = []
    for fc in np.geomspace(50, 15000, 40):
        band = (f > fc / 1.06) & (f < fc * 1.06)
        if band.sum() < 3:
            continue
        err.append(np.mean(measured[band]) - np.mean(target[band]))
    err = np.array(err)
    check(np.max(np.abs(err)) < 0.6,
          f"rendered EQ matches the designed curve (max error {np.max(np.abs(err)):.2f} dB)")


def test_dynamic_stages_do_not_pump():
    print("\n[envelope sanity]")
    t = np.arange(int(SR * 4)) / SR
    env = np.where(t < 2.0, 0.02, 0.4)
    x = np.stack([env * np.sin(2 * np.pi * 300 * t)] * 2, axis=1)
    r = render(x, SR, preset_gentle_character())
    st_in = measure(x, SR)
    st_out = measure(r.samples, SR)
    check(abs(st_out.lufs - st_in.lufs) < 0.6, "step test: loudness matched")
    # no gain jumps: block RMS ratio between neighbours should stay bounded
    blk = int(SR * 0.05)
    n = r.samples.shape[0] // blk
    p_in = np.array([np.sqrt(np.mean(x[i * blk:(i + 1) * blk] ** 2)) for i in range(n)])
    p_out = np.array([np.sqrt(np.mean(r.samples[i * blk:(i + 1) * blk] ** 2)) for i in range(n)])
    ok = p_in > 1e-9
    gain = 20 * np.log10(np.maximum(p_out[ok], 1e-9) / p_in[ok])
    check(np.max(np.abs(np.diff(gain))) < 6.0,
          f"no accidental gain jumps (max block-to-block change "
          f"{np.max(np.abs(np.diff(gain))):.2f} dB)")


def test_material_robustness():
    print("\n[different kinds of material]")
    from timbreforge.dsp.graph import preset_reference_hypothesis
    prof = preset_reference_hypothesis()
    rng = np.random.default_rng(9)

    cases = {}
    # quiet source
    cases["quiet (-40 dBFS)"] = tone(330.0, 3.0, amp=0.01)
    # very loud source with sharp transients
    n = SR * 3
    loud = np.zeros((n, 2))
    for start in range(0, n, SR // 2):
        ln = min(int(SR * 0.05), n - start)
        loud[start:start + ln] += (0.98 * np.exp(-np.arange(ln) / (SR * 0.01))[:, None]
                                   * np.sign(rng.standard_normal((ln, 2))))
    cases["loud transients"] = loud
    # mono source
    cases["mono"] = tone(220.0, 2.0, amp=0.3, stereo=False)
    # stereo with out-of-phase content (worst case for width/mono)
    t = np.arange(SR * 2) / SR
    a = 0.3 * np.sin(2 * np.pi * 200 * t)
    cases["anti-phase stereo"] = np.stack([a, -a], axis=1)
    # noise only
    cases["noise"] = noise(2.0, amp=0.3)
    # silence
    cases["silence"] = np.zeros((SR, 2))

    for name, x in cases.items():
        try:
            r = render(x, SR, prof)
            st = measure(r.samples, SR)
            st_in = measure(x, SR)
            ok_len = r.frames == (x.shape[0])
            ok_finite = np.isfinite(r.samples).all()
            ok_peak = st.true_peak_db <= prof.limiter.ceiling_db + 0.05
            ok_loud = st.lufs <= st_in.lufs + 0.3
            check(ok_len and ok_finite and ok_peak and ok_loud,
                  f"{name}: length, finiteness, ceiling and loudness ok "
                  f"({st_in.lufs:.1f} -> {st.lufs:.1f} LUFS, "
                  f"{st.true_peak_db:+.1f} dBTP)")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{name}: raised {type(exc).__name__}: {exc}")

    # silence must not be turned into noise
    r = render(cases["silence"], SR, prof)
    check(float(np.abs(r.samples).max()) < 1e-6, "silence stays silent")


def test_cancellation():
    print("\n[cancellation]")
    x = noise(3.0, amp=0.2)
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 3

    try:
        render(x, SR, full_profile(), should_cancel=cancel)
        check(False, "cancellation raises")
    except Cancelled:
        check(True, "cancellation raises Cancelled")


def test_export_roundtrip():
    print("\n[export round-trip]")
    x = noise(2.0, amp=0.2)
    tmp = tempfile.mkdtemp()
    for fmt, sub, ext, tol_ms in (("WAV", "PCM_24", ".wav", 0.0),
                                  ("FLAC", "PCM_24", ".flac", 0.0),
                                  ("OGG", "VORBIS", ".ogg", 40.0),
                                  ("MP3", "MPEG_LAYER_III", ".mp3", 120.0)):
        path = os.path.join(tmp, "out" + ext)
        settings = ExportSettings(path=path, format=fmt, subtype=sub)
        facts = save_audio(x, settings, sample_rate=SR)
        back = load_audio(path)
        delta_ms = abs(back.duration - 2.0) * 1000
        check(back.sample_rate == SR, f"{fmt}: sample rate preserved")
        check(delta_ms <= tol_ms + 25, f"{fmt}: duration kept ({back.duration:.4f} s)")
        check(back.channels == 2, f"{fmt}: channels preserved")
        check(os.path.getsize(path) > 1000, f"{fmt}: non-empty file")

    # long Ogg export: a single huge libsndfile write crashed here, the block
    # writer must keep it working
    long_path = os.path.join(tmp, "long.ogg")
    long_x = (np.random.default_rng(4).standard_normal((SR * 50, 2)) * 0.1)
    facts = save_audio(long_x, ExportSettings(path=long_path, format="OGG",
                                              subtype="VORBIS"), sample_rate=SR)
    back = load_audio(long_path)
    check(back.frames == long_x.shape[0], f"50 s Ogg export round-trips "
                                          f"({back.frames} of {long_x.shape[0]} frames)")

    # refuses to overwrite the source
    src = os.path.join(tmp, "src.wav")
    save_audio(x, ExportSettings(path=src, format="WAV", subtype="PCM_24"), sample_rate=SR)
    try:
        save_audio(AudioData(samples=x, sample_rate=SR, path=src),
                   ExportSettings(path=src, source_path=src, overwrite=True))
        check(False, "source file protected from overwrite")
    except Exception:
        check(True, "source file protected from overwrite")


def test_bad_profiles():
    print("\n[validation]")
    p = Profile()
    p.eq_bands = [Band("peaking", 1000, 40.0, 1.0)]
    check(len(p.validate()) > 0, "out-of-range EQ gain rejected")
    p = Profile()
    p.stereo = StereoParams(enabled=True, width=5.0)
    check(len(p.validate()) > 0, "out-of-range width rejected")
    p = Profile()
    p.limiter = LimiterParams(ceiling_db=0.5)
    check(len(p.validate()) > 0, "unsafe ceiling rejected")


def test_profiles_json():
    print("\n[profile serialisation]")
    p = full_profile()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    p.save(path)
    q = Profile.load(path)
    check(q.to_dict() == p.to_dict(), "profile round-trips through JSON")
    check(q.validate() == [], "round-tripped profile is valid")
    os.unlink(path)


def main():
    test_identity()
    test_length_and_pitch()
    test_loudness_policy()
    test_multiband_transparency()
    test_limiter()
    test_nonlinearity_is_level_compensated()
    test_stereo_and_mono()
    test_eq_accuracy()
    test_dynamic_stages_do_not_pump()
    test_material_robustness()
    test_cancellation()
    test_export_roundtrip()
    test_bad_profiles()
    test_profiles_json()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED"))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
