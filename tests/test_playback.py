"""Qt-free tests for the v2 pieces: playback preparation, limits, simple profile.

Run with:  python tests/test_playback.py

These deliberately do not import PySide6, so they run anywhere numpy does - the
Qt widgets themselves need a display stack and are covered by
``tests/test_gui_smoke.py`` where one is available.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timbreforge.dsp.audio import ExportSettings, save_audio
from timbreforge.dsp.audio_limits import (check_export, check_import, check_loaded,
                                          describe_limits, rate_is_supported)
from timbreforge.dsp.graph import SimpleSettings, profile_from_simple, render
from timbreforge.dsp.meter import integrated_loudness, measure
from timbreforge.dsp.playback import (DeviceFormat, PlaybackBuffer, _to_dtype,
                                      apply_gain_bytes, pick_device_format,
                                      prepare_playback, silence_buffer, test_tone)

FAILS: list[str] = []


def check(cond: bool, msg: str):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def tone(seconds: float = 1.0, sr: int = 44100, ch: int = 2,
         amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    left = amp * np.sin(2 * np.pi * 440 * t)
    right = amp * np.sin(2 * np.pi * 660 * t)
    if ch == 1:
        return left[:, None]
    return np.stack([left, right], axis=1)


# ---------------------------------------------------------------------------
def test_device_format():
    print("\n[device format negotiation]")
    supported = [DeviceFormat(48000, 2, "float32"),
                 DeviceFormat(44100, 2, "float32")]
    fmt = pick_device_format(44100, 2, supported)
    check(fmt.sample_rate == 44100 and fmt.channels == 2,
          "exact match is preferred over backend order")
    check(fmt.dtype == "float32", "picked float32")
    check(fmt.bytes_per_frame == 8, "float32 stereo frame is 8 bytes")

    fmt = pick_device_format(44100, 2, [DeviceFormat(48000, 2, "float32")])
    check(fmt.sample_rate == 48000, "falls back to a supported rate that differs")

    mono = pick_device_format(48000, 1, [DeviceFormat(48000, 2, "int16")])
    check(mono.channels == 2, "mono source is mapped to a stereo device")
    check(mono.bytes_per_frame == 4, "int16 stereo frame is 4 bytes")

    default = pick_device_format(44100, 2)
    check(default.sample_rate == 44100 and default.channels == 2,
          "no support list -> source rate and channel count")
    check(DeviceFormat(44100, 2, "float32").bytes_to_frames(16) == 2,
          "byte<->frame conversion is consistent")


def test_prepare_playback():
    print("\n[playback preparation]")
    x = tone(0.5, 44100, 2)
    fmt = DeviceFormat(44100, 2, "float32")
    buf = prepare_playback(x, 44100, fmt)
    check(isinstance(buf, PlaybackBuffer), "returns a PlaybackBuffer")
    check(buf.frames == x.shape[0], f"frame count preserved ({buf.frames})")
    check(len(buf.data) == buf.frames * 8, "byte length matches the device format")
    check(abs(buf.as_array() - x).max() < 1e-6, "samples survive float32 conversion")
    check(not buf.resampled, "no resampling when rates match")
    check(abs(buf.duration - 0.5) < 1e-6, "duration reported correctly")

    buf48 = prepare_playback(x, 44100, DeviceFormat(48000, 2, "float32"))
    check(buf48.resampled, "resampling flagged when the device rate differs")
    check(abs(buf48.duration - 0.5) < 0.002,
          f"duration preserved across the rate change ({buf48.duration:.4f} s)")
    check(buf48.source_rate == 44100, "source rate remembered")

    mono = prepare_playback(tone(0.2, 44100, 1), 44100, fmt)
    check(mono.frames == int(0.2 * 44100), "mono source keeps its length")
    check(len(mono.data) == mono.frames * 8, "mono source is duplicated to stereo")

    loud = prepare_playback(np.full((100, 2), 4.0), 44100, fmt)
    check(np.abs(loud.as_array()).max() <= 1.0, "samples are clipped for the device")

    faded = prepare_playback(x, 44100, fmt, fade_in_ms=10)
    arr = faded.as_array()
    check(abs(arr[0]).max() < 1e-3, "fade-in starts at silence")
    check(np.abs(arr[2000:4000]).max() > 0.1,
          "fade-in does not affect the rest of the buffer")

    quiet = prepare_playback(x, 44100, fmt, gain_db=-6.0)
    check(abs(quiet.as_array().max() - x.max() * 0.5012) < 5e-3,
          "gain is applied in the buffer when asked")


def test_buffer_read():
    print("\n[buffer reads]")
    x = tone(0.1, 44100, 2)
    fmt = DeviceFormat(44100, 2, "float32")
    buf = prepare_playback(x, 44100, fmt)
    first = buf.read(0, 800)
    check(len(first) == 800, "read returns the requested byte count")
    second = buf.read(800, 800)
    check(second[:16] != first[:16], "consecutive reads return different audio")
    misaligned = buf.read(801, 800)
    check(len(misaligned) == 800, "unaligned offsets are snapped to a frame boundary")
    end = buf.read(len(buf.data) - 8, 16)
    check(len(end) == 8, "reading past the end returns only what is left")
    check(buf.read(len(buf.data), 16) == b"", "reading at the end returns nothing")


def test_gain_bytes():
    print("\n[read-path gain]")
    for dtype in ("float32", "int16", "int32"):
        raw = _to_dtype(np.array([[0.5, -0.25], [0.9, -1.0]]), dtype).tobytes()
        half = apply_gain_bytes(raw, dtype, 0.5)
        np_dtype = np.dtype(dtype)
        before = np.frombuffer(raw, dtype=np_dtype).astype(np.float64)
        after = np.frombuffer(half, dtype=np_dtype).astype(np.float64)
        full = {"float32": 1.0, "int16": 32767.0, "int32": 2147483647.0}[dtype]
        err = float(np.max(np.abs(after / full - 0.5 * before / full)))
        # integer formats round to their own grid; float32 cannot hold an int32
        # sample exactly, so the tolerance follows the format's own resolution
        tol = max(2e-7, 2.0 / full)
        check(err < tol, f"{dtype}: gain halves the signal (max error {err:.2e})")
        check(len(half) == len(raw), f"{dtype}: byte length is preserved")
    loud = _to_dtype(np.array([[0.9], [-0.9]]), "int16").tobytes()
    boosted = np.frombuffer(apply_gain_bytes(loud, "int16", 4.0), dtype=np.int16)
    check(int(np.abs(boosted).max()) == 32767, "int16 boost clips instead of wrapping")
    check(apply_gain_bytes(b"", "float32", 0.5) == b"", "empty input is passed through")


def test_aux_buffers():
    print("\n[auxiliary buffers]")
    fmt = DeviceFormat(48000, 2, "int16")
    sil = silence_buffer(fmt, ms=50)
    check(sil.frames == 2400, "silence buffer has the requested length")
    check(np.all(sil.as_array() == 0), "silence buffer is silent")
    t = test_tone(fmt, freq=1000, seconds=0.25)
    check(abs(t.duration - 0.25) < 1e-3, "test tone has the requested length")
    arr = t.as_array()
    check(np.abs(arr).max() > 0.05, "test tone is audible")


# ---------------------------------------------------------------------------
def test_limits():
    print("\n[import/export limits]")
    check(check_import("/nonexistent/file.wav").ok is False, "missing file is rejected")
    check(check_import("").ok is False, "empty path is rejected")

    tmp = tempfile.mkdtemp()
    empty = os.path.join(tmp, "empty.wav")
    open(empty, "wb").close()
    check(check_import(empty).ok is False, "empty file is rejected")
    check(check_import(tmp).ok is False, "a directory is rejected")

    wav = os.path.join(tmp, "ok.wav")
    save_audio(tone(1.0), ExportSettings(path=wav, format="WAV", subtype="PCM_24"),
               sample_rate=44100)
    v = check_import(wav)
    check(v.ok, "a normal wav passes the import check")

    check(check_loaded(44100, 44100, 2).ok, "1 s stereo at 44.1 kHz is accepted")
    check(not check_loaded(0, 44100, 2).ok, "zero frames is rejected")
    check(not check_loaded(100, 44100, 2).ok, "0.002 s is rejected as too short")
    check(not check_loaded(44100, 4000, 2).ok, "4 kHz sample rate is rejected")
    check(not check_loaded(44100, 44100, 32).ok, "32 channels is rejected")
    check(check_loaded(44100, 44100, 6).ok, "6 channels is accepted with a note")
    check(not check_loaded(44100, 44100, 6).warnings == [], "multichannel carries a note")
    check(check_loaded(44100, 44101, 2).warnings != [], "odd sample rate carries a note")

    v = check_export(path=os.path.join(tmp, "out.wav"), fmt="WAV", sample_rate=44100,
                     frames=44100, channels=2, source_path=wav)
    check(v.ok, "normal wav export passes")
    check(not check_export(path=wav, fmt="WAV", sample_rate=44100, frames=44100,
                           channels=2, source_path=wav).ok,
          "exporting over the source is refused")
    check(not check_export(path=os.path.join(tmp, "out.xyz"), fmt="XYZ",
                           sample_rate=44100, frames=44100, channels=2).ok,
          "unknown extension is refused")
    check(not check_export(path=os.path.join(tmp, "out.mp3"), fmt="MP3",
                           sample_rate=96000, frames=44100, channels=2).ok,
          "mp3 above 48 kHz is refused before rendering")
    v = check_export(path=os.path.join(tmp, "out.ogg"), fmt="OGG", sample_rate=44100,
                     frames=44100, channels=2)
    check(v.warnings != [], "lossy export warns about encoded peaks")
    check(not check_export(path=os.path.join(tmp, "sub", "out.wav"), fmt="WAV",
                           sample_rate=44100, frames=44100, channels=2).ok,
          "missing folder is refused")
    check(not check_export(path=os.path.join(tmp, "out.wav"), fmt="WAV",
                           sample_rate=44100, frames=0, channels=2).ok,
          "zero-frame export is refused")
    check("Export formats" in describe_limits(), "limits summary renders")
    check(rate_is_supported(44100) and not rate_is_supported(1000),
          "rate range check works")


# ---------------------------------------------------------------------------
def test_simple_profile():
    print("\n[simple controls -> profile]")
    neutral = profile_from_simple(SimpleSettings(0, 0, 0, 0))
    stages = neutral.active_stages()
    check(stages == ["eq", "gain_match", "limiter"],
          f"neutral maps to the passthrough stages ({stages})")
    check(neutral.validate() == [], "neutral profile is valid")
    check(all(b.gain_db == 0 for b in neutral.eq_bands),
          "neutral EQ has no gain bands")

    for vals in [(55, 0, 35, 25), (100, 0, 100, 100), (100, 100, 100, 100),
                 (100, -100, 0, 0), (30, -40, 10, 5)]:
        p = profile_from_simple(SimpleSettings(*vals))
        check(p.validate() == [], f"{vals} produces a valid profile")
        check("limiter" in p.active_stages(), f"{vals} keeps the limiter")
        check(p.loudness_match, f"{vals} keeps the level match")

    more = profile_from_simple(SimpleSettings(100, 0, 80, 60))
    check(len(more.active_stages()) > len(neutral.active_stages()),
          "raising the controls adds stages")

    bright = profile_from_simple(SimpleSettings(50, 80, 20, 0))
    dark = profile_from_simple(SimpleSettings(50, -80, 20, 0))
    shelf = lambda p: [b.gain_db for b in p.eq_bands if b.type == "highshelf"]
    check(shelf(bright)[0] > shelf(dark)[0], "tone tilts the high shelf")

    clipped = profile_from_simple(SimpleSettings(500, 500, 500, 500))
    check(clipped.validate() == [], "out-of-range input is clamped, not rejected")
    check(max(abs(b.gain_db) for b in clipped.eq_bands) <= 9.0,
          "clamping caps the EQ at a sane amount")


def test_simple_render():
    print("\n[simple profile rendering]")
    sr = 44100
    x = tone(6.0, sr, 2, amp=0.25)
    for vals in [(0, 0, 0, 0), (55, 0, 35, 25), (100, 100, 100, 100)]:
        prof = profile_from_simple(SimpleSettings(*vals))
        out = render(x, sr, prof).samples
        check(out.shape == x.shape, f"{vals}: length preserved")
        st_in = measure(x, sr)
        st_out = measure(out, sr)
        check(st_out.lufs <= st_in.lufs + 0.3,
              f"{vals}: not louder than input ({st_out.lufs:.2f} <= {st_in.lufs:.2f})")
        check(st_out.true_peak_db <= prof.limiter.ceiling_db + 0.3,
              f"{vals}: within the ceiling ({st_out.true_peak_db:.2f} dBTP)")
        check(np.isfinite(out).all(), f"{vals}: output is finite")

    mono = tone(3.0, sr, 1, amp=0.3)
    prof = profile_from_simple(SimpleSettings(70, 10, 50, 30))
    out = render(mono, sr, prof).samples
    check(out.shape == mono.shape, "mono material stays mono and keeps its length")

    quiet = tone(3.0, sr, 2, amp=0.002)
    out = render(quiet, sr, prof).samples
    gain = integrated_loudness(out, sr) - integrated_loudness(quiet, sr)
    check(gain <= 0.5, f"quiet material is not pushed up ({gain:+.2f} LU)")

    silence = np.zeros((sr, 2))
    out = render(silence, sr, prof).samples
    check(np.abs(out).max() < 1e-6, "silence stays silent")


def main():
    test_device_format()
    test_prepare_playback()
    test_buffer_read()
    test_gain_bytes()
    test_aux_buffers()
    test_limits()
    test_simple_profile()
    test_simple_render()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED"))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
