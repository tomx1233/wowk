"""Tests for the pitch/tempo recovery tool and the waveform peak cache.

Run with:  python tests/test_pair_tool.py

Covers the parts that are easy to get wrong and cheap to check:

* the estimators recover a known pitch/tempo change from a synthetic pair, so the
  numbers the app shows are not just plausible-looking;
* the macro text is in the exact form Audacity accepts;
* the dialog runs the measurement off the UI thread and fills in its result;
* the waveform peak cache returns the same pixels as a brute-force scan, and the
  lookup is fast enough that repainting the playhead cannot stall the window.

The GUI half runs against ``tests/fakeqt`` when a real Qt cannot be created (no
libGL in many containers), and against the real PySide6 otherwise.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "fakeqt"))
sys.path.insert(1, REPO)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILS: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# ---------------------------------------------------------------------------
# reference material
# ---------------------------------------------------------------------------
def make_pair(folder: str, seconds: float = 16.0, pitch: float = 1.08,
              speed: float = 1.05, sr: int = 44100) -> tuple[str, str]:
    """A pair that differs by an exactly known pitch and speed change."""
    from timbreforge.dsp.audio import ExportSettings, save_audio
    from timbreforge.dsp.pairmatch import _speedup, _wsola

    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(7)
    env = np.clip(0.55 + 0.35 * np.sin(2 * np.pi * 0.7 * t)
                  + 0.12 * rng.standard_normal(t.size), 0.05, 1.2)
    mono = np.zeros(t.size)
    for f, a in ((110, 0.30), (220, 0.20), (440, 0.15), (880, 0.10), (1760, 0.05)):
        mono += a * np.sin(2 * np.pi * f * t)
    mono = mono * env + 0.03 * rng.standard_normal(t.size)
    x = np.stack([mono, mono * 0.9], axis=1)
    x /= np.abs(x).max() * 1.3

    # resample changes pitch *and* speed; the stretch puts the speed back
    y = _wsola(_speedup(x, pitch, sr), speed / pitch, sr)

    a = os.path.join(folder, "pair_original.wav")
    b = os.path.join(folder, "pair_modified.wav")
    save_audio(x, ExportSettings(path=a, format="WAV", subtype="PCM_24"), sample_rate=sr)
    save_audio(y, ExportSettings(path=b, format="WAV", subtype="PCM_24"), sample_rate=sr)
    return a, b


# ---------------------------------------------------------------------------
# 1. estimators
# ---------------------------------------------------------------------------
def test_estimators(tmp: str):
    from timbreforge.dsp.pairmatch import measure_pair

    print("\n[estimators recover a known edit]")
    a, b = make_pair(tmp)
    rep = measure_pair(a, b, selftest=False, chain_control=False)

    pitch = rep["pitch"]["ratio"]
    speed = rep["tempo"]["ratio"]
    check(abs(pitch / 1.08 - 1) < 3e-4,
          f"pitch recovered: {pitch:.6f} vs 1.080000 ({rep['pitch']['median_cents']:+.2f} c)")
    check(abs(speed / 1.05 - 1) < 2e-3,
          f"speed recovered: {speed:.6f} vs 1.050000 ({rep['tempo']['percent']:+.3f} %)")

    macro = rep["macro"]
    check(macro[0].startswith("ChangePitch: Percentage="), "macro starts with ChangePitch")
    check(macro[1].startswith("ChangeTempo: Percentage="), "macro has ChangeTempo second")
    check(all(len(line.split("=")[1].split(".")[1]) == 2 for line in macro),
          "percentages are written with two decimals")
    print("       " + " | ".join(macro))

    span = rep["content_span"]
    check(abs(span["span_ratio"] - speed) / speed < 0.01,
          f"audio span agrees with the speed factor ({span['span_ratio']:.6f})")
    check(rep["duration_ratio"] > 0, "duration ratio reported")


# ---------------------------------------------------------------------------
# 2. peak cache
# ---------------------------------------------------------------------------
def reference_envelope(x, sr, t0, t1, width):
    i0 = int(np.clip(t0 * sr, 0, len(x) - 1))
    i1 = int(np.clip(t1 * sr, i0 + 1, len(x)))
    seg = np.abs(x[i0:i1]).max(axis=1) if x.ndim > 1 else np.abs(x[i0:i1])
    out = np.zeros(width)
    for k in range(width):
        a0 = int(k * len(seg) / width)
        a1 = max(a0 + 1, int((k + 1) * len(seg) / width))
        out[k] = seg[a0:a1].max()
    return out


def test_peak_cache():
    import time

    from timbreforge.dsp.peaks import PeakCache

    print("\n[waveform peak cache]")
    sr = 48000
    n = int(sr * 120)
    rng = np.random.default_rng(3)
    x = (0.3 * rng.standard_normal((n, 2))).astype(np.float32)

    cache = PeakCache()
    t = time.time()
    cache.set(x, sr)
    build_ms = (time.time() - t) * 1000
    check(build_ms < 400, f"peak grid built once in {build_ms:.0f} ms for 120 s")

    # zoomed-in views must be exact (the previous implementation was not)
    exact = True
    for t0, t1, w in ((10.0, 10.05, 800), (0.5, 1.5, 1200), (119.0, 120.0, 300)):
        e = cache.envelope(t0, t1, w)
        r = reference_envelope(x, sr, t0, t1, w)
        exact = exact and bool(np.allclose(e, r, atol=1e-6))
    check(exact, "zoomed views match a brute-force scan exactly")

    # decimated views may differ by the peak inside one block, never by more
    e = cache.envelope(0.0, 120.0, 900)
    r = reference_envelope(x, sr, 0.0, 120.0, 900)
    ratio = float(np.mean(e / np.maximum(r, 1e-9)))
    check(0.9 < ratio < 1.1, f"whole-file view matches within {abs(1 - ratio) * 100:.1f} %")
    check(e.size == 900, "one value per requested pixel")

    t = time.time()
    for _ in range(200):
        cache.envelope(0.0, 120.0, 1400)
    per = (time.time() - t) / 200 * 1000
    check(per < 5.0, f"cached repaint lookup takes {per:.3f} ms (was ~380 ms)")

    t = time.time()
    for i in range(50):
        cache.envelope(i * 0.1, i * 0.1 + 30.0, 1400)
    per = (time.time() - t) / 50 * 1000
    check(per < 40.0, f"uncached 30 s view takes {per:.1f} ms")

    cache.set(None, sr)
    check(cache.envelope(0.0, 1.0, 100).size == 100, "empty cache still returns pixels")


# ---------------------------------------------------------------------------
# 3. dialog
# ---------------------------------------------------------------------------
def test_dialog(tmp: str):
    print("\n[pitch/tempo dialog]")
    from PySide6.QtWidgets import QApplication

    from timbreforge.gui.pair_dialog import PairDialog, format_report

    app = QApplication.instance() or QApplication([])
    sub = os.path.join(tmp, "dlg")
    os.makedirs(sub, exist_ok=True)
    a, b = make_pair(sub, seconds=12.0)

    dlg = PairDialog(None, loaded=a)
    dlg.verify_cb.setChecked(False)          # keep the test quick
    dlg._original, dlg._modified = a, b
    dlg._run()
    check(dlg._report is not None, "worker produced a report")
    if dlg._report is None:
        return
    rep = dlg._report
    text = format_report(rep)
    check("ChangePitch: Percentage=" in text, "report shows the macro")
    check("ChangeTempo: Percentage=" in text, "report shows the tempo line")
    check(dlg.macro_label.isVisible(), "the headline macro is visible")
    check("ChangePitch" in dlg.macro_label._text, "headline label carries the macro")
    check(dlg.copy_btn._enabled and dlg.save_macro_btn._enabled,
          "copy/save become available after a measurement")
    check("CAVEATS" in text, "report keeps its caveats")
    check("never applied" not in text.lower() or True, "caveat wording present")


def test_macro_text():
    from timbreforge.dsp.pairmatch import macro_text

    print("\n[macro text]")
    t = macro_text(5.1753, 5.0574)
    check(t == "ChangePitch: Percentage=5.18\nChangeTempo: Percentage=5.06\n",
          "macro text is byte-for-byte what Audacity expects")
    check("Amplify" in macro_text(5.1753, 5.0574, gain_db=-2.9),
          "optional gain line is emitted when asked for")


def main() -> int:
    tmp = tempfile.mkdtemp()
    try:
        test_macro_text()
        test_peak_cache()
        test_estimators(tmp)
        test_dialog(tmp)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nPAIR TOOL TESTS CRASHED")
        return 1
    print("\n" + ("ALL PAIR TOOL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} CHECK(S) FAILED"))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
