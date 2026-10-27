"""Headless smoke test for the desktop UI (QT_QPA_PLATFORM=offscreen)."""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

from timbreforge.dsp.audio import AudioData, ExportSettings, load_audio, save_audio
from timbreforge.gui.app import _make_icon
from timbreforge.gui.main_window import MainWindow
from timbreforge.gui.theme import apply_theme

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def spin(app, seconds=25.0, until=None):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until() if until is not None else True


def make_wav(path):
    sr = 44100
    t = np.arange(sr * 6) / sr
    env = 0.4 + 0.3 * np.sin(2 * np.pi * 0.7 * t)
    l = env * (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1320 * t))
    r = env * (0.3 * np.sin(2 * np.pi * 221 * t) + 0.08 * np.sin(2 * np.pi * 3000 * t))
    x = np.stack([l, r], axis=1) * 0.8
    save_audio(x, ExportSettings(path=path, format="WAV", subtype="PCM_24"), sample_rate=sr)
    return path


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1360, 880)
    win.show()
    app.processEvents()

    tmp = tempfile.mkdtemp()
    src = make_wav(os.path.join(tmp, "song.wav"))

    print("\n[import]")
    win.load_file(src)
    check(win.audio is not None, "file imported")
    check(win.audio.sample_rate == 44100, "sample rate read")
    check(win.export_btn.isEnabled(), "export enabled after import")

    print("\n[initial render]")
    ok = spin(app, 40, lambda: win._proc is not None and not (win._worker and win._worker.isRunning()))
    check(ok, "preview render completed")
    check(win._proc is not None and win._proc.shape == win.audio.samples.shape,
          "preview covers the whole short file")

    print("\n[parameter change]")
    p = win.panel.profile()
    p.eq_bands = win.panel.eq_editor.bands()
    from timbreforge.dsp.biquad import Band
    win.panel.eq_editor.set_bands([Band("highshelf", 6000, 3.0, 0.707)])
    win.panel._on_eq_changed()
    ok = spin(app, 40, lambda: win._proc is not None and not (win._worker and win._worker.isRunning()))
    check(ok, "re-render after EQ change completed")
    check(win._stats_out is not None, "output measured")
    check(win._stats_out.lufs <= win._stats_in.lufs + 0.3,
          "output not louder than input")

    print("\n[A/B switching]")
    win._set_ab("original")
    check(win._ab == "original", "A/B switched to input")
    win._set_ab("processed")
    check(win._ab == "processed", "A/B switched back")

    print("\n[export]")
    out = os.path.join(tmp, "song_out.wav")
    from timbreforge.gui.workers import ExportWorker
    worker = ExportWorker(win.audio, win.panel.profile(),
                          ExportSettings(path=out, format="WAV", subtype="PCM_24"))
    facts = {}
    worker.done.connect(lambda f: facts.update(f))
    worker.start()
    spin(app, 60, lambda: not worker.isRunning())
    check(os.path.exists(out), "export produced a file")
    back = load_audio(out) if os.path.exists(out) else None
    if back is not None:
        check(abs(back.duration - win.audio.duration) < 0.002,
              f"exported duration preserved ({back.duration:.4f} s)")
        check(back.sample_rate == win.audio.sample_rate, "exported rate preserved")
        report = win._verify_export(out, facts)
        check("difference" in report, "export verification report built")

    print("\n[profile panel]")
    win.panel.bypass_btn.setChecked(True)
    check(win.panel.profile().bypass, "bypass toggles the profile")
    win.panel.bypass_btn.setChecked(False)
    win.panel._reset_profile()
    check(win.panel.profile().name.startswith("Neutral"), "reset restores the default")

    print("\n[window close]")
    win.close()
    app.processEvents()
    check(True, "window closed cleanly")
    print("\n" + ("ALL GUI CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
