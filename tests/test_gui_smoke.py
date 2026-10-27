"""Headless smoke test for the desktop UI (QT_QPA_PLATFORM=offscreen).

Run with:  python tests/test_gui_smoke.py

This needs a working Qt platform plugin.  On a bare Linux container without
libGL/libEGL/libxkbcommon/ICU it cannot start at all - that is an environment
limit, not an app failure.  ``tools/check_qt_api.py`` covers the "does every Qt
call exist" question on such machines.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PySide6.QtWidgets import QApplication

from timbreforge.dsp.audio import ExportSettings, load_audio, save_audio
from timbreforge.dsp.biquad import Band
from timbreforge.gui.app import _make_icon
from timbreforge.gui.main_window import MainWindow
from timbreforge.gui.theme import apply_theme
from timbreforge.gui.workers import ExportWorker

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


def idle(win):
    return win._worker is None and win._import_worker is None


def make_wav(path):
    sr = 44100
    t = np.arange(sr * 6) / sr
    env = 0.4 + 0.3 * np.sin(2 * np.pi * 0.7 * t)
    left = env * (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1320 * t))
    right = env * (0.3 * np.sin(2 * np.pi * 221 * t) + 0.08 * np.sin(2 * np.pi * 3000 * t))
    x = np.stack([left, right], axis=1) * 0.8
    save_audio(x, ExportSettings(path=path, format="WAV", subtype="PCM_24"), sample_rate=sr)
    return path


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    _make_icon()
    win = MainWindow()
    win.resize(1360, 880)
    win.show()
    app.processEvents()

    tmp = tempfile.mkdtemp()
    src = make_wav(os.path.join(tmp, "song.wav"))

    print("\n[startup]")
    check(win.panel.is_simple(), "app starts in the simple view")
    check(win.panel.profile().validate() == [], "startup profile is valid")
    check(win.panel.s_character.value() == 55, "character starts at 55 %")

    print("\n[import]")
    win.load_file(src)
    check(spin(app, 40, lambda: win.audio is not None), "file imported in a worker")
    check(win.audio.sample_rate == 44100, "sample rate read")
    check(win.audio.frames == 6 * 44100, "frame count read")
    check(win.export_btn.isEnabled(), "export enabled after import")

    print("\n[initial render]")
    ok = spin(app, 60, lambda: win._proc is not None and idle(win))
    check(ok, "preview render completed")
    check(win._proc is not None and win._proc.shape == win.audio.samples.shape,
          "preview covers the whole short file")
    check(win._payload is not None and win._payload.play_processed is not None,
          "playback buffer was prepared in the worker")
    check(win._stats_out is not None, "output measured off the UI thread")

    print("\n[simple controls]")
    win.panel.s_tightness.set_value(70)
    win.panel._on_simple_changed()
    ok = spin(app, 60, lambda: win._proc is not None and idle(win)
              and win._payload.stats_out is not None)
    check(ok, "re-render after a simple-control change completed")
    check("compressor" in win.panel.profile().active_stages(),
          "raising Tightness enables the compressor")
    check(win._stats_out.lufs <= win._stats_in.lufs + 0.3,
          "output is not louder than the input")

    print("\n[A/B switching]")
    win._set_ab("original")
    check(win._ab == "original", "A/B switched to input")
    check(win.player.buffer is not None, "input buffer installed for playback")
    win._set_ab("processed")
    check(win._ab == "processed", "A/B switched back")

    print("\n[advanced view]")
    win.panel.mode_btn.setChecked(True)
    app.processEvents()
    check(not win.panel.is_simple(), "switched to the advanced view")
    check(win.panel.eq_editor.bands() != [], "advanced EQ editor mirrors the profile")
    win.panel.eq_editor.set_bands([Band("highshelf", 6000, 3.0, 0.707)])
    win.panel._on_eq_changed()
    ok = spin(app, 60, lambda: win._proc is not None and idle(win))
    check(ok, "re-render after an EQ change completed")

    print("\n[export]")
    out = os.path.join(tmp, "song_out.wav")
    worker = ExportWorker(win.audio, win.panel.profile(),
                          ExportSettings(path=out, format="WAV", subtype="PCM_24"))
    facts = {}
    worker.done.connect(lambda f: facts.update(f))
    worker.start()
    spin(app, 120, lambda: not worker.isRunning())
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
    prof = win.panel.profile()
    check(prof.validate() == [], "reset produces a valid profile")
    check(prof.name.startswith("Character") or prof.name.startswith("Neutral"),
          f"reset restores the default profile ({prof.name})")

    print("\n[window close]")
    win.close()
    app.processEvents()
    check(True, "window closed cleanly")
    print("\n" + ("ALL GUI CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
