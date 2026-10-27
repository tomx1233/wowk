"""Run the real GUI classes without Qt (headless simulation).

Run with:  python tests/test_gui_sim.py

Why this exists
---------------
``test_gui_smoke.py`` needs a real Qt platform plugin.  On a container with no
libGL / libEGL / libxkbcommon / ICU, ``QApplication`` cannot be created at all,
so the window can neither be built nor exercised - which is exactly when wiring
bugs (a renamed method, a slot with the wrong argument count, a worker payload
field that does not exist) would slip through.

This test swaps in ``tests/fakeqt/PySide6``: a small stand-in that keeps the parts
of Qt the app actually depends on (widget construction, ``Signal`` delivery,
``QThread.start()`` running ``run()`` inline, a fake PortAudio stream) and turns
everything else into a no-op.  Everything below the GUI - decoding, the DSP
graph, loudness matching, the export writer - is the real code.

What it proves: constructing the window works, the import -> render -> apply ->
A/B -> export pipeline runs end to end, every control can be touched, and the
``sounddevice`` playback callback is exercised.  What it does not prove: that the
widgets paint, size or lay out correctly.  Those still need a real display; see
``tests/test_gui_smoke.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "fakeqt"))     # fake PySide6 first
sys.path.insert(1, REPO)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("TIMBREFORGE_PROFILE", None)

FAILS: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def make_wav(path: str, seconds: float = 4.0, sr: int = 44100) -> str:
    from timbreforge.dsp.audio import ExportSettings, save_audio

    t = np.arange(int(sr * seconds)) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t),
                  0.3 * np.sin(2 * np.pi * 330 * t)], axis=1)
    save_audio(x, ExportSettings(path=path, format="WAV", subtype="PCM_24"),
               sample_rate=sr)
    return path


def phase_without_device(tmp: str, src: str):
    """Everything a user does with no audio hardware at all."""
    from PySide6.QtWidgets import QApplication

    import timbreforge.gui.main_window as mw
    from timbreforge.dsp.biquad import Band
    from timbreforge.gui.theme import apply_theme
    from timbreforge.gui.workers import ExportWorker

    app = QApplication([])
    apply_theme(app)

    print("[construct]")
    win = mw.MainWindow()
    check(win.panel.is_simple(), "starts in the simple view")
    check(win.panel.profile().validate() == [], "startup profile is valid")
    check(win.panel.s_character.value() == 55, "character starts at 55 %")

    print("\n[simple controls]")
    win.panel.s_character.set_value(80)
    win.panel._on_simple_changed()
    check(win.panel.profile().name == "Character 80%",
          f"profile follows the character control ({win.panel.profile().name})")
    check("compressor" in win.panel.profile().active_stages(),
          "raising character engages dynamics")
    win.panel._apply_simple_preset((30, 0, 20, 15))
    check(win.panel.s_tightness.value() == 20, "starting-point presets apply")
    win.panel._reset_profile()
    check(win.panel.s_character.value() == 55, "reset restores the defaults")

    print("\n[import -> render pipeline]")
    win.load_file(src)
    check(win.audio is not None, "audio imported")
    check(win._proc is not None, "preview rendered by the worker")
    check(win._payload is not None and win._payload.play_processed is not None,
          "playback buffer prepared off the UI thread")
    check(win._payload.play_original is not None, "input playback buffer prepared too")
    check(win._proc.shape == win.audio.samples.shape, "whole short file previewed")
    check(win._stats_out is not None, "output measured")
    check(win._stats_out.lufs <= win._stats_in.lufs + 0.3, "not louder than the input")
    check(win.player.backend == "none", "no device -> no backend")
    check(win._audition_btn.isVisible(), "audition fallback is offered")

    print("\n[parameter change]")
    before = win._payload
    win.panel.s_air.set_value(90)
    win.panel._on_simple_changed()
    win._start_render()
    check(win._payload is not before, "a change produces a new render")
    check(win._payload.stats_out.lufs <= win._payload.stats_in.lufs + 0.3,
          "still level matched afterwards")

    print("\n[A/B]")
    win._set_ab("original")
    check(win.player.buffer is not None, "input buffer installed")
    check(win.player.gain_db() <= 0.0, "A/B gain is attenuation only")
    win._set_ab("processed")
    check(win.player.buffer is win._payload.play_processed, "processed buffer installed")
    win.match_cb.setChecked(False)
    check(win.player.gain_db() == 0.0, "turning level match off removes the gain")
    win.match_cb.setChecked(True)

    print("\n[no-device playback path]")
    win._begin_playback()          # offers the audition dialog instead of raising
    check(True, "play with no device does not raise")
    win._audition()
    check(bool(win._audition_tmp) and os.path.exists(win._audition_tmp),
          "audition renders a temporary file")

    print("\n[seek outside the rendered region]")
    win._region = (0, 1000)
    win._on_seek(2.0)
    check(True, "seeking outside the region re-renders without raising")

    print("\n[advanced view]")
    win.panel.mode_btn.setChecked(True)
    check(not win.panel.is_simple(), "advanced view active")
    check(win.panel.eq_editor.bands() != [], "EQ editor mirrors the generated profile")
    win.panel.eq_editor.set_bands([Band("highshelf", 6000, 3.0, 0.707)])
    win.panel._on_eq_changed()
    win._start_render()
    check(win._payload is not None, "render after an advanced EQ change")

    print("\n[bypass]")
    win.panel.bypass_btn.setChecked(True)
    win._start_render()
    check(win.panel.profile().bypass, "bypass reaches the profile")
    win.panel.bypass_btn.setChecked(False)

    print("\n[export]")
    from timbreforge.dsp.audio import ExportSettings

    out = os.path.join(tmp, "out.wav")
    facts: dict = {}
    worker = ExportWorker(win.audio, win.panel.profile(),
                          ExportSettings(path=out, format="WAV", subtype="PCM_24",
                                         source_path=src))
    worker.done.connect(lambda f: facts.update(f))
    worker.start()
    check(os.path.exists(out), "export wrote a file")
    report = win._verify_export(out, facts)
    check("difference" in report, "verification report built")
    check(any("+0.000000 s" in line for line in report.splitlines()),
          "exported duration matches the input")

    win.close()
    return win


def phase_with_device(tmp: str, src: str):
    """The same window, but with a (fake) working PortAudio device."""
    import importlib

    sys.path.insert(0, os.path.join(HERE, "fakeqt", "audiofake"))
    import sounddevice  # noqa: F401  - the fake

    import timbreforge.gui.player as player_mod
    import timbreforge.gui.main_window as mw
    from timbreforge.dsp.graph import Profile
    from timbreforge.gui.workers import ExportWorker

    importlib.reload(player_mod)
    importlib.reload(mw)

    print("\n[with an audio device]")
    win = mw.MainWindow()
    check(win.player.backend == "sd", f"sounddevice backend ({win.player.backend_name})")
    win.load_file(src)
    check(win.player.available, "player reports an available device")

    win._begin_playback()
    check(win.player.is_playing(), "playback starts through the device callback")
    check(win.player.position() > 0, "the callback consumed audio")
    win._set_ab("original")
    win._set_ab("processed")
    win._on_seek(1.0)
    win._on_stop()
    check(not win.player.is_playing(), "stop works")
    check(win.player.play_test_tone(), "test tone plays")

    print("\n[every advanced control]")
    win.panel.mode_btn.setChecked(True)
    p = win.panel
    for widget in (p.dyn_enable, p.comp_enable, p.mb_enable, p.sat_enable,
                   p.tr_enable, p.st_enable, p.lim_enable, p.match_enable):
        widget.setChecked(True)
        widget.setChecked(False)
        widget.setChecked(True)
    p.sat_shaper.setCurrentIndex(2)
    p.sat_os.setCurrentIndex(2)
    p.eq_mode.setCurrentIndex(1)
    p.eq_mode.setCurrentIndex(0)
    p.st_width.set_value(120)
    p.trim.set_value(-2.0)
    win._start_render()
    check(win._payload is not None, "rendered after touching every control")
    check(win._payload.stats_out.true_peak_db
          <= win.panel.profile().limiter.ceiling_db + 0.3,
          "stays under the true-peak ceiling")

    print("\n[shipped profile]")
    prof = Profile.load(os.path.join(REPO, "profiles", "reference_character.json"))
    win.panel.set_profile(prof, win.audio.sample_rate)
    win._start_render()
    check(win.panel.profile().name == prof.name, "loaded profile is active")
    check(win._payload.stats_out.lufs <= win._stats_in.lufs + 0.6,
          f"profile is not louder ({win._payload.stats_out.lufs:.2f} vs "
          f"{win._stats_in.lufs:.2f})")

    print("\n[selection]")
    win.wave.set_selection((0.5, 1.5))
    win._on_selection((0.5, 1.5))
    win._start_render()
    check(win._region[1] - win._region[0] == int(1.0 * win.audio.sample_rate),
          "a selection drives the render region")
    win.wave.set_selection(None)

    print("\n[lossy export]")
    from timbreforge.dsp.audio import ExportSettings

    out = os.path.join(tmp, "out.ogg")
    facts: dict = {}
    worker = ExportWorker(win.audio, win.panel.profile(),
                          ExportSettings(path=out, format="OGG", subtype="VORBIS",
                                         ogg_quality=0.6))
    worker.done.connect(lambda f: facts.update(f))
    worker.start()
    check(os.path.exists(out), "ogg export wrote a file")
    report = win._verify_export(out, facts)
    check("true peak" in report, "encoded true peak reported")
    win.close()
    return win


def main() -> int:
    tmp = tempfile.mkdtemp()
    src = make_wav(os.path.join(tmp, "song.wav"))
    try:
        phase_without_device(tmp, src)
        phase_with_device(tmp, src)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\nSIMULATION CRASHED")
        return 1
    print("\n" + ("ALL SIMULATION CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} CHECK(S) FAILED"))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
