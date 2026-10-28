# Timbreforge

A standalone desktop audio-character processor: import a file, preview a
non-destructive processing graph, compare it against the untouched input at
matched loudness, and export through a Save-As dialog.

It exists to reproduce one specific goal — the *tonal and textural* character of
the reference recording (`NewTest.ogg`) using ordinary material — **without**
pitch, tempo or speed changes, and **without** becoming a loudness maximiser.

---

## What changed in 1.1 (and what was wrong in 1.0)

1.0 shipped three problems the release notes did not admit to:

| Symptom | Cause | Fix in 1.1 |
|---|---|---|
| Play did nothing on Windows | `PySide6-Essentials` was pinned, and it contains **no Qt Multimedia**, so the only playback backend that existed was missing | `sounddevice` is now a required dependency and the primary backend; Qt Multimedia is a secondary backend; if neither works the app renders the preview to a temp file and opens it in your own player ("Audition in player") |
| The window froze after every slider move | every change re-rendered the region *and* re-measured it on the UI thread, and changes were queued instead of superseded | decoding/measuring/rendering moved to worker threads, changes are debounced (300 ms), an in-flight render is cancelled rather than queued, buffers are pre-prepared off the UI thread, and A/B gain is applied by the player so switching sides never re-renders |
| "Too complicated" | the first screen was the full 10-stage editor | the app now opens in a **simple view** with five controls; the full editor is one click away and is loaded with the profile the simple controls generated |

If 1.0's exported files did not sound like the reference, that is *not* fixed by
1.1 and cannot be: the reference recordings were never provided, so the shipped
profile is still an unvalidated hypothesis (see the status below). Send the four
files and the analysis commands turn that into evidence.

---

## Status: read this first

**The reference files are not in this repository.** `NewTest.ogg`,
`newtestmodified3.ogg`, `TestSong.ogg` and `newtestoriginal3.ogg` were not
supplied with the workspace, so:

* nothing in this repo is a *verified* measurement of those files;
* the shipped profile (`profiles/reference_character.json`) is a **hypothesis**
  built from the figures the previous investigation reported, plus the quality
  rules in the brief. It is deliberately conservative and is *not* claimed to be
  a reproduction;
* every number in `examples/` was produced from synthetic demo material and is
  labelled as such.

The analysis tooling in this repo is what turns the hypothesis into evidence.
Drop the four files next to this README and run the commands in
[Reference analysis](#reference-analysis) — that produces the verified report,
the method comparison and a fitted starting-point profile. See
`docs/ANALYSIS.md` for what has and has not been established.

---

## Install and launch

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt

./run.sh                 # macOS / Linux
run.bat                  # Windows

# or directly
python -m timbreforge
python -m timbreforge "/path/to/song.ogg"   # open with a file loaded
```

Packages used: `numpy`, `scipy`, `soundfile` (libsndfile), `PySide6`,
`sounddevice`. `matplotlib` is only needed for the analysis figures. Nothing else
is required — no Audacity, no Discord/bot configuration, no network access.

### Audio output

Playback tries three backends, in this order:

1. **`sounddevice`** (PortAudio). This is the normal path and the reason it is a
   required dependency: the wheel bundles PortAudio on Windows and macOS.
2. **Qt Multimedia**, if the installed PySide6 build contains it. Note that
   `PySide6-Essentials` — which earlier versions of this project pinned — does
   **not** include Qt Multimedia, so this backend simply did not exist there.
   That was the cause of "play plays nothing" on Windows.
3. **None.** The app then shows an *Audition in player* button (also
   `View ▸ Audition preview in system player`, Ctrl+Shift+P): it renders the
   preview region to a temporary WAV and opens it in your system's audio player.
   Rendering, measurement and export never depend on an audio device.

Check the current backend any time in the status line, or with
`View ▸ Test audio output`.

If playback is still silent:

```bash
python -c "import sounddevice as sd; print(sd.query_devices()); print(sd.default.device)"
```

* **prints a traceback** → PortAudio is missing. `pip install --force-reinstall sounddevice`
  (Windows/macOS wheels bundle it); on Debian/Ubuntu also `sudo apt install libportaudio2`.
* **lists devices but the wrong one is default** → choose the right output device
  in the OS sound settings; the app follows the system default.
* **device busy / sample-rate error** → close other apps that hold the device in
  exclusive mode, or select a different default device. The app negotiates the
  device's supported rate and resamples for monitoring only — the exported file
  always keeps the input's sample rate.
* **Windows + Bluetooth headset** → those often expose only a "hands-free"
  profile; pick the stereo output device in Windows settings.

---

## Using the app

1. **Import** — `File ▸ Import` (Ctrl+O), drag a file onto the window, or pass it
   on the command line. WAV, FLAC, Ogg Vorbis and MP3 are supported through
   libsndfile (ffmpeg is used as a fallback decoder if installed). Decoding and
   measuring happen on a worker thread, so the window stays responsive and the
   Cancel button works.
2. **Preview** — the render happens on a worker thread. Files longer than 45 s
   preview a 30 s window around the playhead (tick *Preview whole file* to render
   everything). Drag on the waveform to select a region; double-click clears it.
   `Ctrl` + wheel zooms, wheel scrolls, `F` fits.
3. **A/B** — buttons `A · Input` / `B · Processed` swap what you hear without
   losing the playhead. *Level-match A/B* attenuates the louder side (never
   boosts) so the comparison is honest. `Bypass` in the profile panel renders
   the input through the same path for a bit-identical reference.
4. **Adjust** — the app opens in the **simple view**, five controls:

   | Control | What it does |
   |---|---|
   | Character | overall amount: tone shaping, level-dependent brightness, gentle dynamics, a little harmonic texture. 0 % renders neutral. |
   | Tone | −100 darker/fuller … +100 brighter/leaner |
   | Tightness | how much the dynamics are controlled (0 % leaves them alone) |
   | Air | extra top-end lift on top of Character |
   | Output | extra gain on top of the automatic level match (leave at 0) |

   The four *Starting points* buttons (Subtle → Maximum) are quick presets, and
   each slider's tooltip explains what it changes. These five controls do **not**
   form a second DSP path: they deterministically generate a complete profile
   (see `timbreforge/dsp/graph.py::profile_from_simple`), which the same engine
   renders. *Advanced view* loads that generated profile into the full editor, so
   the two views can never disagree about what you are hearing.

   Every control edits one parameter of the one processing graph (see below).
   Re-renders are debounced (~300 ms) and an older render is cancelled rather
   than queued; the status line lists the active stages, any level-matching
   applied, and the headroom left below the ceiling.
5. **Export** — `Export…` (Ctrl+E) picks format, bit depth/quality, sample rate
   and destination, renders the **whole** file with the same engine, then
   re-decodes the written file and reports timing, true peak, loudness, crest,
   correlation and any clipping. The imported file can never be overwritten;
   an existing destination asks first; cancel leaves no partial file behind.
6. **Profiles** — `Save as…` / `Load…` store the complete graph as JSON.
   Three are shipped in `profiles/`:
   * `neutral_level_matched.json` — no processing, level matched, −1 dBTP
   * `gentle_character.json` — broad tilt + one soft compressor
   * `reference_character.json` — the hypothesis. Load it with
     `File ▸ Processing ▸ Load`, by pointing `TIMBREFORGE_PROFILE` at it, or from
     the preset list. It is *not* loaded at startup any more: the app now starts
     in the simple view.

---

## The processing graph

One graph, used by preview and export alike, in a fixed order:

| # | Stage | What it does | Why it sits there |
|---|---|---|---|
| 1 | Static EQ | Up to N RBJ biquads, minimum phase by default, linear-phase FIR option | tone first, so later dynamics react to the intended balance |
| 2 | Dynamic EQ | level-dependent peaking bands (de-esser style) | tonal change that only happens on loud material |
| 3 | Transient shaping | attack/sustain gain curve | before compression, or the compressor undoes it |
| 4 | Multiband dynamics | per-band compression around a complementary linear-phase crossover | band-dependent density |
| 5 | Broadband compressor | feed-forward, RMS or peak detector, linked or independent | overall density; manual makeup only |
| 6 | Saturation | oversampled waveshaping, level-compensated by attenuation only | after the dynamics, so it is not gain-modulated |
| 7 | Stereo width | side-signal gain only | mono sum stays bit-identical |
| 8 | Output trim | user gain | an explicit decision, not automatic |
| 9 | Loudness match | brings the render to the **input's** loudness (≤, capped at ±18 dB) | A/B fairness; the default cannot get louder |
| 10 | True-peak limiter | offline look-ahead, −1 dBTP default | the last thing before the file is written |

Guarantees enforced by code and checked by tests:

* **No pitch, tempo, speed or time-stretch anywhere.** Output length equals input
  length exactly (`render()` raises if it does not). The only length difference
  in an export is container padding from the codec, which is reported.
* **Not a loudness maximiser.** The loudness match only attenuates by default;
  positive output trim is a deliberate user action. Lossy overshoot is caught by
  re-measuring the encoded file.
* **Transparent where it should be.** With neutral settings the multiband stage
  sums back to the input to within 1e-16, the stereo stage leaves the mid signal
  bit-identical, and the whole graph at default settings is level-match + limiter.
* **Mono compatibility** is measured (mono fold-down change is reported).

---

## Reference analysis

With `TestSong.ogg` (original) and `NewTest.ogg` (processed) present:

```bash
# 1. Is it the same recording, and what actually differs?
python tools/analyze_pair.py --original TestSong.ogg --reference NewTest.ogg \
    --out-dir analysis/NewTest --figures

# 2. Compare processing families at matched loudness
python tools/compare_methods.py --original TestSong.ogg --reference NewTest.ogg \
    --out analysis/method_comparison.md

# 3. Optional: same, for the second pair
python tools/analyze_pair.py --original newtestoriginal3.ogg \
    --reference newtestmodified3.ogg --out-dir analysis/NewTest3 --figures
```

The same analysis is available inside the app under `Analysis ▸ Reference pair
analysis…`, where it can also fit a starting-point EQ profile from the measured
apparent transfer.

### Recovering an edit someone else made

`tools/pair_match.py` is the narrow tool for "these two files differ only by a
speed/pitch change — what exactly was done?". It estimates the pitch and tempo
factor between them, validates both estimators on controls built from the
original on every run (resample-only, and resample + WSOLA, accurate to ~20 ppm),
and prints the Audacity macro that reproduces the edit:

```bash
python3 tools/pair_match.py originalfile.ogg ModifiedFile.ogg \
    --out-dir analysis/pair --figures
```

It also reports the audio span of each file (so leading/trailing silence cannot
be mistaken for a speed change), the level residual, the pitch-compensated tone
residual, and the bandwidth of each source. A worked example, including how to
tell a real equaliser move from a codec/source difference, is in
`docs/PAIR_MATCH.md`.

What the analysis reports, and why each part exists:

* **same recording?** — normalised correlation at the measured lag, per-segment
  lag spread (drift), and a tuning estimate from the spectra. Duration alone is
  never used as proof.
* **alignment** — a waveform search over the full plausible range with a
  parabolic sub-sample refinement, cross-checked against an onset-envelope
  correlation, with an explicit ambiguity flag for repetitive material.
* **tuning** — the frequency ratio that best aligns the two log-spectra, so a
  genuine pitch shift or stretch shows up as cents ≠ 0.
* **levels, dynamics, stereo, spectrum** — the same measurements applied to both
  files, plus the reference re-measured at the original's loudness so gain does
  not dominate the comparison.
* **static or level-dependent?** — band ratios per loudness quartile with the
  overall gain removed. A static filter gives the same pattern at every level; a
  dynamic or nonlinear process does not. This is the test that separates "EQ" from
  "EQ + something that only happens when it is loud".
* **level relation** — the slope of reference level versus original level
  (compression signature) and the implied ratio.

---

## Tests

```bash
python tests/test_dsp.py          # graph invariants (pitch/length, loudness, limiter, stereo, export)
python tests/test_analyze.py      # analysis recovers a known lag, EQ, compression, pitch error
python tests/test_playback.py     # playback buffers, device negotiation, import/export limits, simple controls
python tests/test_gui_sim.py      # the real window + workers, run against a stand-in Qt (no display needed)
python tests/test_gui_smoke.py    # the real window with the real Qt (needs a display or offscreen platform)
python tools/check_qt_api.py      # static check of every Qt call against the PySide6 stubs
python tools/check_wiring.py      # every attribute the window uses on the panel/player/workers exists
```

`test_playback.py` deliberately imports no Qt, so it runs anywhere numpy does.
`test_gui_sim.py` does import Qt — a small stand-in in `tests/fakeqt/PySide6` that
keeps widget construction, signal delivery, `QThread.start()` (running `run()`
inline) and a fake PortAudio stream, and no-ops everything visual. Everything
below the GUI is the real code, so it exercises import → render → apply → A/B →
export end to end. It cannot tell you whether the window *looks* right.
`test_gui_smoke.py` needs a display or `QT_QPA_PLATFORM=offscreen`; the
development sandbox this release was built in has no Qt platform libraries at all
(no libGL/libEGL/libxkbcommon/ICU), so the GUI could not be executed there. As a
substitute, `tools/check_qt_api.py` parses the shipped PySide6 `.pyi` stubs and
verifies every class, method, property and enum member the GUI touches exists.
A clean run means "no known-bad Qt call", not "the window was seen rendering" —
run the smoke test on your own machine if you want the latter.

---

## Layout

```
timbreforge/
  dsp/        audio.py  meter.py  biquad.py  dynamics.py  nonlinear.py
              stereo.py  graph.py  analyze.py      <- the whole engine
  gui/        app.py  main_window.py  panel.py  controls.py  widgets.py
              player.py  workers.py  export_dialog.py  analysis_dialog.py  theme.py
  cli.py      headless render / measure / compare
tools/        make_demo.py  analyze_pair.py  compare_methods.py  render_example.py
              pair_match.py  (estimate an external pitch/tempo edit)
profiles/     neutral_level_matched.json  gentle_character.json  reference_character.json
examples/     demo render, level-matched A/B, validation report, spectrograms
docs/         ANALYSIS.md (findings vs hypotheses)  METHODS.md (families tested)
              PAIR_MATCH.md (worked pitch/tempo recovery)
tests/        test_dsp.py  test_analyze.py  test_playback.py  test_gui_sim.py
```

---

## Known limitations

* **Untested file size.** Everything is float64 in memory: a 2.5 minute stereo
  file needs roughly 1 GB of peak RAM during export with all stages enabled
  (measured 1.07 GB / 149 s). Preview windows keep editing responsive.
* **Lossy codecs.** Ogg Vorbis and MP3 overshoot the requested true-peak ceiling
  slightly (measured −2.25 dBTP against a −2.37 dBTP render). The export report
  says so; lower the ceiling if you need a hard guarantee in a lossy file.
* **No exact reference match is claimed** and none may be achievable: the
  reference's reported density (crest ≈ 2.6 dB, most samples near full scale)
  can only be reached with the loudness maximising this project refuses. The app
  implements the closest controlled, non-harsh result instead.
* **Ogg writing** is done in blocks because this environment's libsndfile
  segfaults on a single very large Vorbis write (reproduced above 2^22 samples).
  The block writer keeps long exports working.
