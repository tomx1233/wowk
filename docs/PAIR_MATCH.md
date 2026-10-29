# `originalfile.ogg` → `ModifiedFile.ogg`: what was actually done

Reproduce with:

```
python3 tools/pair_match.py refaudio/originalfile.ogg refaudio/ModifiedFile.ogg \
        --out-dir analysis/pair --figures
```

Everything below is produced by that command (JSON in `analysis/pair/measurements.json`,
figures in `analysis/pair/`, macro in `analysis/pair/macro.txt`). The tool validates its
own estimators on controls built from the original file every time it runs, so the
numbers are reproducible on any machine with numpy/scipy/soundfile.

## The macro (answer)

```
ChangePitch: Percentage=5.18
ChangeTempo: Percentage=5.06
```

Import in Audacity: **Tools ▸ Macros… ▸ New**, then **Import…** and pick
`macros/ModifiedFile.txt` from this repo (or `analysis/pair/macro.txt` after a run,
or just type the two lines with **Insert**). Apply to the whole track.

**In the app:** `Analysis ▸ Pitch/tempo of two files…` — Ctrl+T, or the
*Pitch/tempo…* button in the top bar. It runs this measurement on any pair and can
copy or save the macro. That tool only measures; the processing chain itself has no
pitch or tempo control.

If you prefer the fields by hand: *Change Pitch → Percentage Change* = 5.18,
*Change Tempo → Percentage Change* = 5.06. Order does not matter — pitch and tempo
commute here.

Optional, only if you want the level to match as well (the modified file is ~2.9 dB
quieter, see below): `Amplify: Amplification=-2.90`.

Full-precision values: pitch **5.1753 %**, tempo **5.0574 %**.

## Measurements

| quantity | value | uncertainty | how |
|---|---|---|---|
| pitch factor | **1.05175** (+87.36 cents, +5.18 %) | ±0.06 % (≈±1 cent) | whitened log-spectrum shift, median of 10 windows × 12 s; window spread 0.27 c |
| speed factor | **1.05057** (+5.06 %) | ±0.02 % | onset-flux envelope, integer lag + parabolic refine; constant offset 90 ms, no drift |
| audio span | 226.650 s → 215.800 s | ±0.05 s | first/last 50 ms frame above −60 dBFS |
| span ratio | 1.05028 | ±0.0005 | cross-check of the speed factor (0.03 % from it) |
| level change | −2.91 dB median | p5…p95 −6.02…+0.93 dB, σ 2.19 dB | 0.4 s windows at time-mapped positions |
| level-vs-level slope | −0.20 | ±0.05 | mild compression-like behaviour, not a compressor |
| loudness / crest | −10.91 → −13.20 LUFS, 13.24 → 16.92 dB | — | `timbreforge.dsp.meter` |

Length check: 226.669 s / 1.05057 = **215.75 s**, and the modified file's audio really
does end at 215.80 s. The extra ~0.58 s at the end of `ModifiedFile.ogg` is digital
silence, not music; likewise it starts with ~50 ms of silence. So the file-length ratio
(1.04732) is *not* the speed factor — the silence has to be excluded first.

## Contradiction with "1.054"

A pitch percentage of 1.054 (i.e. +5.4 %) is **1.0540 = +91.05 cents**. The pair measures
+87.4 cents, and the independent estimators that were validated earlier on this same
material read +86.6 c (low band 60–600 Hz, 58 partial pairs) and +87.1 c (mid band,
5 positions). All three agree within 0.8 cents; 4 cents is outside that.

The estimator itself is not the problem — the self-tests in the run are:

```
true 1.054000 -> measured 1.053999 (-1 ppm)     resample-only control
true 1.050000 -> measured 1.050000 (+0 ppm)
through resample+WSOLA: true 1.054000 -> measured 1.054018 (+17 ppm, spread 0.02 c)
```

The last line matters: a full pitch-shift chain (resample + WSOLA time-stretch, i.e. the
same structure SoundTouch uses) is recovered to 17 ppm, so neither the time-stretch stage
nor the Ogg codec biases the reading.

So either the pitch field was not 5.4, or the effect you used did not deliver what was
typed. 5.18 % is measurably closer. **Which field did you type 1.054 into?** If it was the
Change Pitch "Percentage Change" box, the render we measured is 0.22 % flatter than
requested, which I cannot reproduce with any pitch shifter here and would need your exact
Audacity version to check.

Note that pitch (5.18 %) and tempo (5.06 %) are within 2 cents / 0.1 % of each other. A
single *Change Speed* at +5.06 % would produce very nearly the same file (pitch 1.0506 vs
the measured 1.0518, i.e. 2 cents apart), so a tempo-only edit cannot be excluded by ear —
but it can be excluded by your statement that both effects were used.

## Tonal / textural residual (what is left after pitch correction)

Band levels of the modified file minus the pitch-warped original, normalised so that
1–4 kHz is 0 dB (gain offset −0.51 dB):

| Hz | 47 | 65 | 89 | 123 | 170 | 234 | 323 | 446 | 615 | 848 | 1170 | 1614 | 2226 | 3071 | 4235 | 5842 | 8057 | 11112 | 15327 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dB | −6.5 | −1.8 | −2.3 | −2.1 | −1.6 | −1.5 | −3.0 | −3.4 | −4.5 | −2.7 | −1.2 | −1.3 | +1.5 | +1.0 | +1.4 | +1.8 | +1.7 | −0.1 | +8.5 |

Read that as: a broad, shallow dip of 1.5–4.5 dB from ~90 Hz to ~1.5 kHz, ~+1.5 dB from
2–8 kHz, and two outliers at the extremes (−6.5 dB at 47 Hz, +8.5 dB at 15.3 kHz). There
is no shelf/peak structure that looks like a hand-drawn `FilterCurve`, and the mid-band
residual (±2 dB) is the same order as the difference between two encodes of the same
material, so I am **not** adding an EQ to the macro.

The two extremes are explained by the source material, not by an effect:

```
1/12-octave LTAS, dB re the file's own 1-4 kHz band
  kHz    11.2  12.3  13.4  14.7  15.7  16.2  16.7  17.2  17.7  18.8  19.4  20.7
  orig   -9.1  -8.1 -11.1 -12.7 -17.5 -31.2 -32.1 -32.3 -32.8 -33.0 -33.2 -72.3
  mod   -12.5  -6.4  -6.2  -5.2  -7.7  -8.6 -14.6 -21.7 -25.2 -38.0 -61.0 -101.5
```

The original stops carrying music at ~15.8 kHz and sits on a flat −32 dB codec noise
floor from 16 kHz up (looks like a transcode of a 16 kHz-band-limited source). The
modified file keeps real, tonal, envelope-correlated music up to ~18.5 kHz (flatness
−3.9 dB in 16–17.8 kHz, versus 0.0 dB — white — above the original's cliff; its 2–6 kHz
envelope correlates 0.52 with the 16–17.8 kHz band).

**No pitch, tempo, gain or EQ stage can create 2.5 kHz of extra bandwidth.** So
`ModifiedFile.ogg` was not rendered from the decode of `originalfile.ogg`; it came from a
wider-bandwidth copy of the same recording (or `originalfile.ogg` is a band-limited
transcode of it). That, not an equaliser, is why the top octave reads +8.5 dB. The bass
dip could be either the same master difference or the lossy encoding of the low end; with
one pair I cannot separate those, and I am not going to invent a low-shelf for it.

## What this does not tell you

- No listening was possible in this environment (headless, no audio device). All of the
  above is measurement; I have no listening impression to report.
- The macro reproduces pitch, tempo and length. It will **not** reproduce the top-octave
  difference, because the audio is not there in the original.
- The 2.9 dB level difference and the 3.7 dB crest-factor difference point at the
  modified file being a *less* squashed render, not a compressed or limited one — but
  decoded peak/RMS are strongly affected by the two different lossy encodes, so I would
  not build a dynamics story on them without the original WAVs.
- Two effects at nearly equal percentages cannot be distinguished from one varispeed
  edit by measurement alone; only your memory of the session separates them.
