# Analysis report

Scope: what has been *verified*, what is still a *hypothesis*, and what the app
therefore does. Written to be useful even though the reference files were not
supplied.

---

## 0. Evidence status (the important part)

| Item | Status |
|---|---|
| `NewTest.ogg`, `newtestmodified3.ogg`, `TestSong.ogg`, `newtestoriginal3.ogg` | **Not present** in this workspace. Nothing below is a measurement of them. |
| All figures quoted from the previous investigation | **Unverified claims.** Reproduced here as claims, not as facts. |
| Behaviour of this implementation (length, pitch, loudness policy, limiter, stereo, export) | **Verified** — `tests/test_dsp.py` |
| Behaviour of the analysis code (does it recover a known lag / EQ / compression / pitch error?) | **Verified** — `tests/test_analyze.py` |
| The shipped profile | **Hypothesis**, conservative, explicitly not a reproduction. |

So the honest statement is: *the measuring apparatus is built and tested; the
reference has not been measured.*

---

## 1. What the previous attempt reported, and how each item should be read

Each claim is listed with what it can and cannot establish, and the specific test
in `tools/analyze_pair.py` that decides it.

| Claim | Can it establish the process? | How it is re-checked here |
|---|---|---|
| Both stereo Ogg Vorbis, ≈44.1 kHz | Only container facts | `sf.info` per file, reported in the header |
| Durations 149.3346 / 149.3361 s, Δ ≈ 67 samples (1.5 ms) | A hint at codec padding, **not** proof of identical timing | `geometry.frame_delta`; duration is never used as proof |
| "No convincing timing drift" | Only true if measured per segment | `alignment.segments`: lag per eighth, drift in ppm, `consistent_offset` |
| Chroma correlation ≈0.91 at zero semitones | Chroma is coarse: it survives ±30 cents and heavy timbre change. It cannot prove identical fine tuning | `tuning`: spectral-peak ratio in cents, plus a 200-cent scan |
| RMS 0.3086 → 0.8174 (+8.46 dB) | Describes loudness only; implies the reference is far louder | `levels_*` (RMS, LUFS, LRA) on both, then the reference re-measured at matched loudness |
| Peaks 2.92 → 1.1065, crest 19.5 → 2.63 dB, many samples at full scale | Suggests heavy limiting/clipping/decode overshoot, but **does not identify the mechanism** | `clipped_fraction`, crest, envelope range; the mechanism is discussed, never assumed |
| Envelope p95–p5 21.9 → 1.9 dB | Says the dynamics were crushed | `envelope_range_db`; the app explicitly refuses to reproduce it |
| Centroid 386 → 1958 Hz, less sub, more mid/HF, higher flatness | Brightening | `band_delta_db_at_match`, `third_octave`, `spectral_flatness`; flatness separates "brighter" from "noisier" |
| HF decorrelation, L/R 0.968 → 0.863, S/M −16.75 → −12.48 dB | Stereo change, possibly from band-limited processing rather than a widener | `correlation`, `side_mid_db`, `mono_fold_db`; `level_stratified` shows whether it is level dependent |
| Shared peaks 2.69 / 5.38 / 8.07 Hz, no 4 Hz tremolo | Rhythm-locked modulation, not an LFO | `envelope_modulation` peak list per file |
| "Earlier interpretations alternated between clipping, EQ, compression, noise, stereo" | Nothing was settled | This is why section 4 exists |

Two specific traps the analysis is written to avoid:

1. **"The spectral ratio *is* the EQ curve."** It is only an EQ curve if the
   difference is the same at every loudness. `level_stratified_transfer` measures
   the band ratios in four loudness quartiles with the overall gain removed, and
   reports `level_dependence_db`. Validation: a static shelf reads 0.01 dB; a
   compressor reads 0.3 dB (correctly "not a level-dependent EQ"); a genuinely
   level-dependent band reads 25 dB.
2. **"Low HF correlation means added noise."** Noise raises spectral flatness
   while a shelving filter mostly does not, and noise is not level dependent in
   the same way as a saturation stage. Flatness, level stratification and the
   modulation spectrum together distinguish these; no single number does.

---

## 2. Measurement protocol used by the tooling

`tools/analyze_pair.py <original> <reference>`:

1. **Same recording check** — band-limited (150–4000 Hz) mono waveform
   correlation over the full plausible lag range, parabolic sub-sample
   refinement, onset-envelope cross-check, per-segment lag spread. Verdict is
   "same recording" only if duration, correlation, drift and tuning agree.
   Periodic material can make the lag ambiguous; that is reported instead of
   hidden.
2. **Tuning** — the ratio (in cents) that best aligns the two long-term
   log-spectra. A resampled file is detected (validated at +106 cents).
3. **Levels** — sample peak, true peak (4× oversampled), RMS, LUFS (BS.1770-4),
   LRA, crest, 50 ms envelope range, clipped fraction, DC.
4. **Spectra** — band energies, 1/3 octave, centroid, flatness, rolloff, plus the
   apparent transfer curve (1/3-octave smoothed) after alignment.
5. **Static vs level-dependent** — see above.
6. **Level relation** — slope of reference level vs original level over 200 ms
   windows, and the implied compression ratio.
7. **Modulation** — envelope-spectrum peaks, so rhythm-locked peaks are
   distinguished from an independent LFO.

Every comparison is made at **matched loudness**: the reference is attenuated (or
boosted, only for measurement) to the original's LUFS first. Gain differences
therefore cannot drive the conclusions.

---

## 3. What the reported numbers imply, and what they cannot

* **Crest 19.5 dB → 2.63 dB and a large fraction of samples at ±1.0** is not a
  tone change; it is roughly 17 dB of crest removal. No EQ produces that. It
  points to aggressive dynamics and/or hard clipping (the decoded peak of 1.1065
  is itself a sign of inter-sample overshoot from lossy coding of a clipped
  signal). This is exactly the behaviour the brief forbids reproducing.
* **Centroid 386 → 1958 Hz** is a large, broad brightening. It is consistent with
  a high-shelf/presence lift *and* with harmonic generation and *and* with a
  crushed signal whose low end was compressed away. Three families, one symptom —
  which is why `tools/compare_methods.py` compares all of them at matched
  loudness rather than choosing one.
* **L/R 0.968 → 0.863 with S/M rising ~4 dB** can come from a widener, from
  independent per-channel dynamics, or from band-limited nonlinear processing
  that decorrelates HF. The analysis measures the *mono fold-down change* as
  well, because a widened image that collapses in mono is a defect, not a
  character.
* **The 2.69 / 5.38 / 8.07 Hz peaks** are beat harmonics; the absence of an
  independent 4 Hz peak argues against a tremolo/LFO stage. The app has no
  modulation stage and the analysis keeps verifying that.

---

## 4. Decision rules (measurement → processing family)

These are the rules the tooling implements, so the profile follows evidence
instead of a guess:

| Observation | Conclusion | Stage used |
|---|---|---|
| Band ratios identical in all loudness quartiles | static tone change | stage 1 (EQ), curve fitted from `transfer.csv` |
| HF band ratio grows with loudness | level-dependent HF | stage 2 (dynamic EQ) and/or stage 4 high band |
| Envelope-range and level-slope reduction, crest lower, band ratios unchanged | broadband dynamics | stage 5 (compressor), ratio from `level_slope` |
| One band's energy grows while its peak level does not | band compression | stage 4 (multiband) |
| Flatness rises with level and HF correlation falls without a level-independent HF lift | harmonic generation | stage 6 (saturation, oversampled, attenuation-compensated) |
| S/M rises with the mono fold-down level unchanged | genuine width, not phase trickery | stage 7 (side gain) |
| Correlation falls and mono fold-down level drops | phase decorrelation | **rejected** — no phase rotation is offered |
| Nothing above applies to the top octave but it is brighter at all levels | plain shelf | stage 1 only |

Anything that requires reproducing crest ≈ 2.6 dB is **refused** by design (see
section 6).

---

## 5. The shipped graph and why each stage is there

`profiles/reference_character.json` — hypothesis, all stages conservative:

| Stage | Setting | Justification |
|---|---|---|
| EQ | 32 Hz high-pass (0.707), −2.8 dB lowshelf @110 Hz, −1.6 dB @380 Hz, +1.8 dB @1.6 kHz, +3.6 dB highshelf @7 kHz, +1.6 dB @12 kHz | "less sub-bass, more mid/high" from the reported band behaviour, kept ≤ 4 dB so nothing becomes hissy |
| Dynamic EQ | −4 dB @6.8 kHz, above −30 dBFS, 2.2:1, 3 ms/90 ms, max 6 dB | reported HF movement/flatness; a de-esser shape, not a static lift |
| Multiband | crossovers 180 Hz / 2.4 kHz, 1.4–1.7:1, mix 0.45–0.7 | band-dependent density without flattening the whole mix |
| Compressor | −20 dB, 1.7:1, 22 ms/210 ms, 10 dB knee, mix 0.55, makeup 0 | gentle shared density; the 2:1 band is deliberate |
| Saturation | tanh, +4 dB drive, 12% mix, 4× oversampling, RMS-compensated | HF texture that is not a pure EQ lift; **attenuation-only** compensation so it cannot become a loudness boost |
| Stereo | 110% width, side only | reported S/M change, with the mono sum untouched |
| Loudness match | on, LUFS, attenuation-capped | the render lands at the input's loudness; it cannot get louder |
| Limiter | −1 dBTP, 6 ms look-ahead, 140 ms release | the required ceiling, verified on the encoded file |

Nothing in the shipped graph is claimed to reproduce the reference. It is the
most defensible interpretation of the reported measurements that still satisfies
the quality rules; the analysis is what should replace its numbers.

---

## 6. Where the goals conflict (stated, not hidden)

The reported reference is extremely dense: crest ≈ 2.6 dB, envelope range
≈ 1.9 dB, most samples near full scale. Reproducing that *is* loudness
maximisation, and it cannot coexist with:

* natural dynamics and unflattened transients,
* no export clipping,
* a −1 dBTP ceiling on the encoded file,
* "do not automatically make the output louder than the input".

The app therefore prioritises the tonal/textural part of the character (tilt,
presence, level-dependent brightness, slight width) and deliberately declines the
density part. If the measured pair shows that the *tonal* part is also a
by-product of the crushing (e.g. the HF lift only appears on the loudest
passages), the analysis will say so through `level_dependence_db`, and the
profile should then keep only the conservative stages.

This limitation is stated in the app (`Analysis ▸ Guide / about`) and in the
export report; it is not quietly ignored.

---

## 7. Manual check to run once the files are available

```bash
python tools/analyze_pair.py --original TestSong.ogg --reference NewTest.ogg \
    --out-dir analysis/NewTest --figures
python tools/compare_methods.py --original TestSong.ogg --reference NewTest.ogg \
    --out analysis/method_comparison.md
python tools/render_example.py --input TestSong.ogg \
    --profile profiles/reference_character.json --out-dir analysis/TestSong \
    --name TestSong --encode ogg --ab
```

Read the three outputs in this order:

1. `analysis/NewTest/report.txt` §"SAME RECORDING?" — if the verdict is not
   "same recording", stop: the pair does not describe one processing chain.
2. `report.txt` §"IS THE DIFFERENCE STATIC OR LEVEL-DEPENDENT?" — this decides
   whether the profile needs dynamic stages at all. If the answer is "mostly
   static", the honest profile is a single EQ cascade fitted from
   `analysis/NewTest/transfer.csv`, and every dynamic/harmonic stage should be
   switched off.
3. `report.txt` §"SHORT-TERM LEVEL RELATION" — the implied ratio sets the
   compressor/band ratios; the level slope tells you whether the reference's
   density comes from broadband compression at all.
4. `analysis/TestSong/validation.txt` — the numbers for the render that would be
   shipped. Compare crest and envelope range with the input, not with the
   reference, and confirm the true peak and loudness rules still hold.

Only then adjust `profiles/reference_character.json`, keeping the same stage
order, and re-run the third command to confirm nothing regressed.

---

## 8. Remaining differences and open questions

1. **Reference not measured.** Everything about `NewTest.ogg` above is a claim
   from the brief.
2. **No listening comparison was possible** for the reference pair (files absent,
   and no monitoring device in the build environment). All statements about
   sound are derived from measurements; none are listening impressions.
3. **Crest/density gap.** The shipped render keeps crest within a dB or two of
   the input instead of collapsing it (measured on demo material: 12.8 → 13.5 dB).
   That is a deliberate difference, not a defect to be fixed later.
4. **Lossy overshoot.** Vorbis/MP3 output can exceed the render's true peak by a
   few tenths of a dB; the export report measures it and says so.
5. **The score in `compare_methods` is band-energy based.** It is a screening
   tool, not a verdict: it cannot hear and it can be fooled by a chain that
   matches band energies with the wrong mechanism. The per-candidate
   "explains / fails to explain" columns and the level-stratified figures must be
   read together with the score.
6. **Ambiguous alignment on loop-based material.** The lag estimator flags it
   (`ambiguous`, `onset_peak_ratio`) instead of pretending certainty.
