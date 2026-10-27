# Processing families: hypotheses, tests, verdicts

The brief asks for a broad first pass over the practical families, not a single
guess. This file records each family, the concrete test, what it would explain,
what it would not, and the current verdict. Parameters and numbers are
reproducible with `tools/compare_methods.py` (fixed candidate list) and
`tools/analyze_pair.py`.

**Status note.** The verdicts below are *design decisions* validated on synthetic
material; they are not measurements of `NewTest.ogg`, which was not supplied.
Columns "measured" show what the tooling produces when a pair is available. The
smoke-test table in `examples/method_comparison_demo.md` shows the output format.

---

## A. Static tonal processing

| | |
|---|---|
| **Hypothesis** | The reference is a tonal reshape plus gain: less sub-bass, more presence/air. |
| **Implementation** | One RBJ biquad cascade: high/low shelves, broad bells, optional HP/LP; minimum phase default, linear-phase FIR option (4097 taps, group delay removed). |
| **Explains** | Band-energy deltas, centroid, any *constant* spectral ratio. |
| **Fails to explain** | Crest collapse, envelope flattening, level-dependent brightness, stereo change. |
| **Test** | `level_dependence_db` per band: a true static filter is identical at every loudness. Rendered EQ vs designed curve: within 0.06 dB (tested). |
| **Verdict** | **Retained** as stage 1 — cheap, transparent, and the only family that can be fitted from `transfer.csv` without guessing a mechanism. Minimum phase by default; linear phase available when pre-ringing must be avoided (it costs latency, which is compensated so timing is unchanged). |

Notches/narrow resonances: supported (type `notch`) but not used — a narrow notch
is almost never the explanation for a broad centroid change, and a wrong notch is
audible.

---

## B. Dynamic tonal processing

| | |
|---|---|
| **Hypothesis** | Part of the brightness only appears on loud passages (de-esser / dynamic EQ / band compression). |
| **Implementation** | Dynamic EQ: fixed peaking filter, blend amount driven by a band-passed detector with attack/release and a hard ceiling. Multiband: per-band compressor around a complementary linear-phase crossover. |
| **Explains** | HF energy that grows faster than level; spectral flatness changes that track loudness; sibilance control. |
| **Fails to explain** | A ratio that is constant at all levels (that is a static filter), or an envelope collapse across all bands (that is broadband compression). |
| **Test** | Quartile transfer shapes with gain removed. Validation: level-dependent band = 25 dB spread; static shelf = 0.01 dB; broadband compressor = 0.3 dB (correctly classified as *not* a level-dependent EQ). |
| **Verdict** | **Retained** as stages 2 and 4, but only where the measurement shows level dependence. Defaults are mild (≤ 6 dB reduction, ratios ≤ 2.2:1). |

---

## C. Dynamics and transients

| | |
|---|---|
| **Hypothesis** | Some of the character is compression; the reference pushes it to the extreme. |
| **Implementation** | Feed-forward compressor (RMS or peak detector, soft knee, parallel mix, optional sidechain HP, linked or independent channels), transient shaper, offline look-ahead true-peak limiter. All makeup gains are manual and default to 0 dB. |
| **Explains** | Envelope-range reduction, level slope < 1, crest reduction, higher average density. |
| **Fails to explain** | A crest of 2.6 dB with natural attacks; any HF-only change; a stereo change. |
| **Test** | `level_relation.level_slope` → implied ratio; envelope range and crest before/after; block-to-block gain change (pumping) < 6 dB in tests; loudness neutrality when makeup is 0. |
| **Verdict** | **Retained, deliberately gentle.** Typical settings 1.4–1.7:1 with 40–55% parallel mix. A "reproduce the density" chain (8:1 + limit at −0.1 dBTP) is implemented in the comparison list as `J_maximised_rejected` purely to demonstrate what is refused — it is never shipped in a profile. |

Peak vs RMS detection: both are available; RMS with a 12 ms window is the default
because peak detection on already-bright material exaggerates HF ducking.

Channel-linked vs independent: independent processing is available but not
recommended — it decorrelates the image, and the stereo change is better explained
(and controlled) by the width stage.

---

## D. Harmonic and nonlinear processing

| | |
|---|---|
| **Hypothesis** | Some of the HF lift and the "flatness" is generated harmonics rather than EQ. |
| **Implementation** | tanh / arctan / cubic / asymmetric shapers, run 4× oversampled with polyphase FIR resampling in blocks, DC blocked, blended with a parallel mix, and **level-compensated by attenuation measured after shaping**. |
| **Explains** | Rising spectral flatness, intermodulation, HF that grows with level, a centroid shift without a matching static EQ curve. |
| **Fails to explain** | A clean static tilt; anything that is present at low levels too. |
| **Test** | RMS gain of the saturation stage ≤ 0.05 dB (tested); compensation reported as ≤ 0 dB; flatness delta and `level_dependence_db` before/after. |
| **Verdict** | **Retained at low mix (12%)** as a texture stage, because it can explain flatness where a shelf cannot. Anything above ~20% mix on dense material starts to sound gritty and to intermodulate badly — that is a listening judgement, and the controls keep it opt-in. |

Oversampled vs aliased: only oversampled is offered. Aliased shaping folds HF
energy back as inharmonic whistles, which the brief explicitly forbids.

---

## E. Stereo and phase

| | |
|---|---|
| **Hypothesis** | The image widened (S/M +4 dB, correlation −0.1). |
| **Implementation** | Side-signal gain only (`S' = k·S`), optional mono-below crossover. No phase rotation, no per-channel EQ. |
| **Explains** | S/M change and correlation drop with the mono sum untouched. |
| **Fails to explain** | Correlation changes accompanied by a mono fold-down level drop (that is phase decorrelation). |
| **Test** | Mid signal is bit-identical after width changes; mono fold-down level change reported; width 0% collapses to a correlated mono image (tested). |
| **Verdict** | **Retained, capped at 110–120%.** Widening beyond that on already-wide material produces a hollow, phasey image and damages mono compatibility — explicitly rejected. |

Independent L/R processing as the explanation for the width change: **rejected**
as a first choice; it is measurable if it happens (per-channel level statistics
diverge) and the analysis reports per-channel crest/RMS, but nothing in the
reported figures requires it.

---

## F. Other evidence-supported possibilities

| Candidate | What it would explain | Verdict |
|---|---|---|
| Short ambience / early reflections | HF decorrelation, a sense of space | **Not implemented.** It cannot be verified from the reported numbers, adds a tail that changes timing perception, and the brief says not to add ambience without support. |
| Frequency-domain spectral-envelope shaping | A very smooth, arbitrary tilt | Covered by the EQ cascade; a brickwall FFT curve would add pre-ringing for no benefit at these magnitudes. |
| Harmonic/percussive-dependent processing | Different treatment of attacks vs sustains | Partially covered by the transient shaper; a full HPSS split is disproportionate and unverifiable without the files. |
| Codec-generation effects | Spectral flatness, HF differences, small duration offsets | **Accounted for, not imitated.** The analysis reports container/subtype and codec padding separately from musical timing; the app exports clean files instead of degrading them on purpose. |
| Same-speed sample-rate conversion | HF/phase differences if the reference was resampled | Supported on export (band-limited, duration preserving). If the measured tuning or drift shows a rate mismatch, that is a finding, not something to imitate. |
| Modulation / tremolo (e.g. 4 Hz) | Periodic level movement | **Rejected** for now: the reported 2.69/5.38/8.07 Hz peaks line up with the beat, and no independent LFO peak was established. The analysis keeps checking this. |
| Noise / dither injection | Higher flatness, HF "air" | **Rejected.** It is the classic wrong explanation for HF decorrelation, it is level independent, and it would be audible as hiss. |

---

## Effect order

The graph order (tone → dynamic tone → transients → band dynamics → broadband
dynamics → harmonics → stereo → trim → match → limit) was chosen so that:

* dynamics react to the intended spectrum (EQ first);
* transient shaping is not immediately undone by compression;
* saturation is not gain-modulated by a compressor after it;
* the limiter is last and sees the final gain;
* nothing is applied twice: there is exactly one static EQ cascade, one
  compression chain and one saturation stage in the graph.

Alternative orders were considered and rejected:

* **Compression before EQ** — makes the compression react to the wrong balance
  and doubles the effective tone change at loud passages.
* **Stereo before dynamics** — a linked compressor then reacts to a widened side
  signal, reducing width unpredictably.
* **Limiter before loudness match** — the match would immediately undo the
  ceiling.
