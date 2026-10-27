> **This file is a tool smoke test, not evidence about NewTest.ogg.**
> It was produced by `tools/compare_methods.py` on synthetic demo material
> (`tools/make_demo.py`) whose "reference" was rendered by this same app with the
> hypothesis profile. Its only purpose is to show that the comparison tool runs
> end to end. Re-run it with the real supplied pair before drawing any conclusion
> about the reference processing.

# Candidate methods, measured against the reference

Compared on 24.0 s starting at 0.0 s, lag-corrected by -10 samples.

All candidates keep pitch and speed at 1.00x. Scores are gain invariant: each render is attenuated to the original's loudness before comparison, so a louder chain cannot win.

| candidate | family | band RMSE dB | centroid Hz | crest dB | env range dB | L/R corr | S/M dB | export LUFS | true peak dBTP |
|---|---|---|---|---|---|---|---|---|---|
| A_identity | A. static tone | 3.29 | 150 | 12.7 | 6.7 | 0.910 | -13.2 | -14.2 | -2.85 |
| B_tilt | A. static tone | 0.72 | 179 | 14.1 | 6.2 | 0.890 | -12.3 | -14.2 | -1.72 |
| C_tilt_comp | C. dynamics | 0.98 | 182 | 13.4 | 3.9 | 0.888 | -12.2 | -14.2 | -2.47 |
| D_multiband | B. dynamic tone | 1.76 | 180 | 13.9 | 4.9 | 0.865 | -11.4 | -14.2 | -2.00 |
| E_dyneq | B. dynamic tone | 1.94 | 166 | 12.9 | 6.4 | 0.894 | -12.5 | -14.2 | -2.75 |
| F_saturation | D. nonlinear | 1.76 | 168 | 11.7 | 5.5 | 0.894 | -12.5 | -14.2 | -4.08 |
| G_comp_sat | D. nonlinear | 3.21 | 155 | 11.2 | 3.7 | 0.919 | -13.7 | -14.2 | -4.50 |
| H_stereo | E. stereo | 3.29 | 150 | 12.9 | 6.7 | 0.883 | -12.0 | -14.2 | -2.67 |
| I_combined | F. combined | 0.75 | 188 | 13.7 | 4.7 | 0.834 | -10.4 | -14.2 | -2.25 |
| J_maximised_rejected | control | 1.84 | 164 | 17.7 | 2.9 | 0.924 | -14.0 | -27.4 | -11.09 |

Reference itself: centroid 187 Hz, crest 13.8 dB, envelope range 4.8 dB, corr 0.845, S/M -10.7 dB, -14.2 LUFS.

## What each candidate explains and misses

### A_identity - Reference is only an EQ change plus gain
- explains: any static spectral difference
- fails to explain: density, crest and stereo changes
- measured: band RMSE 3.29 dB, crest error 1.1 dB, envelope error 1.9 dB, correlation error 0.065

### B_tilt - Static tilt: less sub-bass, more air
- explains: band-energy deltas and centroid
- fails to explain: anything level dependent
- measured: band RMSE 0.72 dB, crest error 0.3 dB, envelope error 1.4 dB, correlation error 0.045

### C_tilt_comp - Tilt plus a real compressor
- explains: envelope range, crest, density
- fails to explain: HF-specific behaviour
- measured: band RMSE 0.98 dB, crest error 0.4 dB, envelope error 0.9 dB, correlation error 0.043

### D_multiband - Band-wise dynamics make the top end denser without flattening everything
- explains: level-dependent HF behaviour, flatter envelope
- fails to explain: a purely static spectral tilt
- measured: band RMSE 1.76 dB, crest error 0.1 dB, envelope error 0.1 dB, correlation error 0.020

### E_dyneq - High band is controlled only when loud (de-esser style)
- explains: level-dependent HF, sibilance control
- fails to explain: broadband tightness
- measured: band RMSE 1.94 dB, crest error 0.9 dB, envelope error 1.6 dB, correlation error 0.048

### F_saturation - HF lift comes from generated harmonics, not EQ
- explains: harmonic content, spectral flatness
- fails to explain: spectral centroid change without added harmonics
- note: level compensated by attenuation only
- measured: band RMSE 1.76 dB, crest error 2.1 dB, envelope error 0.6 dB, correlation error 0.049

### G_comp_sat - Density first, then harmonic generation
- explains: crest collapse plus harmonic brightness
- fails to explain: the harshness and intermodulation of the reference
- note: worst case for harshness; watch pumping
- measured: band RMSE 3.21 dB, crest error 2.6 dB, envelope error 1.1 dB, correlation error 0.074

### H_stereo - Only width changed
- explains: side/mid ratio
- fails to explain: correlation drop without a dedicated widener
- measured: band RMSE 3.29 dB, crest error 0.9 dB, envelope error 1.9 dB, correlation error 0.038

### I_combined - Tilt + band dynamics + gentle saturation + width
- explains: several deltas at once without loudness maximising
- fails to explain: the reference's extreme density (deliberately not targeted)
- note: this is what the app ships as a starting point
- measured: band RMSE 0.75 dB, crest error 0.1 dB, envelope error 0.1 dB, correlation error 0.011

### J_maximised_rejected - Reproduce the reference's loudness directly
- explains: crest and density figures
- fails to explain: the quality rules; kept only as a demonstration of what is refused
- note: REFERENCE ONLY - violates the quality rules, never shipped
- measured: band RMSE 1.84 dB, crest error 3.9 dB, envelope error 1.9 dB, correlation error 0.079

## Ranking (best band-shape match first)

1. B_tilt - band RMSE 0.72 dB
2. I_combined - band RMSE 0.75 dB
3. C_tilt_comp - band RMSE 0.98 dB
4. F_saturation - band RMSE 1.76 dB
5. D_multiband - band RMSE 1.76 dB
6. J_maximised_rejected - band RMSE 1.84 dB
7. E_dyneq - band RMSE 1.94 dB
8. G_comp_sat - band RMSE 3.21 dB
9. A_identity - band RMSE 3.29 dB
10. H_stereo - band RMSE 3.29 dB

The ranking is about spectral shape only. Crest and envelope columns show why the reference cannot be matched on dynamics: its crest is around 2.6 dB, which requires the loudness maximising that this project refuses. A candidate is therefore chosen for tonal/textural closeness with natural dynamics, not for the lowest RMSE at any cost.
