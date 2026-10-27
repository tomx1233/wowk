"""DSP backend for Timbreforge.

Modules
-------
audio      file decoding/encoding, format mapping, same-speed sample-rate
           conversion, atomic export
meter      peak / true-peak / RMS / LUFS / crest / stereo / spectral metrics
biquad     RBJ biquad designs, cascade processing, linear-phase FIR EQ
dynamics   compressor, multiband compressor, dynamic EQ, transient shaper,
           true-peak lookahead limiter
nonlinear  oversampled waveshaping with automatic level compensation
stereo     mid/side width with mono-compatibility guard
graph      the one documented processing graph + JSON profiles
analyze    reference-pair analysis and candidate comparison tooling
"""

from __future__ import annotations

from .audio import AudioData, load_audio, save_audio, probe_audio, AudioError
from .graph import Profile, render, render_profile, DEFAULT_PROFILE
from .meter import Stats, measure

__all__ = [
    "AudioData",
    "load_audio",
    "save_audio",
    "probe_audio",
    "AudioError",
    "Profile",
    "render",
    "render_profile",
    "DEFAULT_PROFILE",
    "Stats",
    "measure",
]
