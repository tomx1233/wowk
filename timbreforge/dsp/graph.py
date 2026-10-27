"""The single processing graph used by both preview and export.

Documented order (and the reason for it)
---------------------------------------
1.  Static EQ            - one minimum-phase (or linear-phase) biquad cascade.
                           Applied first so later dynamics react to the intended
                           spectral balance.
2.  Dynamic EQ           - level-dependent peaking bands; tonal changes that
                           only happen on loud material.
3.  Transient shaping    - before compression, so the dynamics stage does not
                           immediately undo attack/sustain changes.
4.  Multiband dynamics   - band-limited compression around a complementary
                           linear-phase crossover (bands sum to the input, zero
                           net delay).
5.  Broadband compressor - one full-range feed-forward stage.
6.  Saturation           - oversampled, level-compensated (attenuation only),
                           after the dynamics so it is not gain-modulated.
7.  Stereo width         - side-only gain; the mono sum is bit-identical.
8.  Output trim          - user trim, default 0 dB.
9.  Loudness matching    - brings the render to the *input's* loudness (never
                           above it) so A/B is level-fair.
10. True-peak limiter    - last stage; ceiling -1 dBTP by default, offline
                           look-ahead with zero output latency.

There is no pitch, tempo, speed, time-stretch or rate control anywhere in the
graph, by design.  The renderer asserts that output length == input length.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .. import SCHEMA_VERSION
from . import dynamics as dyn
from . import nonlinear as nl
from . import stereo as st
from .audio import AudioData
from .biquad import (Band, apply_linear_phase, linear_phase_fir, magnitude_db,
                     process_cascade)
from .meter import (integrated_loudness, measure, rms_db, true_peak, db, Stats)

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

STAGE_ORDER = (
    "eq",
    "dynamic_eq",
    "transient",
    "multiband",
    "compressor",
    "saturation",
    "stereo",
    "trim",
    "limiter",
)

#: relative cost used for progress reporting
_STAGE_WEIGHTS = {
    "eq": 0.08, "dynamic_eq": 0.10, "transient": 0.05, "multiband": 0.14,
    "compressor": 0.10, "saturation": 0.14, "stereo": 0.03, "trim": 0.01,
    "gain_match": 0.05, "limiter": 0.30,
}


@dataclass
class Profile:
    """A complete, serialisable processing description."""

    name: str = "Neutral \u00b7 level matched"
    description: str = (
        "No tonal, dynamic, harmonic or stereo processing. The render is only "
        "level-matched to the input and true-peak limited at -1 dBTP.")
    eq_bands: list[Band] = field(default_factory=list)
    eq_mode: str = "minimum"              # "minimum" | "linear"
    eq_fir_taps: int = 4097
    dynamic_eq: dyn.DynamicEqParams = field(default_factory=dyn.DynamicEqParams)
    transient: dyn.TransientParams = field(default_factory=dyn.TransientParams)
    multiband: dyn.MultibandParams = field(default_factory=dyn.MultibandParams)
    compressor: dyn.CompressorParams = field(default_factory=dyn.CompressorParams)
    saturation: nl.SaturationParams = field(default_factory=nl.SaturationParams)
    stereo: st.StereoParams = field(default_factory=st.StereoParams)
    output_trim_db: float = 0.0
    loudness_match: bool = True
    loudness_match_mode: str = "lufs"     # "lufs" | "rms"
    limiter: dyn.LimiterParams = field(default_factory=dyn.LimiterParams)
    bypass: bool = False
    schema: int = SCHEMA_VERSION

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "bypass": self.bypass,
            "eq": {"mode": self.eq_mode, "fir_taps": self.eq_fir_taps,
                   "bands": [b.to_dict() for b in self.eq_bands]},
            "dynamic_eq": self.dynamic_eq.to_dict(),
            "transient": self.transient.to_dict(),
            "multiband": self.multiband.to_dict(),
            "compressor": self.compressor.to_dict(),
            "saturation": self.saturation.to_dict(),
            "stereo": self.stereo.to_dict(),
            "output_trim_db": self.output_trim_db,
            "loudness_match": self.loudness_match,
            "loudness_match_mode": self.loudness_match_mode,
            "limiter": self.limiter.to_dict(),
            "stage_order": list(STAGE_ORDER),
        }

    @staticmethod
    def from_dict(d: dict) -> "Profile":
        if not isinstance(d, dict):
            raise ValueError("profile must be a JSON object")
        p = Profile()
        p.name = str(d.get("name", p.name))
        p.description = str(d.get("description", p.description))
        p.bypass = bool(d.get("bypass", False))
        eq = d.get("eq", {}) or {}
        p.eq_mode = str(eq.get("mode", "minimum"))
        p.eq_fir_taps = int(eq.get("fir_taps", 4097))
        p.eq_bands = [Band.from_dict(b) for b in eq.get("bands", [])]
        p.dynamic_eq = dyn.DynamicEqParams.from_dict(d.get("dynamic_eq", {}) or {})
        p.transient = dyn.TransientParams.from_dict(d.get("transient", {}) or {})
        p.multiband = dyn.MultibandParams.from_dict(d.get("multiband", {}) or {})
        p.compressor = dyn.CompressorParams.from_dict(d.get("compressor", {}) or {})
        p.saturation = nl.SaturationParams.from_dict(d.get("saturation", {}) or {})
        p.stereo = st.StereoParams.from_dict(d.get("stereo", {}) or {})
        p.output_trim_db = float(d.get("output_trim_db", 0.0))
        p.loudness_match = bool(d.get("loudness_match", True))
        p.loudness_match_mode = str(d.get("loudness_match_mode", "lufs"))
        p.limiter = dyn.LimiterParams.from_dict(d.get("limiter", {}) or {})
        return p

    # -- validation -------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        errs.extend(f"EQ band {i + 1}: {e}" for i, e in enumerate(
            _validate_bands(self.eq_bands)))
        errs.extend(self.dynamic_eq.validate())
        errs.extend(self.transient.validate())
        errs.extend(self.multiband.validate())
        errs.extend(self.compressor.validate())
        errs.extend(self.saturation.validate())
        errs.extend(self.stereo.validate())
        errs.extend(self.limiter.validate())
        if not (-24.0 <= self.output_trim_db <= 24.0):
            errs.append("output trim must be within +/-24 dB")
        if self.eq_mode not in ("minimum", "linear"):
            errs.append("EQ mode must be 'minimum' or 'linear'")
        if self.loudness_match_mode not in ("lufs", "rms"):
            errs.append("loudness match mode must be 'lufs' or 'rms'")
        return errs

    def active_stages(self) -> list[str]:
        if self.bypass:
            return []
        out = []
        if any(b.is_active() for b in self.eq_bands):
            out.append("eq")
        if self.dynamic_eq.enabled and any(b.enabled for b in self.dynamic_eq.bands):
            out.append("dynamic_eq")
        if self.transient.enabled and (abs(self.transient.attack_db) > 0.05
                                       or abs(self.transient.sustain_db) > 0.05):
            out.append("transient")
        if self.multiband.enabled and any(b.enabled for b in self.multiband.bands):
            out.append("multiband")
        if self.compressor.enabled:
            out.append("compressor")
        if self.saturation.enabled and self.saturation.mix > 0:
            out.append("saturation")
        if self.stereo.enabled and abs(self.stereo.width - 1.0) > 1e-4:
            out.append("stereo")
        if abs(self.output_trim_db) > 1e-6:
            out.append("trim")
        if self.loudness_match:
            out.append("gain_match")
        if self.limiter.enabled:
            out.append("limiter")
        return out

    def curve(self, freqs: np.ndarray, sr: int) -> np.ndarray:
        """Combined static EQ magnitude (dB).  Dynamic stages are not shown."""
        return magnitude_db(self.eq_bands, freqs, sr)

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "Profile":
        with open(path, "r", encoding="utf-8") as fh:
            return Profile.from_dict(json.load(fh))


def _validate_bands(bands: Iterable[Band]) -> list[str]:
    errs = []
    for b in bands:
        if b.type not in ("peaking", "lowshelf", "highshelf", "highpass", "lowpass",
                          "notch", "bandpass", "allpass"):
            errs.append(f"unknown type '{b.type}'")
        if not (10 <= b.freq <= 22000):
            errs.append(f"frequency {b.freq:.0f} Hz is outside 10 Hz - 22 kHz")
        if not (0.1 <= b.q <= 18):
            errs.append("Q must be between 0.1 and 18")
        if abs(b.gain_db) > 24:
            errs.append("band gain is limited to +/-24 dB")
    return errs


DEFAULT_PROFILE = Profile()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


@dataclass
class RenderResult:
    samples: np.ndarray
    sample_rate: int
    reports: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False
    output_trim_db: float = 0.0
    loudness_match_db: float = 0.0
    latency_samples: int = 0

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    def to_audio(self, source_name: str = "") -> AudioData:
        return AudioData(samples=self.samples, sample_rate=self.sample_rate,
                         title=source_name)


class Cancelled(RuntimeError):
    pass


def render(x: np.ndarray, sr: int, profile: Profile, *,
           progress: ProgressFn | None = None,
           should_cancel: CancelFn | None = None,
           target_loudness: float | None = None) -> RenderResult:
    """Render one buffer through the graph.

    ``target_loudness`` overrides the automatic "match the input" target; the app
    uses the input's own loudness, so the render is never louder than the input.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    problems = profile.validate()
    if problems:
        raise ValueError("Invalid profile: " + "; ".join(problems))

    n = x.shape[0]
    result = RenderResult(samples=x.copy(), sample_rate=int(sr))

    def tick(frac: float, label: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), label)

    def check() -> None:
        if should_cancel and should_cancel():
            raise Cancelled("Render cancelled.")

    tick(0.0, "Analysing input")
    input_lufs = integrated_loudness(x, sr) if profile.loudness_match else -70.0
    input_rms = rms_db(x)

    if profile.bypass:
        tick(1.0, "Bypassed")
        result.samples = x.copy()
        return result

    y = x.copy()
    acc = 0.0
    total = sum(_STAGE_WEIGHTS.values())

    def step(name: str) -> float:
        return _STAGE_WEIGHTS[name] / total

    # 1 -- static EQ ------------------------------------------------------
    active_eq = [b for b in profile.eq_bands if b.is_active()]
    if active_eq:
        tick(acc / total, "Tonal shaping")
        if profile.eq_mode == "linear":
            taps = max(1023, int(profile.eq_fir_taps) | 1)
            h = linear_phase_fir(active_eq, sr, taps)
            y, delay = apply_linear_phase(y, h)
            result.reports.append({"stage": "eq", "mode": "linear", "taps": taps,
                                   "group_delay_samples": delay})
            result.latency_samples += 0  # delay removed, timing preserved
        else:
            sections = [b.coefficients(sr) for b in active_eq]
            y = process_cascade(y, sections)
            result.reports.append({"stage": "eq", "mode": "minimum",
                                   "sections": len(sections)})
    acc += step("eq")
    check()

    # 2 -- dynamic EQ -----------------------------------------------------
    if profile.dynamic_eq.enabled:
        tick(acc / total, "Dynamic tone")
        y, rep = dyn.dynamic_eq(y, sr, profile.dynamic_eq)
        result.reports.append({"stage": "dynamic_eq", **rep})
    acc += step("dynamic_eq")
    check()

    # 3 -- transient shaping ---------------------------------------------
    if profile.transient.enabled:
        tick(acc / total, "Transient shaping")
        y, rep = dyn.transient_shape(y, sr, profile.transient)
        result.reports.append({"stage": "transient", **rep})
    acc += step("transient")
    check()

    # 4 -- multiband ------------------------------------------------------
    if profile.multiband.enabled:
        tick(acc / total, "Multiband dynamics")
        y, rep = dyn.multiband(y, sr, profile.multiband)
        result.reports.append({"stage": "multiband", **rep})
    acc += step("multiband")
    check()

    # 5 -- broadband compression -----------------------------------------
    if profile.compressor.enabled:
        tick(acc / total, "Compression")
        y, rep = dyn.compress(y, sr, profile.compressor)
        result.reports.append({"stage": "compressor", **rep})
    acc += step("compressor")
    check()

    # 6 -- saturation -----------------------------------------------------
    if profile.saturation.enabled and profile.saturation.mix > 0:
        tick(acc / total, "Harmonic shaping")
        y, rep = nl.saturate(y, sr, profile.saturation)
        result.reports.append({"stage": "saturation", **rep})
    acc += step("saturation")
    check()

    # 7 -- stereo ---------------------------------------------------------
    if profile.stereo.enabled:
        tick(acc / total, "Stereo")
        y, rep = st.set_width(y, profile.stereo, sr)
        result.reports.append({"stage": "stereo", **rep})
    acc += step("stereo")
    check()

    # 8 -- output trim ----------------------------------------------------
    if abs(profile.output_trim_db) > 1e-9:
        y = dyn.apply_gain_db(y, profile.output_trim_db)
        result.reports.append({"stage": "trim", "gain_db": profile.output_trim_db})
    acc += step("trim")

    # 9 -- loudness matching (target = the input's own loudness) ----------
    if profile.loudness_match:
        tick(acc / total, "Level matching")
        if profile.loudness_match_mode == "rms":
            current = rms_db(y)
            target = target_loudness if target_loudness is not None else input_rms
        else:
            current = integrated_loudness(y, sr)
            target = target_loudness if target_loudness is not None else input_lufs
        gain_db = float(np.clip(target - current, -18.0, 18.0))
        if abs(gain_db) > 0.01:
            y = dyn.apply_gain_db(y, gain_db)
        result.loudness_match_db = gain_db
        result.reports.append({"stage": "gain_match", "gain_db": gain_db,
                               "mode": profile.loudness_match_mode,
                               "target": target})
    acc += step("gain_match")
    check()

    # 10 -- true-peak limiter --------------------------------------------
    if profile.limiter.enabled:
        tick(acc / total, "True-peak limiting")
        y, rep = dyn.limit(y, sr, profile.limiter)
        result.reports.append({"stage": "limiter", **rep})
        if not rep.get("bypassed"):
            result.warnings.append(
                f"Limiter reduced peaks by up to {abs(rep.get('max_reduction_db', 0)):.1f} dB "
                f"to stay under {profile.limiter.ceiling_db:.1f} dBTP.")
    tick(1.0, "Done")

    if y.shape[0] != n:
        raise RuntimeError(
            f"internal error: render changed length ({y.shape[0]} != {n})")
    result.samples = np.ascontiguousarray(y, dtype=np.float64)
    return result


def render_profile(audio: AudioData, profile: Profile, **kw) -> RenderResult:
    return render(audio.samples, audio.sample_rate, profile, **kw)


# ---------------------------------------------------------------------------
# convenience: a few built-in starting points (all deliberately conservative)
# ---------------------------------------------------------------------------


def preset_bypass() -> Profile:
    return Profile()


def preset_gentle_character() -> Profile:
    """A safe, non-harsh starting point: broad tilt, one gentle compressor."""
    p = Profile(name="Gentle character",
                description="Broad low cut + air lift, one soft compressor, "
                            "-1 dBTP ceiling. Not derived from any reference file.")
    p.eq_bands = [
        Band("highpass", 28.0, 0.0, 0.707),
        Band("lowshelf", 120.0, -1.5, 0.707),
        Band("highshelf", 7000.0, 2.0, 0.707),
    ]
    p.compressor = dyn.CompressorParams(enabled=True, threshold_db=-22.0, ratio=1.8,
                                        attack_ms=25.0, release_ms=220.0, knee_db=10.0,
                                        detector="rms", mix=0.6, makeup_db=0.0)
    return p


def preset_reference_hypothesis() -> Profile:
    """A conservative interpretation of the *reported* reference behaviour.

    Built only from the numbers listed in the earlier investigation (less
    sub-bass, more mid/high energy, stronger high-frequency movement, slightly
    wider image), deliberately leaving out the extreme density of the reference.
    It has NOT been validated against the audio files: the pair analysis is what
    validates or replaces it.
    """
    p = Profile(
        name="Reference character (hypothesis)",
        description=(
            "Broad low-end tilt, presence/air lift, level-dependent de-essing, gentle "
            "band dynamics and a 12% harmonic blend. Conservative by design: no "
            "loudness maximising, -1 dBTP. Built from reported measurements only - "
            "validate against the supplied files before trusting it."))
    p.eq_bands = [
        Band("highpass", 32.0, 0.0, 0.707),
        Band("lowshelf", 110.0, -2.8, 0.707),
        Band("peaking", 380.0, -1.6, 0.85),
        Band("peaking", 1600.0, 1.8, 0.8),
        Band("highshelf", 7000.0, 3.6, 0.707),
        Band("peaking", 12000.0, 1.6, 0.7),
    ]
    p.dynamic_eq = dyn.DynamicEqParams(enabled=True, bands=[
        dyn.DynamicEqBand(enabled=True, freq=6800.0, q=1.4, gain_db=-4.0,
                          threshold_db=-30.0, ratio=2.2, attack_ms=3.0,
                          release_ms=90.0, direction="above", max_reduction_db=6.0),
    ])
    p.multiband = dyn.MultibandParams(enabled=True, crossovers_hz=[180.0, 2400.0], bands=[
        dyn.CompressorParams(enabled=True, threshold_db=-22.0, ratio=1.6,
                             attack_ms=25.0, release_ms=200.0, mix=0.7),
        dyn.CompressorParams(enabled=True, threshold_db=-21.0, ratio=1.7,
                             attack_ms=18.0, release_ms=170.0, mix=0.6),
        dyn.CompressorParams(enabled=True, threshold_db=-26.0, ratio=1.4,
                             attack_ms=10.0, release_ms=140.0, mix=0.45),
    ])
    p.compressor = dyn.CompressorParams(enabled=True, threshold_db=-20.0, ratio=1.7,
                                        attack_ms=22.0, release_ms=210.0, knee_db=10.0,
                                        detector="rms", mix=0.55, makeup_db=0.0)
    p.saturation = nl.SaturationParams(enabled=True, shaper="tanh", drive_db=4.0,
                                       mix=0.12, oversample=4, compensate="rms")
    p.stereo = st.StereoParams(enabled=True, width=1.10)
    p.limiter = dyn.LimiterParams(enabled=True, ceiling_db=-1.0, lookahead_ms=6.0)
    return p


@dataclass
class SimpleSettings:
    """The five controls of the simple interface.

    These are *not* an alternative DSP path: they deterministically generate a
    :class:`Profile`, which is then rendered by the same engine as everything
    else.  Switching to the advanced view loads the generated profile, so what
    you hear is always exactly what the visible parameters say.
    """

    character: float = 55.0      # 0 = neutral, 100 = full effect
    tone: float = 0.0            # -100 darker .. +100 brighter
    tightness: float = 35.0      # 0 = untouched dynamics .. 100 = controlled
    air: float = 25.0            # 0 = none .. 100 = extra top-end lift
    output_trim_db: float = 0.0
    ceiling_db: float = -1.0

    def clamped(self) -> "SimpleSettings":
        return SimpleSettings(
            character=float(np.clip(self.character, 0.0, 100.0)),
            tone=float(np.clip(self.tone, -100.0, 100.0)),
            tightness=float(np.clip(self.tightness, 0.0, 100.0)),
            air=float(np.clip(self.air, 0.0, 100.0)),
            output_trim_db=float(np.clip(self.output_trim_db, -24.0, 24.0)),
            ceiling_db=float(np.clip(self.ceiling_db, -6.0, -0.1)),
        )


def profile_from_simple(settings: SimpleSettings) -> Profile:
    """Turn the five simple controls into a complete profile.

    Everything scales from zero, so character 0 / air 0 / tightness 0 is a
    genuinely neutral render (only the loudness match and the limiter act).
    Magnitudes are capped so that no single control can make the result harsh.
    """
    s = settings.clamped()
    c = s.character / 100.0
    tone = s.tone / 100.0
    tight = s.tightness / 100.0
    air = s.air / 100.0

    p = Profile()
    p.name = f"Character {s.character:.0f}%"
    p.description = (
        f"Simple controls: character {s.character:.0f}%, tone {s.tone:+.0f}, "
        f"tightness {s.tightness:.0f}, air {s.air:.0f}. Generated profile - open the "
        "advanced view to see and edit every value.")

    # -- tone ------------------------------------------------------------
    # character lifts the upper mids and above; tone tilts the whole balance
    low = -2.4 * c + 3.0 * tone
    high = 2.4 * c + 3.0 * tone + 3.6 * air
    low = float(np.clip(low, -6.0, 6.0))
    high = float(np.clip(high, -6.0, 9.0))
    lowmid = float(np.clip(-1.2 * c + 1.0 * tone, -4.0, 4.0))
    presence = float(np.clip(1.5 * c + 1.5 * tone, -4.0, 5.0))
    p.eq_bands = [Band("highpass", 30.0, 0.0, 0.707)]
    if abs(low) > 0.05:
        p.eq_bands.append(Band("lowshelf", 120.0, round(low, 2), 0.707))
    if abs(lowmid) > 0.05:
        p.eq_bands.append(Band("peaking", 380.0, round(lowmid, 2), 0.85))
    if abs(presence) > 0.05:
        p.eq_bands.append(Band("peaking", 1600.0, round(presence, 2), 0.8))
    if abs(high) > 0.05:
        p.eq_bands.append(Band("highshelf", 7000.0, round(high, 2), 0.707))

    # -- level-dependent HF control --------------------------------------
    if c > 0.15:
        p.dynamic_eq = dyn.DynamicEqParams(enabled=True, bands=[
            dyn.DynamicEqBand(
                enabled=True, freq=6800.0, q=1.4,
                gain_db=round(-2.5 * c, 2),
                threshold_db=-30.0, ratio=2.0,
                attack_ms=5.0, release_ms=120.0,
                direction="above", max_reduction_db=round(5.0 * c, 1))])

    # -- dynamics --------------------------------------------------------
    if tight > 0.02:
        p.compressor = dyn.CompressorParams(
            enabled=True,
            threshold_db=float(-14.0 - 10.0 * tight),
            ratio=float(1.0 + 1.4 * tight),
            attack_ms=20.0 - 8.0 * tight,
            release_ms=200.0 - 40.0 * tight,
            knee_db=10.0, detector="rms",
            mix=float(np.clip(0.25 + 0.4 * tight, 0.0, 0.75)),
            makeup_db=0.0)
    if c > 0.25:
        p.multiband = dyn.MultibandParams(
            enabled=True, crossovers_hz=[180.0, 2400.0],
            bands=[
                dyn.CompressorParams(enabled=True, threshold_db=-22.0,
                                     ratio=float(1.0 + 0.6 * c), attack_ms=25.0,
                                     release_ms=200.0, mix=round(0.6 * c, 2)),
                dyn.CompressorParams(enabled=True, threshold_db=-21.0,
                                     ratio=float(1.0 + 0.7 * c), attack_ms=18.0,
                                     release_ms=170.0, mix=round(0.55 * c, 2)),
                dyn.CompressorParams(enabled=True, threshold_db=-26.0,
                                     ratio=float(1.0 + 0.5 * c), attack_ms=10.0,
                                     release_ms=140.0, mix=round(0.4 * c, 2))])

    # -- harmonic texture -------------------------------------------------
    if c > 0.1:
        p.saturation = nl.SaturationParams(
            enabled=True, shaper="tanh", drive_db=round(3.0 * c, 2),
            mix=round(0.12 * c, 3), oversample=4, compensate="rms")

    # -- image -------------------------------------------------------------
    if c > 0.2:
        p.stereo = st.StereoParams(enabled=True, width=round(1.0 + 0.12 * c, 3))

    p.output_trim_db = s.output_trim_db
    p.loudness_match = True
    p.loudness_match_mode = "lufs"
    p.limiter = dyn.LimiterParams(enabled=True, ceiling_db=s.ceiling_db,
                                  lookahead_ms=6.0)
    return p


def default_profile_path() -> str | None:
    """Location of the shipped default profile, if it exists."""
    env = os.environ.get("TIMBREFORGE_PROFILE")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(here, "profiles", "reference_character.json")
    return candidate if os.path.isfile(candidate) else None


def preset_available() -> dict[str, Callable[[], Profile]]:
    return {"Neutral \u00b7 level matched": preset_bypass,
            "Gentle character": preset_gentle_character,
            "Reference character (hypothesis)": preset_reference_hypothesis}
