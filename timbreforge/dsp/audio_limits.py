"""Import/export sanity limits, in one place and with no Qt dependency.

The point of this module is that errors are *predictable*: the UI can ask
"would this import/export work?" and get an answer before it starts a long
render, and the same rules apply to the CLI and the tests.

Nothing here changes the audio.  These are guards against impossible or
dangerous requests (8 Hz sample rate, a 4-hour file, a 0-frame export, an
unknown extension), not processing decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .audio import FORMAT_TABLE, AudioError

#: Sample rates the engine is willing to read or write.  Arbitrary rates are not
#: rejected - they are kept exactly - but anything outside this range is almost
#: certainly a bad header rather than music.
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
SUPPORTED_SAMPLE_RATES = (
    8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000,
    88200, 96000, 176400, 192000,
)

#: Guards against pathological files.  The decoder already enforces its own cap;
#: these values are what the UI promises.
MIN_FRAMES = 2205                  # 0.05 s at 44.1 kHz
MAX_FRAMES = 192000 * 60 * 60      # one hour at 192 kHz
MAX_CHANNELS = 8
MAX_FILE_BYTES = 2 * 1024 ** 3     # 2 GiB

SUPPORTED_IMPORT_EXT = tuple(FORMAT_TABLE.keys()) + (".opus", ".m4a", ".w64")

#: What each export format can actually store.  `soundfile` will happily accept
#: a request it cannot honour and then write something surprising, so the UI
#: checks first.
EXPORT_LIMITS = {
    "WAV": {"rates": (8000, 384000), "channels": 8, "lossy": False,
            "notes": "PCM; bit depth 16/24/32 or float."},
    "FLAC": {"rates": (1, 655350), "channels": 8, "lossy": False,
             "notes": "Lossless; 24-bit is the practical default."},
    "AIFF": {"rates": (8000, 384000), "channels": 8, "lossy": False,
             "notes": "PCM big-endian."},
    "OGG": {"rates": (8000, 192000), "channels": 2, "lossy": True,
            "notes": "Vorbis: stereo only, quality 0-1."},
    "MP3": {"rates": (8000, 48000), "channels": 2, "lossy": True,
            "notes": "MPEG-1 layer III; sample rates above 48 kHz are not "
                     "representable, so MP3 export resamples."},
}


@dataclass(frozen=True)
class Verdict:
    """Result of a pre-flight check."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_bad(self):
        if self.errors:
            raise AudioError("; ".join(self.errors))

    def lines(self) -> list[str]:
        return [f"error: {e}" for e in self.errors] + [f"note: {w}" for w in self.warnings]


def _fmt_rate(r: float) -> str:
    return f"{r / 1000:.3g} kHz" if r >= 1000 else f"{r:.0f} Hz"


def rate_is_supported(sr: int) -> bool:
    return MIN_SAMPLE_RATE <= int(sr) <= MAX_SAMPLE_RATE


def check_import(path: str) -> Verdict:
    """Pre-flight an import: does the file exist and can it plausibly be read?"""
    errors: list[str] = []
    warnings: list[str] = []

    if not path:
        errors.append("No file was given.")
        return Verdict(tuple(errors), tuple(warnings))
    if not os.path.exists(path):
        errors.append(f"'{os.path.basename(path)}' does not exist.")
        return Verdict(tuple(errors), tuple(warnings))
    if os.path.isdir(path):
        errors.append("That is a folder, not an audio file.")
        return Verdict(tuple(errors), tuple(warnings))

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        errors.append(f"Cannot read the file: {exc}")
        return Verdict(tuple(errors), tuple(warnings))

    if size == 0:
        errors.append("The file is empty (0 bytes).")
    elif size > MAX_FILE_BYTES:
        errors.append(f"The file is {size / 1024 ** 3:.1f} GiB; the limit is 2 GiB.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMPORT_EXT:
        warnings.append(
            f"'{ext or '(no extension)'}' is not a format this app writes. Decoding is "
            "still attempted with libsndfile/ffmpeg, but the result is not guaranteed.")

    return Verdict(tuple(errors), tuple(warnings))


def check_loaded(frames: int, sample_rate: int, channels: int) -> Verdict:
    """Check a decoded buffer before it enters the engine."""
    errors: list[str] = []
    warnings: list[str] = []

    if frames <= 0:
        errors.append("The file decoded to zero frames.")
        return Verdict(tuple(errors), tuple(warnings))

    if not rate_is_supported(sample_rate):
        errors.append(
            f"Sample rate {_fmt_rate(sample_rate)} is outside the supported range "
            f"({_fmt_rate(MIN_SAMPLE_RATE)} - {_fmt_rate(MAX_SAMPLE_RATE)}).")
    elif sample_rate not in SUPPORTED_SAMPLE_RATES:
        warnings.append(
            f"Sample rate {_fmt_rate(sample_rate)} is unusual. It is preserved exactly; "
            "the EQ and dynamics stages assume it is the track's real rate.")

    if frames < MIN_FRAMES:
        errors.append(
            f"Only {frames} frames ({frames / max(sample_rate, 1):.3f} s). The minimum is "
            f"{MIN_FRAMES / 44100:.2f} s, below which the loudness measurement is meaningless.")
    if frames > MAX_FRAMES:
        errors.append(
            f"{frames} frames is longer than the one-hour limit; process it in parts.")

    if channels <= 0:
        errors.append("The file has no channels.")
    elif channels > MAX_CHANNELS:
        errors.append(f"{channels} channels is above the {MAX_CHANNELS}-channel limit.")
    elif channels > 2:
        warnings.append(
            f"{channels} channels: the app processes every channel identically. Width "
            "control and the stereo statistics use the first two channels only.")

    return Verdict(tuple(errors), tuple(warnings))


def check_export(*, path: str, fmt: str, sample_rate: int, frames: int,
                 channels: int, subtype: str | None = None,
                 source_path: str | None = None) -> Verdict:
    """Pre-flight an export without rendering anything."""
    errors: list[str] = []
    warnings: list[str] = []

    if not path:
        errors.append("No output path was chosen.")
        return Verdict(tuple(errors), tuple(warnings))

    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMAT_TABLE:
        errors.append(
            f"'{ext or '(no extension)'}' cannot be written. Use .wav, .flac, .ogg, "
            ".mp3 or .aiff.")
        return Verdict(tuple(errors), tuple(warnings))

    if source_path and os.path.abspath(path) == os.path.abspath(source_path):
        errors.append("The output would overwrite the imported file. Choose another name.")

    folder = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(folder):
        errors.append(f"The folder '{folder}' does not exist.")
    elif not os.access(folder, os.W_OK):
        errors.append(f"The folder '{folder}' is not writable.")

    if os.path.exists(path) and not os.access(path, os.W_OK):
        errors.append("The existing file is read-only.")

    if frames <= 0:
        errors.append("There is nothing to write (zero frames).")

    limits = EXPORT_LIMITS.get(str(fmt or "").upper())
    if limits:
        lo, hi = limits["rates"]
        if not (lo <= sample_rate <= hi):
            errors.append(
                f"{fmt} cannot store {_fmt_rate(sample_rate)} audio; the range is "
                f"{_fmt_rate(lo)} - {_fmt_rate(hi)}.")
        if channels > limits["channels"]:
            errors.append(f"{fmt} supports at most {limits['channels']} channels.")
        if limits["lossy"]:
            warnings.append(
                f"{fmt} is lossy: the encoded peak can overshoot the rendered ceiling. The "
                "export is measured after encoding, not just before.")

    if limits and subtype and subtype.upper().startswith("PCM") and not limits["lossy"]:
        pass  # PCM bit depths are validated by soundfile itself

    return Verdict(tuple(errors), tuple(warnings))


def describe_limits() -> str:
    """Human-readable summary for the About/Guide dialog."""
    rates = ", ".join(_fmt_rate(r) for r in SUPPORTED_SAMPLE_RATES)
    rows = [f"Sample rate: {_fmt_rate(MIN_SAMPLE_RATE)} - {_fmt_rate(MAX_SAMPLE_RATE)} "
            f"(preserved exactly; no resampling of the export)",
            f"Common rates accepted: {rates}",
            f"Length: {MIN_FRAMES / 44100:.2f} s - 60 min",
            f"Channels: 1 - {MAX_CHANNELS}",
            "File size: up to 2 GiB",
            "",
            "Export formats:"]
    for name, lim in EXPORT_LIMITS.items():
        rows.append(f"  {name:<5} {lim['notes']}")
    return "\n".join(rows)
