"""Audio decoding / encoding and format handling.

Design notes
------------
* Everything is decoded to float64 in ``[frames, channels]`` layout.  Float
  samples outside [-1, 1] are kept (they are meaningful: they are how a
  compressed/inter-sample-peak source is represented) and reported instead of
  being silently clipped on import.
* The source sample rate is preserved by default.  Sample-rate conversion, when
  explicitly requested on export, is done with a polyphase FIR that keeps the
  duration (and therefore the pitch and timing) unchanged.
* Exports are written to a temporary file next to the destination and then
  atomically renamed, so a cancelled or failed export never leaves a truncated
  file behind.  The original input file is never overwritten implicitly.
* Encoding is done in blocks.  Besides making progress and cancellation real,
  this avoids a crash in some libsndfile builds whose Ogg Vorbis encoder
  segfaults on a single very large write (reproduced here above 2^22 samples).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


class AudioError(RuntimeError):
    """Raised for any user-facing import/export problem."""


# ---------------------------------------------------------------------------
# format table
# ---------------------------------------------------------------------------

#: extension -> (soundfile format, default subtype, human label)
FORMAT_TABLE: dict[str, tuple[str, str, str]] = {
    ".wav": ("WAV", "PCM_24", "WAV (PCM)"),
    ".flac": ("FLAC", "PCM_24", "FLAC"),
    ".ogg": ("OGG", "VORBIS", "Ogg Vorbis"),
    ".oga": ("OGG", "VORBIS", "Ogg Vorbis"),
    ".mp3": ("MP3", "MPEG_LAYER_III", "MP3"),
    ".aiff": ("AIFF", "PCM_24", "AIFF"),
    ".aif": ("AIFF", "PCM_24", "AIFF"),
}

IMPORT_FILTER = (
    "Audio files (*.wav *.flac *.ogg *.oga *.mp3 *.aiff *.aif);;"
    "WAV (*.wav);;FLAC (*.flac);;Ogg Vorbis (*.ogg *.oga);;MP3 (*.mp3);;All files (*)"
)
EXPORT_FILTER = (
    "WAV 24-bit (*.wav);;WAV 16-bit (*.wav);;FLAC 24-bit (*.flac);;"
    "Ogg Vorbis (*.ogg);;MP3 320 kbps (*.mp3)"
)

#: subtype -> bit depth, used for dither decisions and reporting
_BITS_FOR_SUBTYPE = {
    "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_S8": 8, "FLOAT": 32, "DOUBLE": 64,
}


def bits_for_subtype(subtype: str | None) -> int | None:
    return _BITS_FOR_SUBTYPE.get(subtype or "")


def format_for_extension(path: str) -> tuple[str, str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMAT_TABLE:
        raise AudioError(
            f"Unsupported export extension '{ext or '(none)'}'. "
            "Use .wav, .flac, .ogg or .mp3."
        )
    return FORMAT_TABLE[ext]


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


@dataclass
class AudioData:
    """Decoded audio, float64, shape (frames, channels)."""

    samples: np.ndarray
    sample_rate: int
    path: str | None = None
    format: str | None = None
    subtype: str | None = None
    bits: int | None = None
    title: str = ""
    artist: str = ""
    warnings: list[str] = field(default_factory=list)

    # -- geometry ---------------------------------------------------------
    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def is_stereo(self) -> bool:
        return self.channels >= 2

    @property
    def is_empty(self) -> bool:
        return self.frames == 0

    @property
    def filename(self) -> str:
        return os.path.basename(self.path) if self.path else "untitled"

    # -- derived audio ----------------------------------------------------
    def mono(self) -> np.ndarray:
        """Average downmix, shape (frames,)."""
        if self.channels == 1:
            return self.samples[:, 0].copy()
        return self.samples.mean(axis=1)

    def copy(self) -> "AudioData":
        return AudioData(
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            path=self.path,
            format=self.format,
            subtype=self.subtype,
            bits=self.bits,
            title=self.title,
            artist=self.artist,
            warnings=list(self.warnings),
        )

    def trimmed(self, start: int, stop: int) -> "AudioData":
        start = max(0, int(start))
        stop = min(self.frames, int(stop))
        if stop <= start:
            raise AudioError("Empty selection.")
        out = self.copy()
        out.samples = self.samples[start:stop].copy()
        return out

    def info(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "format": self.format,
            "subtype": self.subtype,
            "bits": self.bits,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration": self.duration,
            "title": self.title,
            "artist": self.artist,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_MAX_DECODE_SECONDS = 30 * 60  # sanity guard against absurd files
_MIN_SECONDS = 0.05


def _ffmpeg_decode(path: str) -> AudioData:
    """Fallback decoder for containers libsndfile cannot read (if ffmpeg exists)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioError(
            "libsndfile could not decode this file and ffmpeg is not installed."
        )
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        proc = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", path,
             "-vn", "-c:a", "pcm_f32le", "-f", "wav", tmp],
            capture_output=True,
            timeout=600,
        )
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioError("ffmpeg could not decode the file: "
                             + (msg[-1] if msg else "unknown error"))
        return load_audio(tmp, _from_temp=True)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_audio(path: str, *, _from_temp: bool = False) -> AudioData:
    """Decode ``path`` into float64 samples, keeping the original sample rate."""
    if not path:
        raise AudioError("No file selected.")
    if not os.path.exists(path):
        raise AudioError(f"File not found:\n{path}")
    if os.path.isdir(path):
        raise AudioError(f"'{os.path.basename(path)}' is a folder, not an audio file.")
    if os.path.getsize(path) == 0:
        raise AudioError(f"'{os.path.basename(path)}' is empty (0 bytes).")

    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises many types
        try:
            return _ffmpeg_decode(path)
        except AudioError:
            raise AudioError(
                f"Unreadable audio file '{os.path.basename(path)}'.\n"
                f"Decoder said: {exc}"
            ) from exc

    if info.frames <= 0 or info.samplerate <= 0:
        raise AudioError(f"'{os.path.basename(path)}' contains no audio frames.")
    if info.duration > _MAX_DECODE_SECONDS:
        raise AudioError(
            f"File is {info.duration / 60:.1f} minutes long; the safety limit is "
            f"{_MAX_DECODE_SECONDS // 60} minutes."
        )
    if info.duration < _MIN_SECONDS:
        raise AudioError("File is too short to process (under 50 ms).")

    try:
        samples, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        try:
            return _ffmpeg_decode(path)
        except AudioError:
            raise AudioError(
                f"Could not decode the audio data in '{os.path.basename(path)}'.\n"
                f"Decoder said: {exc}"
            ) from exc

    warnings: list[str] = []
    if samples.size == 0:
        raise AudioError("Decoded audio is empty.")
    bad = ~np.isfinite(samples)
    if bad.any():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        warnings.append("Non-finite samples were replaced with silence.")
    peak = float(np.abs(samples).max())
    if peak > 1.0:
        warnings.append(
            f"Decoded peak is {peak:.3f} (above full scale). Samples are kept "
            "unclipped so the measurement remains honest."
        )
    if samples.shape[1] > 2:
        warnings.append(
            f"{samples.shape[1]} channels found; only the first two are used."
        )
        samples = samples[:, :2]

    meta = info.extra_info if isinstance(info.extra_info, dict) else {}
    return AudioData(
        samples=np.ascontiguousarray(samples, dtype=np.float64),
        sample_rate=int(sr),
        path=None if _from_temp else os.path.abspath(path),
        format=info.format,
        subtype=info.subtype,
        bits=bits_for_subtype(info.subtype),
        title=str(meta.get("title", "") or ""),
        artist=str(meta.get("artist", "") or ""),
        warnings=warnings,
    )


def probe_audio(path: str) -> dict[str, Any]:
    """Cheap metadata read (no full decode)."""
    try:
        info = sf.info(path)
        return {
            "path": path,
            "filename": os.path.basename(path),
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.duration,
        }
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Not a readable audio file: {exc}") from exc


# ---------------------------------------------------------------------------
# sample rate conversion (duration preserving)
# ---------------------------------------------------------------------------


def resample_same_speed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample that preserves duration exactly.

    This is a plain band-limited resampler: it changes the number of samples
    only, never the playback rate, pitch or tempo.  Output length is rounded to
    the nearest sample so musical timing is kept.
    """
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    channels = [signal.resample_poly(x[:, c], up, down, window=("kaiser", 9.0))
                for c in range(x.shape[1])]
    n_in = x.shape[0]
    n_out = int(round(n_in * sr_out / sr_in))
    y = np.stack(channels, axis=1)
    if y.shape[0] > n_out:
        y = y[:n_out]
    elif y.shape[0] < n_out:
        y = np.pad(y, ((0, n_out - y.shape[0]), (0, 0)))
    return y


# ---------------------------------------------------------------------------
# exporting
# ---------------------------------------------------------------------------


@dataclass
class ExportSettings:
    path: str
    format: str | None = None          # soundfile format key, else from extension
    subtype: str | None = None         # soundfile subtype
    sample_rate: int | None = None     # None -> keep source rate
    ogg_quality: float = 0.7           # 0..1 for Vorbis
    dither: bool = False
    overwrite: bool = False
    source_path: str | None = None

    def resolved(self, src_sr: int) -> tuple[str, str, int]:
        fmt, subtype, _ = format_for_extension(self.path)
        if self.format:
            fmt = self.format
        if self.subtype:
            subtype = self.subtype
        sr = int(self.sample_rate or src_sr)
        return fmt, subtype, sr


def _tpdf_dither(x: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    if bits >= 24 or bits <= 0:
        return x
    lsb = 1.0 / float(2 ** (bits - 1))
    noise = (rng.random(x.shape) - rng.random(x.shape)) * lsb
    return x + noise


def save_audio(data: AudioData | np.ndarray, settings: ExportSettings,
               sample_rate: int | None = None,
               *, progress=None, cancel=None) -> dict[str, Any]:
    """Write an audio file.  Returns a dict of facts about what was written."""
    if isinstance(data, AudioData):
        x = data.samples
        sr_src = data.sample_rate
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when writing a raw array")
        x = np.asarray(data, dtype=np.float64)
        sr_src = sample_rate
    if x.ndim == 1:
        x = x[:, None]

    dest = os.path.abspath(settings.path)
    if settings.source_path and os.path.abspath(settings.source_path) == dest:
        raise AudioError(
            "Refusing to overwrite the imported file. Choose a different name."
        )
    if os.path.exists(dest) and not settings.overwrite:
        raise AudioError(f"'{os.path.basename(dest)}' already exists.")

    fmt, subtype, sr_out = settings.resolved(sr_src)
    parent = os.path.dirname(dest) or "."
    if not os.path.isdir(parent):
        raise AudioError(f"Folder does not exist:\n{parent}")

    if progress:
        progress(0.02, "Preparing samples")
    if sr_out != sr_src:
        x = resample_same_speed(x, sr_src, sr_out)

    if settings.dither:
        bits = bits_for_subtype(subtype) or 24
        x = _tpdf_dither(x, bits, np.random.default_rng(20240517))

    if cancel and cancel():
        raise AudioError("Export cancelled.")

    # peak guard: never let an integer container wrap around
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    clipped = peak > 1.0
    if clipped:
        x = np.clip(x, -1.0, 1.0)

    fd, tmp = tempfile.mkstemp(prefix=".tf_export_",
                               suffix=os.path.splitext(dest)[1], dir=parent)
    os.close(fd)
    try:
        if progress:
            progress(0.10, "Encoding")
        kwargs: dict[str, Any] = {"samplerate": sr_out, "channels": int(x.shape[1]),
                                  "format": fmt, "subtype": subtype}
        if fmt == "OGG" and subtype == "VORBIS":
            kwargs["compression_level"] = float(np.clip(settings.ogg_quality, 0.0, 1.0))
        # Block-wise writing: keeps memory low, makes progress/cancel real, and
        # avoids a libsndfile crash on very large single writes.
        block = 1 << 18
        total = max(1, x.shape[0])
        with sf.SoundFile(tmp, mode="w", **kwargs) as handle:
            for start in range(0, x.shape[0], block):
                if cancel and cancel():
                    raise AudioError("Export cancelled.")
                handle.write(x[start:start + block])
                if progress:
                    done = min(1.0, (start + block) / total)
                    progress(0.10 + 0.78 * done, "Encoding")
        if progress:
            progress(0.92, "Finalising file")
        if os.path.getsize(tmp) == 0:
            raise AudioError("Encoder produced an empty file.")
        if os.path.exists(dest):
            os.remove(dest)
        os.replace(tmp, dest)
        tmp = ""
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AudioError(f"Encoding failed: {exc}") from exc
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if progress:
        progress(1.0, "Written")
    return {
        "path": dest,
        "format": fmt,
        "subtype": subtype,
        "sample_rate": sr_out,
        "channels": int(x.shape[1]),
        "frames": int(x.shape[0]),
        "duration": float(x.shape[0]) / sr_out,
        "bytes": os.path.getsize(dest),
        "clipped_on_write": clipped,
    }
