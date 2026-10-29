"""Playback format handling and buffer preparation.

Kept free of any GUI/audio-API imports so it can be unit tested: everything here
is plain numpy plus bytes slicing.

Design notes
------------
* The reference ``source`` sample rate is never altered for rendering or export.
  Playback may need a different rate for the output device; that conversion
  happens here, once per buffer, and never touches exported audio.
* Buffers are converted to the device's native sample type *once* and stored as
  ``bytes``, so the pull callback does nothing but a slice.  That is what keeps
  playback from stuttering while the UI redraws.
* A short fade-in is applied when swapping buffers (A/B), which removes clicks
  without needing to mix two streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .audio import resample_same_speed

#: sample types we know how to hand to a device
SUPPORTED_DTYPES = ("float32", "int16", "int32")


@dataclass(frozen=True)
class DeviceFormat:
    """A concrete output format: rate, channel count and sample type."""

    sample_rate: int = 44100
    channels: int = 2
    dtype: str = "float32"

    def frames_to_bytes(self, frames: int) -> int:
        return int(frames) * self.channels * np.dtype(self.dtype).itemsize

    def bytes_to_frames(self, nbytes: int) -> int:
        per = self.channels * np.dtype(self.dtype).itemsize
        return int(nbytes) // per

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * np.dtype(self.dtype).itemsize

    def describe(self) -> str:
        return (f"{self.sample_rate} Hz, {self.channels} ch, "
                f"{np.dtype(self.dtype).itemsize * 8}-bit {self.dtype}")


def pick_device_format(source_rate: int, source_channels: int,
                       supported: Iterable[DeviceFormat] | None = None,
                       preferred: DeviceFormat | None = None) -> DeviceFormat:
    """Choose an output format, preferring one that needs no conversion.

    ``supported`` is a list of formats the backend has confirmed it accepts, in
    the backend's own order of preference.  If none of them matches the source,
    the first supported entry wins; ``preferred`` is the last resort.
    """
    want_ch = min(2, max(1, int(source_channels)))
    supported = list(supported or [])
    if supported:
        for fmt in supported:
            if fmt.sample_rate == source_rate and fmt.channels == want_ch:
                return fmt
        for fmt in supported:
            if fmt.sample_rate == source_rate:
                return fmt
        return supported[0]
    if preferred is not None:
        return preferred
    return DeviceFormat(int(source_rate), want_ch, "float32")


def _to_dtype(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return x.astype(np.float32, copy=False)
    if dtype == "int16":
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    if dtype == "int32":
        return (np.clip(x, -1.0, 1.0) * 2147483647.0).astype(np.int32)
    raise ValueError(f"unsupported playback sample type {dtype!r}")


_FULL_SCALE = {"float32": 1.0, "int16": 32767.0, "int32": 2147483647.0}


def apply_gain_bytes(data: bytes, dtype: str, gain_linear: float) -> bytes:
    """Scale a block of device-format audio in place-ish, without changing dtype.

    The naive version of this (multiply the raw samples, clip to +/-1) silently
    destroys integer output: an int16 sample of 20000 became 1.  Integer formats
    are therefore decoded to the -1..1 domain, scaled, clipped and re-encoded.
    """
    if gain_linear == 1.0 or not data:
        return data
    full = _FULL_SCALE.get(dtype)
    if full is None:
        return data
    arr = np.frombuffer(data, dtype=np.dtype(dtype))
    if dtype == "float32":
        out = np.clip(arr * np.float32(gain_linear), -1.0, 1.0)
        return out.astype(np.float32, copy=False).tobytes()
    out = arr.astype(np.float32) / full
    out *= np.float32(gain_linear)
    np.clip(out, -1.0, 1.0, out=out)
    return np.rint(out * full).astype(np.dtype(dtype)).tobytes()


@dataclass
class PlaybackBuffer:
    """A ready-to-stream buffer in the output device's format."""

    data: bytes
    format: DeviceFormat
    source_rate: int = 44100
    source_frames: int = 0
    resampled: bool = False
    gain_db: float = 0.0

    @property
    def frames(self) -> int:
        return self.format.bytes_to_frames(len(self.data))

    @property
    def duration(self) -> float:
        if not self.format.sample_rate:
            return 0.0
        return self.frames / float(self.format.sample_rate)

    def read(self, byte_offset: int, max_bytes: int) -> bytes:
        """Return up to ``max_bytes`` starting at ``byte_offset`` (frame aligned)."""
        start = byte_offset - (byte_offset % self.format.bytes_per_frame)
        if start >= len(self.data):
            return b""
        return self.data[start:start + max_bytes]

    def as_array(self) -> np.ndarray:
        """Decoded view (float32, frames x channels) - used by tests and the meter."""
        arr = np.frombuffer(self.data, dtype=self.format.dtype)
        return arr.reshape(-1, self.format.channels).astype(np.float32)


def prepare_playback(x: np.ndarray, source_rate: int, fmt: DeviceFormat, *,
                     gain_db: float = 0.0, fade_in_ms: float = 0.0) -> PlaybackBuffer:
    """Convert audio to a :class:`PlaybackBuffer` for ``fmt``.

    ``gain_db`` is applied here (never persisted); ``fade_in_ms`` ramps the first
    few milliseconds so swapping buffers does not click.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    src_frames = int(a.shape[0])

    # channel map: truncate or duplicate to the device's count
    if a.shape[1] != fmt.channels:
        if a.shape[1] > fmt.channels:
            a = a[:, : fmt.channels]
        else:
            reps = int(np.ceil(fmt.channels / a.shape[1]))
            a = np.repeat(a, reps, axis=1)[:, : fmt.channels]

    if gain_db:
        a = a * (10.0 ** (gain_db / 20.0))

    resampled = False
    if fmt.sample_rate != source_rate and source_rate > 0:
        a = resample_same_speed(a, int(source_rate), int(fmt.sample_rate))
        resampled = True

    a = np.clip(a, -1.0, 1.0)

    if fade_in_ms > 0 and a.shape[0] > 1:
        n = int(min(fmt.sample_rate * fade_in_ms / 1000.0, a.shape[0]))
        if n > 1:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
            a = a.copy()
            a[:n] *= ramp

    data = _to_dtype(a, fmt.dtype).tobytes()
    return PlaybackBuffer(data=data, format=fmt, source_rate=int(source_rate),
                          source_frames=src_frames, resampled=resampled,
                          gain_db=float(gain_db))


def silence_buffer(fmt: DeviceFormat, ms: float = 100.0) -> PlaybackBuffer:
    frames = max(1, int(fmt.sample_rate * ms / 1000.0))
    data = np.zeros((frames, fmt.channels), dtype=fmt.dtype).tobytes()
    return PlaybackBuffer(data=data, format=fmt, source_rate=fmt.sample_rate,
                          source_frames=frames)


def test_tone(fmt: DeviceFormat, freq: float = 440.0, seconds: float = 1.0,
              level_db: float = -18.0) -> PlaybackBuffer:
    """A short tone for the 'test audio output' button."""
    n = max(1, int(fmt.sample_rate * seconds))
    t = np.arange(n) / float(fmt.sample_rate)
    amp = 10.0 ** (level_db / 20.0)
    env = np.minimum(1.0, np.minimum(t / 0.01, (seconds - t) / 0.05))
    s = amp * env * np.sin(2 * np.pi * freq * t)
    x = np.stack([s] * fmt.channels, axis=1)
    return prepare_playback(x, fmt.sample_rate, fmt)
