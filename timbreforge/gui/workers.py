"""Background jobs.  Nothing expensive is allowed to run on the UI thread.

* :class:`ImportWorker` decodes the file *and* measures it.
* :class:`RenderWorker` renders a region, measures both sides, computes the
  level-matched A/B gains and prepares the playback buffers.
* :class:`ExportWorker` renders the whole file, writes it and verifies it.

The main window only assigns finished data to widgets.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..dsp.audio import AudioData, ExportSettings, load_audio, save_audio
from ..dsp.graph import Cancelled, Profile, render
from ..dsp.meter import Stats, integrated_loudness, measure, rms_db
from ..dsp.playback import DeviceFormat, PlaybackBuffer, prepare_playback


class _CancellableThread(QThread):
    progress = Signal(float, str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def should_cancel(self) -> bool:
        return self._cancel

    def _emit(self, frac: float, label: str):
        self.progress.emit(float(max(0.0, min(1.0, frac))), label)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


class ImportWorker(_CancellableThread):
    done = Signal(object, object)      # AudioData, Stats

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):  # noqa: D401
        try:
            self._emit(0.05, "Decoding")
            audio = load_audio(self._path)
            if self.should_cancel():
                return
            self._emit(0.45, "Measuring")
            stats = measure(audio.samples, audio.sample_rate)
            self._emit(1.0, "Ready")
            self.done.emit(audio, stats)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ---------------------------------------------------------------------------
# preview render
# ---------------------------------------------------------------------------


def _channel_levels(buf: np.ndarray) -> list[tuple[float, float, float]]:
    """Cheap per-channel (RMS, peak, true-peak) in dB for the level meter.

    The meter's third value is its peak-hold marker; using the sample peak there
    keeps this O(n) - the real true peak is measured once per render, not 25
    times a second on the UI thread.
    """
    if buf.ndim == 1:
        buf = buf[:, None]
    out = []
    for c in range(min(2, buf.shape[1])):
        ch = buf[:, c]
        if ch.size == 0:
            out.append((-60.0, -60.0, -60.0))
            continue
        rms = float(np.sqrt(np.mean(ch * ch)))
        peak = float(np.max(np.abs(ch)))
        r = float(20.0 * np.log10(max(rms, 1e-6)))
        k = float(20.0 * np.log10(max(peak, 1e-6)))
        out.append((max(r, -60.0), max(k, -60.0), max(k, -60.0)))
    while len(out) < 2:
        out.append(out[-1] if out else (-60.0, -60.0, -60.0))
    return out


@dataclass
class RenderPayload:
    """Everything the UI needs after a preview render - all precomputed."""

    samples: np.ndarray
    start_frame: int
    stats_in: Stats | None
    stats_out: Stats | None
    warnings: list[str] = field(default_factory=list)
    gain_proc_db: float = 0.0
    gain_orig_db: float = 0.0
    levels_in: list = field(default_factory=list)
    levels_out: list = field(default_factory=list)
    play_processed: PlaybackBuffer | None = None
    play_original: PlaybackBuffer | None = None
    reports: list = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


class RenderWorker(_CancellableThread):
    done = Signal(object)          # RenderPayload
    cancelled = Signal()

    def __init__(self, samples: np.ndarray, sr: int, profile: Profile,
                 start_frame: int = 0, end_frame: int | None = None,
                 parent=None, *, device: DeviceFormat | None = None,
                 target_loudness: float | None = None):
        super().__init__(parent)
        self._samples = samples
        self._sr = int(sr)
        self._profile = profile
        self._start = int(start_frame)
        self._end = int(end_frame) if end_frame is not None else int(samples.shape[0])
        self._device = device or DeviceFormat(self._sr, 2, "float32")
        self._target = target_loudness

    def run(self):  # noqa: D401
        try:
            seg = self._samples[self._start:self._end]
            if seg.shape[0] == 0:
                self.failed.emit("The selected region is empty.")
                return
            result = render(seg, self._sr, self._profile,
                            progress=lambda f, l: self._emit(0.75 * f, l),
                            should_cancel=self.should_cancel,
                            target_loudness=self._target)
            if self.should_cancel():
                self.cancelled.emit()
                return

            self._emit(0.78, "Measuring")
            stats_out = measure(result.samples, self._sr)
            stats_in = measure(seg, self._sr)
            if self.should_cancel():
                self.cancelled.emit()
                return

            # Level-matched A/B gains (attenuation only).  These are *numbers* the
            # UI applies through the player's gain stage, so toggling level match
            # or A/B never re-renders anything.
            gain_proc = gain_orig = 0.0
            try:
                li = integrated_loudness(seg, self._sr)
                lp = integrated_loudness(result.samples, self._sr)
                d = li - lp
                if d < 0:
                    gain_proc = d        # processed is louder -> attenuate it
                else:
                    gain_orig = -d       # input is louder -> attenuate the input
            except Exception:  # noqa: BLE001
                traceback.print_exc()

            self._emit(0.86, "Preparing playback")
            play_proc = play_orig = None
            try:
                play_proc = prepare_playback(result.samples, self._sr, self._device)
                play_orig = prepare_playback(seg, self._sr, self._device)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

            payload = RenderPayload(
                samples=result.samples,
                start_frame=self._start,
                stats_in=stats_in,
                stats_out=stats_out,
                warnings=[w for w in result.warnings if w],
                gain_proc_db=gain_proc,
                gain_orig_db=gain_orig,
                levels_in=_channel_levels(seg),
                levels_out=_channel_levels(result.samples),
                play_processed=play_proc,
                play_original=play_orig,
                reports=result.reports,
            )
            self._emit(1.0, "Done")
            self.done.emit(payload)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class ExportWorker(_CancellableThread):
    done = Signal(dict)
    cancelled = Signal()

    def __init__(self, audio: AudioData, profile: Profile,
                 settings: ExportSettings, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._profile = profile
        self._settings = settings

    def run(self):  # noqa: D401
        try:
            self._emit(0.0, "Rendering")
            result = render(
                self._audio.samples, self._audio.sample_rate, self._profile,
                progress=lambda f, l: self._emit(0.9 * f, l),
                should_cancel=self.should_cancel)
            if self.should_cancel():
                raise Cancelled()
            out = AudioData(samples=result.samples, sample_rate=self._audio.sample_rate,
                            title=self._audio.title, artist=self._audio.artist)
            facts = save_audio(
                out, self._settings,
                progress=lambda f, l: self._emit(0.9 + 0.1 * f, l),
                cancel=self.should_cancel)
            facts["render"] = {
                "warnings": result.warnings,
                "loudness_match_db": result.loudness_match_db,
                "latency_samples": result.latency_samples,
                "reports": result.reports,
            }
            self.done.emit(facts)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ---------------------------------------------------------------------------
# audition fallback (no audio backend available)
# ---------------------------------------------------------------------------


class AuditionWorker(_CancellableThread):
    """Render the preview region to a temp file so it can be opened by the OS."""

    done = Signal(str)
    cancelled = Signal()

    def __init__(self, samples: np.ndarray, sr: int, profile: Profile,
                 start_frame: int, end_frame: int, parent=None):
        super().__init__(parent)
        self._samples = samples
        self._sr = sr
        self._profile = profile
        self._start = int(start_frame)
        self._end = int(end_frame)

    def run(self):  # noqa: D401
        import tempfile

        from ..dsp.audio import ExportSettings
        try:
            seg = self._samples[self._start:self._end]
            self._emit(0.05, "Rendering for audition")
            result = render(seg, self._sr, self._profile,
                            progress=lambda f, l: self._emit(0.85 * f, l),
                            should_cancel=self.should_cancel)
            if self.should_cancel():
                raise Cancelled()
            fd, path = tempfile.mkstemp(prefix="timbreforge_audition_", suffix=".wav")
            os.close(fd)
            save_audio(AudioData(samples=result.samples, sample_rate=self._sr),
                       ExportSettings(path=path, format="WAV", subtype="PCM_24",
                                      overwrite=True), sample_rate=self._sr)
            self._emit(1.0, "Opening")
            self.done.emit(path)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)
