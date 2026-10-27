"""Background jobs so the window never blocks during render or export."""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..dsp.audio import AudioData, ExportSettings, save_audio
from ..dsp.graph import Cancelled, Profile, render


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


class RenderWorker(_CancellableThread):
    """Renders a region (or the whole file) for preview."""

    done = Signal(object)
    cancelled = Signal()

    def __init__(self, samples: np.ndarray, sr: int, profile: Profile,
                 start_frame: int = 0, end_frame: int | None = None, parent=None,
                 target_loudness: float | None = None):
        super().__init__(parent)
        self._samples = samples
        self._sr = sr
        self._profile = profile
        self._start = int(start_frame)
        self._end = int(end_frame) if end_frame is not None else int(samples.shape[0])
        #: loudness of the *whole* input, so a preview window lands at the same
        #: level as the export instead of matching only its own little region
        self._target = target_loudness

    def run(self):  # noqa: D401
        try:
            seg = self._samples[self._start:self._end]
            if seg.shape[0] == 0:
                self.failed.emit("The selected region is empty.")
                return
            result = render(seg, self._sr, self._profile,
                            progress=self._emit, should_cancel=self.should_cancel,
                            target_loudness=self._target)
            result.warnings.append("")  # placeholder keeps the list non-empty
            result.warnings = [w for w in result.warnings if w]
            result.start_frame = self._start  # type: ignore[attr-defined]
            self.done.emit(result)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


class ExportWorker(_CancellableThread):
    """Renders the whole file and writes it."""

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
                progress=lambda f, l: self._emit(0.0 + 0.9 * f, l),
                should_cancel=self.should_cancel,
            )
            if self.should_cancel():
                raise Cancelled()
            out = AudioData(samples=result.samples, sample_rate=self._audio.sample_rate,
                            title=self._audio.title, artist=self._audio.artist)
            facts = save_audio(
                out, self._settings,
                progress=lambda f, l: self._emit(0.9 + 0.1 * f, l),
                cancel=self.should_cancel,
            )
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
