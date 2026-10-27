"""Playback engine used for A/B auditioning.

Two interchangeable backends:

* ``qt``  - QAudioSink from Qt Multimedia (present in a full PySide6 install).
* ``sd``  - python-sounddevice, used when Qt Multimedia is unavailable.
* ``none``- no audio output in this environment; the rest of the app still works
            and the UI says so instead of failing silently.

Playback only ever resamples for the *output device*; exported audio and the
render graph always keep the source sample rate.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QIODevice, Signal

from ..dsp.audio import resample_same_speed


class AudioPlayer(QObject):
    positionChanged = Signal(float)
    stateChanged = Signal(str)      # "playing" | "paused" | "stopped"
    playbackError = Signal(str)
    finished = Signal()

    FADE_FRAMES = 256

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer: np.ndarray | None = None   # float32 (frames, channels)
        self._prev: np.ndarray | None = None
        self._fade = 0
        self._sr = 44100
        self._pos = 0
        self._playing = False
        self._state = "stopped"
        self._backend = "none"
        self._name = "no audio output"
        self._dev_sr = 44100
        self._dev_ch = 2
        self._lock = threading.Lock()
        self._qt_sink = None
        self._qt_source = None
        self._sd_stream = None
        self._sd_frames_per_block = 1024
        self._setup_backend()

    # -- backend selection ------------------------------------------------
    def _setup_backend(self):
        try:
            from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
            dev = QMediaDevices.defaultAudioOutput()
            if dev is not None and not dev.isNull():
                fmt = QAudioFormat()
                fmt.setSampleRate(48000)
                fmt.setChannelCount(2)
                fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
                if not dev.isFormatSupported(fmt):
                    fmt = dev.preferredFormat()
                if fmt.isValid() and fmt.sampleRate() > 0:
                    self._backend = "qt"
                    self._name = f"Qt Multimedia ({dev.description()})"
                    self._dev_sr = int(fmt.sampleRate())
                    self._dev_ch = max(1, int(fmt.channelCount()))
                    self._qt_fmt = fmt
                    return
        except Exception:  # noqa: BLE001 - optional dependency
            pass
        try:
            import sounddevice as sd
            info = sd.query_devices(kind="output")
            self._dev_sr = int(info.get("default_samplerate", 44100))
            self._dev_ch = 2
            self._backend = "sd"
            self._name = f"sounddevice ({info.get('name', 'output')})"
            return
        except Exception:  # noqa: BLE001
            pass
        self._backend = "none"
        self._name = "no audio output available"
        self._dev_sr = 44100
        self._dev_ch = 2

    @property
    def available(self) -> bool:
        return self._backend != "none"

    @property
    def backend_name(self) -> str:
        return self._name

    # -- public API -------------------------------------------------------
    def set_buffer(self, audio: np.ndarray | None, sr: int, *, keep_position: bool = True):
        """Swap the buffer being auditioned (used for A/B), keeping the playhead."""
        was_playing = self._playing
        if keep_position and self._buffer is not None:
            with self._lock:
                self._prev = self._buffer
                self._fade = self.FADE_FRAMES
        else:
            with self._lock:
                self._prev = None
                self._fade = 0
        if audio is None:
            self.stop()
            self._buffer = None
            return
        prepared = self._prepare(audio, sr)
        with self._lock:
            self._buffer = prepared
            self._sr = sr
        if self._pos >= prepared.shape[0]:
            self._pos = 0
        if was_playing:
            self.play(self._pos / float(self._dev_sr), restart=True)

    def _prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self._dev_ch:
            if x.shape[1] > self._dev_ch:
                x = x[:, : self._dev_ch]
            else:
                reps = int(np.ceil(self._dev_ch / x.shape[1]))
                x = np.repeat(x, reps, axis=1)[:, : self._dev_ch]
        if sr != self._dev_sr:
            x = resample_same_speed(x, sr, self._dev_sr)
        x = np.clip(x, -1.0, 1.0)
        return np.ascontiguousarray(x, dtype=np.float32)

    def duration(self) -> float:
        if self._buffer is None:
            return 0.0
        return self._buffer.shape[0] / float(self._dev_sr)

    def position(self) -> float:
        return self._pos / float(self._dev_sr)

    def is_playing(self) -> bool:
        return self._playing

    def play(self, from_s: float | None = None, *, restart: bool = False):
        if self._buffer is None:
            return
        if from_s is not None:
            self._pos = int(np.clip(from_s * self._dev_sr, 0,
                                    max(0, self._buffer.shape[0] - 1)))
        self._playing = True
        self._set_state("playing")
        if self._backend == "qt":
            self._qt_start()
        elif self._backend == "sd":
            self._sd_start()

    def pause(self):
        self._playing = False
        if self._backend == "qt":
            self._qt_stop_sink(keep_pos=True)
        elif self._backend == "sd":
            self._sd_stop_stream()
        self._set_state("paused")

    def stop(self):
        self._playing = False
        self._pos = 0
        if self._backend == "qt":
            self._qt_stop_sink(keep_pos=True)
        elif self._backend == "sd":
            self._sd_stop_stream()
        self._set_state("stopped")
        self.positionChanged.emit(0.0)

    def seek(self, seconds: float):
        if self._buffer is None:
            return
        self._pos = int(np.clip(seconds * self._dev_sr, 0,
                                max(0, self._buffer.shape[0] - 1)))
        if self._playing:
            self.play(self.position(), restart=True)
        else:
            self.positionChanged.emit(self.position())

    def shutdown(self):
        self._playing = False
        if self._backend == "qt":
            self._qt_stop_sink(keep_pos=False)
        elif self._backend == "sd":
            self._sd_stop_stream()

    def _set_state(self, s: str):
        if s != self._state:
            self._state = s
            self.stateChanged.emit(s)

    # -- shared sample reading -------------------------------------------
    def _read_chunk(self, frames: int) -> tuple[np.ndarray, bool]:
        """Return the next ``frames`` samples and whether the end was reached."""
        with self._lock:
            buf = self._buffer
            if buf is None:
                return np.zeros((frames, self._dev_ch), dtype=np.float32), True
            start = self._pos
            end = min(buf.shape[0], start + frames)
            chunk = np.array(buf[start:end], dtype=np.float32)
            if self._fade > 0 and self._prev is not None:
                prev = self._prev
                n = min(self._fade, chunk.shape[0], max(0, prev.shape[0] - start))
                if n > 0:
                    w = np.linspace(1.0, 0.0, self._fade, endpoint=False,
                                    dtype=np.float32)[:n]
                    chunk[:n] = chunk[:n] * (1 - w)[:, None] + prev[start:start + n] * w[:, None]
                self._fade = max(0, self._fade - chunk.shape[0])
                if self._fade == 0:
                    self._prev = None
            self._pos = end
            done = end >= buf.shape[0]
        if chunk.shape[0] < frames:
            pad = np.zeros((frames - chunk.shape[0], self._dev_ch), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk, done

    def _notify_position(self):
        self.positionChanged.emit(self.position())
        if self._buffer is not None and self._pos >= self._buffer.shape[0]:
            self._playing = False
            if self._backend == "qt":
                self._qt_stop_sink(keep_pos=True)
            elif self._backend == "sd":
                self._sd_stop_stream()
            self._set_state("stopped")
            self.finished.emit()

    # -- Qt backend -------------------------------------------------------
    def _qt_start(self):
        try:
            from PySide6.QtMultimedia import QAudioSink
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Qt Multimedia unavailable: {exc}")
            self._playing = False
            return
        self._qt_stop_sink(keep_pos=True)
        try:
            sink = QAudioSink(self._qt_fmt)
            sink.setBufferSize(max(4096, int(self._dev_sr * 0.15) * self._dev_ch * 4))
            source = _QtSource(self)
            source.open(QIODevice.ReadOnly)
            sink.start(source)
            self._qt_sink = sink
            self._qt_source = source
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Could not start audio output: {exc}")
            self._playing = False
            self._set_state("stopped")

    def _qt_stop_sink(self, keep_pos: bool):
        try:
            if self._qt_sink is not None:
                self._qt_sink.stop()
                self._qt_sink.deleteLater()
        except Exception:  # noqa: BLE001
            pass
        self._qt_sink = None
        self._qt_source = None

    # -- sounddevice backend ---------------------------------------------
    def _sd_start(self):
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"sounddevice unavailable: {exc}")
            self._playing = False
            return
        self._sd_stop_stream()

        def callback(outdata, frames, time_info, status):  # noqa: ANN001
            try:
                chunk, _ = self._read_chunk(frames)
                outdata[:] = chunk
            except Exception:  # noqa: BLE001
                outdata.fill(0)

        try:
            stream = sd.OutputStream(samplerate=self._dev_sr, channels=self._dev_ch,
                                     dtype="float32", callback=callback,
                                     blocksize=self._sd_frames_per_block)
            stream.start()
            self._sd_stream = stream
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Could not open audio output: {exc}")
            self._playing = False
            self._set_state("stopped")

    def _sd_stop_stream(self):
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._sd_stream = None


class _QtSource(QIODevice):
    """Pull-mode source device fed from the numpy buffer."""

    def __init__(self, player: AudioPlayer):
        super().__init__()
        self._player = player
        self._bytes_per_frame = player._dev_ch * (
            4 if player._qt_fmt.sampleFormat().name == "Float" else 2)

    def readData(self, maxlen: int) -> bytes:  # noqa: N802
        if not self._player._playing:
            return b"\x00" * int(maxlen)
        frames = max(1, int(maxlen) // self._bytes_per_frame)
        chunk, done = self._player._read_chunk(frames)
        if self._player._qt_fmt.sampleFormat().name == "Float":
            data = chunk.astype(np.float32, copy=False).tobytes()
        else:
            data = (np.clip(chunk, -1, 1) * 32767.0).astype(np.int16).tobytes()
        if done:
            self._player._playing = False
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._player._notify_position)
        else:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._player._notify_position)
        return data

    def bytesAvailable(self) -> int:  # noqa: N802
        return 1 << 20
