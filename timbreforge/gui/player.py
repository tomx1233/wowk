"""Playback with an explicit backend order and *no* work on the UI thread.

Backends
--------
1. ``sounddevice`` (PortAudio).  Preferred, because the wheel bundles PortAudio on
   Windows and macOS, and it accepts float32 buffers at the device's native rate.
2. ``Qt Multimedia`` (``QAudioSink``), used when sounddevice is missing but a full
   PySide6 build is installed.  Note that ``PySide6-Essentials`` does **not**
   contain Qt Multimedia - that was the reason playback silently did nothing on
   Windows.
3. none - the UI then offers "Audition in player", which renders the preview to a
   temporary file and opens it in the system's own audio player.  That path never
   depends on Python audio bindings at all.

Everything a backend needs is prepared *before* ``play()``: the source audio is
converted once into the device's format by
:func:`timbreforge.dsp.playback.prepare_playback`, so the audio callback only
copies bytes.
"""

from __future__ import annotations

import os
import platform
import subprocess
import weakref

import numpy as np
from PySide6.QtCore import QIODevice, QObject, Qt, QTimer, Signal

from ..dsp.playback import (DeviceFormat, PlaybackBuffer, apply_gain_bytes,
                            prepare_playback, test_tone)

#: Upper bound for the transient monitor gain used while a trim change is
#: still being rendered.  Monitoring may never go louder than this.
MONITOR_BOOST_MAX_DB = 6.0


class AudioPlayer(QObject):
    """Streams a :class:`PlaybackBuffer` to the default output device."""

    playbackError = Signal(str)
    stateChanged = Signal(str)          # "playing" | "paused" | "stopped"
    positionChanged = Signal(float)     # seconds into the installed buffer
    finished = Signal()
    backendChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backend = "none"
        self._backend_detail = "no audio output available"
        self._fmt = DeviceFormat()
        self._sd = None
        self._sd_stream = None
        self._device_index: int | None = None
        self._buffer: PlaybackBuffer | None = None
        self._byte_pos = 0
        self._playing = False
        self._state = "stopped"
        self._qt_sink = None
        self._qt_source = None
        self._tone: PlaybackBuffer | None = None
        self._gain_db = 0.0
        self._gain_linear = 1.0
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(40)
        self._pos_timer.timeout.connect(self._emit_position)
        self._detect_backend()

    # ------------------------------------------------------------------
    # backend detection
    # ------------------------------------------------------------------
    def _detect_backend(self):
        errors: list[str] = []
        try:
            import sounddevice as sd  # noqa: F401
            self._sd = sd
            info = sd.query_devices(kind="output")
            self._device_index = None
            self._backend = "sd"
            self._backend_detail = f"sounddevice / PortAudio ({info.get('name', 'default')})"
            self._fmt = self._best_sd_format(44100, 2)
            self.backendChanged.emit()
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sounddevice: {exc}")

        try:
            from PySide6.QtMultimedia import QAudioFormat, QMediaDevices
            dev = QMediaDevices.defaultAudioOutput()
            if dev is not None and not dev.isNull():
                self._backend = "qt"
                self._backend_detail = f"Qt Multimedia ({dev.description()})"
                self._fmt = self._best_qt_format(44100, 2)
                self.backendChanged.emit()
                return
            errors.append("Qt Multimedia: no default output device")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Qt Multimedia: {exc}")

        self._backend = "none"
        self._backend_detail = ("no audio output backend "
                                "(" + "; ".join(errors) + ")")
        self.backendChanged.emit()

    # -- format negotiation ------------------------------------------------
    def _best_sd_format(self, source_rate: int, channels: int) -> DeviceFormat:
        sd = self._sd
        want_ch = min(2, max(1, channels))
        candidates = [DeviceFormat(source_rate, want_ch, "float32"),
                      DeviceFormat(48000, 2, "float32"),
                      DeviceFormat(44100, 2, "float32")]
        for fmt in candidates:
            try:
                sd.check_output_settings(device=self._device_index,
                                         samplerate=fmt.sample_rate,
                                         channels=fmt.channels,
                                         dtype=fmt.dtype)
                return fmt
            except Exception:  # noqa: BLE001
                continue
        try:
            info = sd.query_devices(kind="output")
            return DeviceFormat(int(info.get("default_samplerate", 44100)),
                                min(2, int(info.get("max_output_channels", 2))),
                                "float32")
        except Exception:  # noqa: BLE001
            return DeviceFormat(44100, 2, "float32")

    def _best_qt_format(self, source_rate: int, channels: int) -> DeviceFormat:
        from PySide6.QtMultimedia import QAudioFormat, QMediaDevices
        dev = QMediaDevices.defaultAudioOutput()
        want_ch = min(2, max(1, channels))
        for dtype in ("float32", "int16", "int32"):
            for rate in (source_rate, 48000, 44100):
                probe = QAudioFormat()
                probe.setSampleRate(int(rate))
                probe.setChannelCount(int(want_ch))
                probe.setSampleFormat(self._qt_sample_format(dtype))
                if probe.isValid() and dev.isFormatSupported(probe):
                    return DeviceFormat(int(rate), int(want_ch), dtype)
        pref = dev.preferredFormat()
        dtype = {QAudioFormat.SampleFormat.Float: "float32",
                 QAudioFormat.SampleFormat.Int16: "int16",
                 QAudioFormat.SampleFormat.Int32: "int32"}.get(
                     pref.sampleFormat(), "int16")
        return DeviceFormat(int(pref.sampleRate() or 44100),
                            int(pref.channelCount() or 2), dtype)

    @staticmethod
    def _qt_sample_format(dtype: str):
        from PySide6.QtMultimedia import QAudioFormat
        return {"float32": QAudioFormat.SampleFormat.Float,
                "int16": QAudioFormat.SampleFormat.Int16,
                "int32": QAudioFormat.SampleFormat.Int32}[dtype]

    def configure_source(self, source_rate: int, channels: int):
        """Pick a device format for a newly imported file (cheap, no I/O)."""
        if self._backend == "sd" and self._sd is not None:
            self._fmt = self._best_sd_format(source_rate, channels)
        elif self._backend == "qt":
            try:
                self._fmt = self._best_qt_format(source_rate, channels)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend_detail

    @property
    def device_format(self) -> DeviceFormat:
        return self._fmt

    def duration(self) -> float:
        return self._buffer.duration if self._buffer else 0.0

    def position(self) -> float:
        if not self._buffer or not self._fmt.sample_rate:
            return 0.0
        return (self._byte_pos / self._fmt.bytes_per_frame) / float(self._fmt.sample_rate)

    def is_playing(self) -> bool:
        return self._playing

    @property
    def buffer(self) -> PlaybackBuffer | None:
        """The buffer currently installed (already in the device format)."""
        return self._buffer

    def set_gain_db(self, gain_db: float, *, allow_boost: bool = False):
        """Instant gain in the read path (no re-render, no stream restart).

        The plain call only attenuates: the level-matched A/B must never make
        monitoring louder than the rendered file.  ``allow_boost`` exists for the
        short-lived monitor trim (see :meth:`MainWindow._on_profile_changed`),
        which anticipates a re-render that is already on its way; even then the
        boost is capped, so a slider drag cannot push the output into clipping
        before the limiter has seen it.
        """
        g = float(gain_db)
        if not allow_boost:
            g = min(0.0, g)
        else:
            g = float(np.clip(g, -60.0, MONITOR_BOOST_MAX_DB))
        self._gain_db = g
        self._gain_linear = 10.0 ** (g / 20.0)

    def gain_db(self) -> float:
        return self._gain_db

    def _emit_position(self):
        if self._playing:
            self.positionChanged.emit(self.position())

    # ------------------------------------------------------------------
    # buffer handling
    # ------------------------------------------------------------------
    def stream_running(self) -> bool:
        return self._sd_stream is not None or self._qt_sink is not None

    def set_buffer(self, buffer: PlaybackBuffer | None, *, keep_position: bool = True,
                   fade_in: bool = True, gain_db: float | None = None):
        """Install a prepared buffer.  Cheap: no audio processing happens here.

        While audio is playing the stream is *not* torn down: the callback picks
        up the new buffer on its next block.  Closing and reopening PortAudio on
        every preview render is what made the app feel like the volume arrived
        seconds late, so the buffer is swapped in place instead (the first few
        milliseconds are ramped to avoid a click).
        """
        was_playing = self._playing
        pos_s = self.position() if keep_position else 0.0
        if gain_db is not None:
            self.set_gain_db(gain_db)
        if buffer is not None and fade_in and was_playing:
            # re-render the first milliseconds with a ramp so the swap is silent
            buffer = self._with_ramp(buffer)
        seamless = (was_playing and self.stream_running()
                    and buffer is not None and self._buffer is not None
                    and buffer.format == self._fmt)
        if was_playing and not seamless:
            self._stop_backend()
        self._buffer = buffer
        if buffer is None:
            self._byte_pos = 0
            self._set_state("stopped")
            return
        self._byte_pos = int(np.clip(pos_s * self._fmt.sample_rate, 0,
                                     max(0, buffer.frames - 1))) * self._fmt.bytes_per_frame
        if was_playing and not seamless:
            self._start_backend()

    @staticmethod
    def _with_ramp(buffer: PlaybackBuffer) -> PlaybackBuffer:
        frames = min(int(buffer.format.sample_rate * 0.008), buffer.frames)
        if frames < 2:
            return buffer
        arr = buffer.as_array()
        ramp = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
        arr[:frames] *= ramp
        from ..dsp.playback import _to_dtype  # local import: internal helper
        return PlaybackBuffer(data=_to_dtype(arr, buffer.format.dtype).tobytes(),
                              format=buffer.format, source_rate=buffer.source_rate,
                              source_frames=buffer.source_frames,
                              resampled=buffer.resampled, gain_db=buffer.gain_db)

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def play(self, from_seconds: float | None = None):
        if self._buffer is None or not self.available:
            return
        if from_seconds is not None:
            self._byte_pos = int(np.clip(from_seconds * self._fmt.sample_rate, 0,
                                         max(0, self._buffer.frames - 1))
                                 ) * self._fmt.bytes_per_frame
        self._playing = True
        self._set_state("playing")
        self._pos_timer.start()
        self._start_backend()

    def pause(self):
        self._stop_backend()
        self._playing = False
        self._pos_timer.stop()
        self._set_state("paused")

    def stop(self):
        self._stop_backend()
        self._playing = False
        self._byte_pos = 0
        self._pos_timer.stop()
        self._set_state("stopped")

    def seek(self, seconds: float):
        """Move the read cursor.  The stream keeps running, so scrubbing is instant."""
        if self._buffer is None:
            return
        self._byte_pos = int(np.clip(seconds * self._fmt.sample_rate, 0,
                                     max(0, self._buffer.frames - 1))
                             ) * self._fmt.bytes_per_frame

    def shutdown(self):
        self._stop_backend()
        self._pos_timer.stop()
        self._playing = False

    def _set_state(self, s: str):
        if s != self._state:
            self._state = s
            self.stateChanged.emit(s)

    # ------------------------------------------------------------------
    # chunk reading (shared by both backends)
    # ------------------------------------------------------------------
    def _read_frames(self, frames: int) -> tuple[bytes, bool]:
        if self._buffer is None:
            return b"", True
        want = self._fmt.frames_to_bytes(max(1, frames))
        chunk = self._buffer.read(self._byte_pos, want)
        if chunk and self._gain_linear != 1.0:
            chunk = apply_gain_bytes(chunk, self._fmt.dtype, self._gain_linear)
        self._byte_pos += len(chunk)
        done = self._byte_pos >= len(self._buffer.data)
        if done:
            self._playing = False
        return chunk, done

    def _notify_finished(self):
        self._pos_timer.stop()
        self._set_state("stopped")
        self.finished.emit()

    # ------------------------------------------------------------------
    # sounddevice backend
    # ------------------------------------------------------------------
    def _start_backend(self):
        if self._backend == "sd":
            self._start_sd()
        elif self._backend == "qt":
            self._start_qt()

    def _stop_backend(self):
        self._stop_sd()
        self._stop_qt()

    def _start_sd(self):
        if self._sd is None:
            return
        self._stop_sd()
        fps_bytes = self._fmt.bytes_per_frame
        block = 1024
        player_ref = weakref.ref(self)

        def callback(outdata, frames, time_info, status):  # noqa: ANN001
            player = player_ref()
            if player is None or not player._playing:
                outdata.fill(0)
                return
            data, done = player._read_frames(frames)
            buf = np.frombuffer(data, dtype=np.uint8)
            n = len(buf) // fps_bytes
            out = np.frombuffer(outdata, dtype=np.uint8)
            out[: n * fps_bytes] = buf[: n * fps_bytes]
            if n < frames:
                out[n * fps_bytes:] = 0
            if done:
                QTimer.singleShot(0, player._notify_finished)

        try:
            stream = self._sd.OutputStream(
                samplerate=self._fmt.sample_rate, channels=self._fmt.channels,
                dtype=self._fmt.dtype, blocksize=block, callback=callback,
                device=self._device_index)
            stream.start()
            self._sd_stream = stream
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Could not open the audio output: {exc}")
            self._playing = False
            self._set_state("stopped")

    def _stop_sd(self):
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._sd_stream = None

    # ------------------------------------------------------------------
    # Qt Multimedia backend
    # ------------------------------------------------------------------
    def _start_qt(self):
        try:
            from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Qt Multimedia unavailable: {exc}")
            self._playing = False
            return
        self._stop_qt()
        probe = QAudioFormat()
        probe.setSampleRate(int(self._fmt.sample_rate))
        probe.setChannelCount(int(self._fmt.channels))
        probe.setSampleFormat(self._qt_sample_format(self._fmt.dtype))
        try:
            device = QMediaDevices.defaultAudioOutput()
            sink = QAudioSink(device, probe, self)
            sink.setBufferSize(max(8192, int(self._fmt.sample_rate * 0.2)
                                   * self._fmt.bytes_per_frame))
            sink.stateChanged.connect(self._qt_state_changed)
            source = _QtSource(self)
            if not source.open(QIODevice.OpenModeFlag.ReadOnly):
                self.playbackError.emit("Could not open the playback stream.")
                self._playing = False
                return
            sink.start(source)
            self._qt_sink = sink
            self._qt_source = source
        except Exception as exc:  # noqa: BLE001
            self.playbackError.emit(f"Could not start Qt audio output: {exc}")
            self._playing = False
            self._set_state("stopped")

    def _qt_state_changed(self, state):
        try:
            from PySide6.QtMultimedia import QAudio
            if state == QAudio.State.StoppedState and self._playing:
                if self._qt_sink is not None and self._qt_sink.error() != QAudio.Error.NoError:
                    self.playbackError.emit(
                        f"Qt audio device error: {self._qt_sink.error()}")
                self._playing = False
                self._notify_finished()
        except Exception:  # noqa: BLE001
            pass

    def _stop_qt(self):
        try:
            if self._qt_sink is not None:
                self._qt_sink.stop()
                self._qt_sink.deleteLater()
        except Exception:  # noqa: BLE001
            pass
        self._qt_sink = None
        if self._qt_source is not None:
            try:
                self._qt_source.close()
            except Exception:  # noqa: BLE001
                pass
        self._qt_source = None

    # ------------------------------------------------------------------
    # diagnostics / fallbacks
    # ------------------------------------------------------------------
    def play_test_tone(self) -> bool:
        """Play a 1 s tone through the current backend; returns False on failure."""
        if not self.available:
            return False
        tone = test_tone(self._fmt)
        self._tone = tone
        self.set_buffer(tone, keep_position=False, fade_in=False)
        self.play(0.0)
        return True

    def open_in_system_player(self, path: str) -> bool:
        """Hand a rendered file to the OS.  Used when no backend exists."""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            return True
        except Exception:  # noqa: BLE001
            return False

    def prepare(self, x: np.ndarray, source_rate: int, *,
                gain_db: float = 0.0) -> PlaybackBuffer:
        """Convert audio for this device (safe to call from a worker thread)."""
        return prepare_playback(x, source_rate, self._fmt, gain_db=gain_db)


class _QtSource(QIODevice):
    """Pull-mode source: returns raw bytes straight from the prepared buffer."""

    def __init__(self, player: AudioPlayer):
        super().__init__()
        self._player = player

    def readData(self, maxlen: int) -> bytes:  # noqa: N802
        data, done = self._player._read_frames(
            self._player.device_format.bytes_to_frames(maxlen))
        if done:
            QTimer.singleShot(0, self._player._notify_finished)
        return data

    def bytesAvailable(self) -> int:  # noqa: N802
        buf = self._player._buffer
        if buf is None:
            return 0
        return max(0, len(buf.data) - self._player._byte_pos)

    def writeData(self, data) -> int:  # noqa: N802
        return 0
