"""Main window: import, audition, adjust, export."""

from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSplitter,
                               QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, APP_SUBTITLE, __version__
from ..dsp.audio import AudioData, AudioError, IMPORT_FILTER, load_audio
from ..dsp.graph import Profile, RenderResult
from ..dsp.meter import (Stats, integrated_loudness, measure, rms_db, db,
                         true_peak)
from .export_dialog import ExportDialog
from .panel import ProcessingPanel
from .player import AudioPlayer
from .theme import COLORS
from .widgets import LevelMeter, WaveformWidget
from .workers import ExportWorker, RenderWorker

PREVIEW_WINDOW_S = 30.0
LONG_FILE_THRESHOLD_S = 45.0


def _chip(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Chip")
    return lab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} \u2014 {APP_SUBTITLE}")
        self.resize(1360, 880)
        self.setAcceptDrops(True)

        self.audio: AudioData | None = None
        self._stats_in: Stats | None = None
        self._stats_out: Stats | None = None
        self._proc: np.ndarray | None = None
        self._orig_clip: np.ndarray | None = None
        self._result: RenderResult | None = None
        self._region = (0, 0)
        self._cache: dict[str, RenderResult] = {}
        self._worker: RenderWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._pending_play = False
        self._rerun = False
        self._ab = "processed"
        self._ab_match = True
        self._gain_proc = 0.0
        self._gain_orig = 0.0
        self._whole_file_preview = False
        self._busy_reason = ""
        self._last_dir = os.path.expanduser("~")

        self.player = AudioPlayer(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(420)
        self._debounce.timeout.connect(lambda: self._start_render(play_after=False))

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(60)
        self._meter_timer.timeout.connect(self._tick_meters)

        self._build_ui()
        self._load_startup_profile()
        self._build_menus()
        self._connect()
        self._set_idle_ui()
        self.statusBar().showMessage("Ready \u2014 import a file to begin.", 8000)

    def _load_startup_profile(self):
        """Start from the shipped profile so a plain import already sounds right."""
        from ..dsp.graph import default_profile_path
        path = default_profile_path()
        if not path:
            return
        try:
            prof = Profile.load(path)
        except Exception:  # noqa: BLE001 - a broken profile must never block startup
            return
        if prof.validate():
            return
        self.panel.set_profile(prof, 44100)

    # ==================================================================
    # layout
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(10)

        outer.addWidget(self._build_topbar())

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        self.panel = ProcessingPanel()
        self.panel.setMinimumWidth(470)
        self.panel.setMaximumWidth(620)
        split.addWidget(self.panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([880, 520])
        outer.addWidget(split, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setFixedHeight(6)
        self._progress.setVisible(False)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("Ghost")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_jobs)
        self.statusBar().addPermanentWidget(self._progress, 1)
        self.statusBar().addPermanentWidget(self._cancel_btn)
        self.statusBar().setSizeGripEnabled(True)

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Panel")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        lay.addWidget(title)
        sub = QLabel(APP_SUBTITLE)
        sub.setObjectName("Faint")
        lay.addWidget(sub)
        lay.addSpacing(12)

        self.import_btn = QPushButton("Import audio\u2026")
        self.import_btn.setObjectName("Ghost")
        self.import_btn.setToolTip("Open WAV, FLAC, OGG or MP3 (Ctrl+O)")
        lay.addWidget(self.import_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("Subtle")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self.file_label, 1)

        self.chip_format = _chip("\u2014")
        self.chip_rate = _chip("\u2014")
        self.chip_len = _chip("\u2014")
        self.chip_ch = _chip("\u2014")
        for c in (self.chip_format, self.chip_rate, self.chip_ch, self.chip_len):
            lay.addWidget(c)

        self.export_btn = QPushButton("Export\u2026")
        self.export_btn.setObjectName("Primary")
        self.export_btn.setToolTip("Render the whole file and save it (Ctrl+E)")
        lay.addWidget(self.export_btn)
        return bar

    def _build_left(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        wave_card = QFrame()
        wave_card.setObjectName("Panel")
        wl = QVBoxLayout(wave_card)
        wl.setContentsMargins(10, 8, 10, 8)
        wl.setSpacing(6)

        tools = QHBoxLayout()
        tools.setSpacing(6)
        lab = QLabel("WAVEFORM")
        lab.setObjectName("SectionTitle")
        tools.addWidget(lab)
        tools.addStretch(1)
        self.show_input_cb = QCheckBox("Show input")
        self.show_input_cb.setChecked(True)
        tools.addWidget(self.show_input_cb)
        self.whole_cb = QCheckBox("Preview whole file")
        self.whole_cb.setToolTip("By default long files preview a 30 s window around the "
                                 "playhead so edits stay responsive.")
        tools.addWidget(self.whole_cb)
        btn_in = QToolButton()
        btn_in.setText("+")
        btn_in.setToolTip("Zoom in")
        btn_out = QToolButton()
        btn_out.setText("\u2013")
        btn_out.setToolTip("Zoom out")
        btn_fit = QToolButton()
        btn_fit.setText("Fit")
        btn_fit.setToolTip("Show the whole file")
        for b in (btn_in, btn_out, btn_fit):
            b.setFixedHeight(22)
            tools.addWidget(b)
        wl.addLayout(tools)

        self.wave = WaveformWidget()
        wl.addWidget(self.wave, 1)

        transport = QHBoxLayout()
        transport.setSpacing(6)
        self.play_btn = QPushButton("\u25b6  Play")
        self.play_btn.setFixedWidth(92)
        self.stop_btn = QPushButton("\u25a0")
        self.stop_btn.setFixedWidth(34)
        self.stop_btn.setToolTip("Stop")
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)

        self.ab_a = QPushButton("A \u00b7 Input")
        self.ab_b = QPushButton("B \u00b7 Processed")
        for b in (self.ab_a, self.ab_b):
            b.setCheckable(True)
            b.setObjectName("Toggle")
            b.setFixedWidth(96)
        self.ab_b.setChecked(True)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.ab_a)
        grp.addButton(self.ab_b)
        self._ab_group = grp
        transport.addWidget(self.ab_a)
        transport.addWidget(self.ab_b)

        self.match_cb = QCheckBox("Level-match A/B")
        self.match_cb.setChecked(True)
        self.match_cb.setToolTip("Attenuate the louder side so the comparison is fair. "
                                 "Attenuation only, never a boost.")
        transport.addWidget(self.match_cb)
        transport.addStretch(1)
        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setObjectName("Mono")
        transport.addWidget(self.time_label)
        wl.addLayout(transport)

        self.region_label = QLabel("")
        self.region_label.setObjectName("Faint")
        wl.addWidget(self.region_label)
        lay.addWidget(wave_card, 3)

        meter_card = QFrame()
        meter_card.setObjectName("Panel")
        ml = QHBoxLayout(meter_card)
        ml.setContentsMargins(12, 8, 12, 10)
        ml.setSpacing(16)

        meter_box = QVBoxLayout()
        meter_box.setSpacing(4)
        mtitle = QLabel("LEVELS")
        mtitle.setObjectName("SectionTitle")
        meter_box.addWidget(mtitle)
        self.meter = LevelMeter(2)
        meter_box.addWidget(self.meter)
        self.meter_note = QLabel("")
        self.meter_note.setObjectName("Faint")
        meter_box.addWidget(self.meter_note)
        ml.addLayout(meter_box, 1)

        grid_wrap = QFrame()
        grid_wrap.setObjectName("Card")
        gl = QGridLayout(grid_wrap)
        gl.setContentsMargins(12, 8, 12, 8)
        gl.setHorizontalSpacing(14)
        gl.setVerticalSpacing(2)
        for col, text in enumerate(("", "INPUT", "PROCESSED", "\u0394")):
            h = QLabel(text)
            h.setObjectName("SectionTitle")
            gl.addWidget(h, 0, col)
        self._stat_labels: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        rows = [("peak", "True peak"), ("rms", "RMS"), ("lufs", "Loudness"),
                ("crest", "Crest"), ("corr", "Stereo corr")]
        for i, (key, label) in enumerate(rows, start=1):
            gl.addWidget(QLabel(label), i, 0)
            a = QLabel("\u2014")
            b = QLabel("\u2014")
            d = QLabel("\u2014")
            for w in (a, b, d):
                w.setObjectName("Mono")
            gl.addWidget(a, i, 1)
            gl.addWidget(b, i, 2)
            gl.addWidget(d, i, 3)
            self._stat_labels[key] = (a, b, d)
        ml.addWidget(grid_wrap, 1)
        lay.addWidget(meter_card, 1)

        self._wire_wave_tools(btn_in, btn_out, btn_fit)
        return wrap

    def _wire_wave_tools(self, zin, zout, fit):
        zin.clicked.connect(lambda: self.wave.zoom(1.6))
        zout.clicked.connect(lambda: self.wave.zoom(1 / 1.6))
        fit.clicked.connect(self.wave.fit)
        self.show_input_cb.toggled.connect(self.wave.show_input)
        self.whole_cb.toggled.connect(self._on_whole_toggle)

    def _build_menus(self):
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a_imp = QAction("&Import audio\u2026", self)
        a_imp.setShortcut(QKeySequence.Open)
        a_imp.triggered.connect(self.import_dialog)
        a_exp = QAction("&Export\u2026", self)
        a_exp.setShortcut("Ctrl+E")
        a_exp.triggered.connect(self.export_dialog)
        a_quit = QAction("&Quit", self)
        a_quit.setShortcut(QKeySequence.Quit)
        a_quit.triggered.connect(self.close)
        for a in (a_imp, a_exp):
            m_file.addAction(a)
        m_file.addSeparator()
        m_file.addAction(a_quit)

        m_edit = mb.addMenu("&Processing")
        a_reset = QAction("&Reset profile", self)
        a_reset.setShortcut("Ctrl+R")
        a_reset.triggered.connect(lambda: self.panel._reset_profile())
        a_bypass = QAction("&Bypass (A/B render)", self)
        a_bypass.setShortcut("B")
        a_bypass.setCheckable(True)
        a_bypass.triggered.connect(lambda chk: self.panel.bypass_btn.setChecked(chk))
        m_edit.addAction(a_reset)
        m_edit.addAction(a_bypass)

        m_view = mb.addMenu("&View")
        a_fit = QAction("&Fit waveform", self)
        a_fit.setShortcut("F")
        a_fit.triggered.connect(self.wave.fit)
        a_clear = QAction("Clear &selection", self)
        a_clear.setShortcut("Esc")
        a_clear.triggered.connect(lambda: self.wave.set_selection(None))
        m_view.addAction(a_fit)
        m_view.addAction(a_clear)

        m_tools = mb.addMenu("&Analysis")
        a_pair = QAction("&Reference pair analysis\u2026", self)
        a_pair.triggered.connect(self.open_analysis)
        a_about = QAction("&Guide / about\u2026", self)
        a_about.triggered.connect(self.show_about)
        m_tools.addAction(a_pair)
        m_tools.addSeparator()
        m_tools.addAction(a_about)

    def _connect(self):
        self.import_btn.clicked.connect(self.import_dialog)
        self.export_btn.clicked.connect(self.export_dialog)
        self.play_btn.clicked.connect(self._on_play_pause)
        self.stop_btn.clicked.connect(self._on_stop)
        self.ab_a.clicked.connect(lambda: self._set_ab("original"))
        self.ab_b.clicked.connect(lambda: self._set_ab("processed"))
        self.match_cb.toggled.connect(self._on_match_toggle)
        self.wave.seekRequested.connect(self._on_seek)
        self.wave.selectionChanged.connect(self._on_selection)
        self.panel.changed.connect(self._on_profile_changed)
        self.panel.profileReplaced.connect(self._on_profile_replaced)
        self.player.positionChanged.connect(self._on_playback_position)
        self.player.stateChanged.connect(self._on_player_state)
        self.player.playbackError.connect(self._on_player_error)
        self.player.finished.connect(self._on_player_finished)

    # ==================================================================
    # import
    # ==================================================================
    def import_dialog(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import audio", self._last_dir,
                                                  IMPORT_FILTER)
        if not path:
            return
        self.load_file(path)

    def load_file(self, path: str):
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Decoding {os.path.basename(path)}\u2026")
        QApplication.processEvents()
        try:
            audio = load_audio(path)
        except AudioError as exc:
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not import", str(exc))
            self.statusBar().showMessage("Import failed.", 6000)
            return
        except Exception as exc:  # noqa: BLE001
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Unexpected import error",
                                 f"{exc}\n\n{traceback.format_exc(limit=2)}")
            return
        finally:
            try:
                QGuiApplication.restoreOverrideCursor()
            except Exception:  # noqa: BLE001
                pass

        self._stop_all()
        self.audio = audio
        self._last_dir = os.path.dirname(audio.path or "")
        self._cache.clear()
        self._proc = None
        self._result = None
        self._stats_out = None
        self._ab = "processed"
        self.ab_b.setChecked(True)
        self.wave.set_audio(audio.samples, None, audio.sample_rate)
        self.panel.set_sample_rate(audio.sample_rate)
        self._update_file_info()
        for w in (self.export_btn, self.play_btn, self.ab_a, self.ab_b):
            w.setEnabled(True)

        self.statusBar().showMessage("Measuring input\u2026")
        QApplication.processEvents()
        try:
            self._stats_in = measure(audio.samples, audio.sample_rate)
        except Exception:  # noqa: BLE001
            self._stats_in = None
            traceback.print_exc()
        self._refresh_stats()

        if audio.warnings:
            QMessageBox.information(self, "Imported with notes", "\n\n".join(audio.warnings))
        self._start_render(play_after=False)
        self.statusBar().showMessage(
            f"Loaded {audio.filename} \u00b7 {audio.duration:.2f} s \u00b7 "
            f"{audio.sample_rate} Hz \u00b7 {audio.channels} ch", 8000)

    def _update_file_info(self):
        a = self.audio
        if a is None:
            self.file_label.setText("No file loaded")
            for c in (self.chip_format, self.chip_rate, self.chip_ch, self.chip_len):
                c.setText("\u2014")
            return
        self.file_label.setText(a.filename)
        self.file_label.setToolTip(a.path or "")
        self.chip_format.setText((a.format or "?").upper())
        self.chip_rate.setText(f"{a.sample_rate / 1000:.1f} kHz")
        self.chip_ch.setText("stereo" if a.channels >= 2 else "mono")
        mm, ss = divmod(int(round(a.duration)), 60)
        self.chip_len.setText(f"{mm}:{ss:02d}")
        self.time_label.setText(f"0:00.00 / {self._fmt(a.duration)}")

    # ==================================================================
    # preview region and rendering
    # ==================================================================
    def _preview_region(self) -> tuple[int, int]:
        if self.audio is None:
            return (0, 0)
        sr = self.audio.sample_rate
        n = self.audio.frames
        sel = self.wave.selection()
        if sel is not None:
            s = int(sel[0] * sr)
            e = min(n, int(sel[1] * sr))
            if e - s > 0:
                return (s, e)
        if self._whole_file_preview or self.audio.duration <= LONG_FILE_THRESHOLD_S:
            return (0, n)
        start = self.wave.playhead() - 5.0
        start = max(0.0, min(start, self.audio.duration - PREVIEW_WINDOW_S))
        s = int(start * sr)
        e = min(n, s + int(PREVIEW_WINDOW_S * sr))
        return (s, e)

    def _preview_target_loudness(self) -> float | None:
        """Loudness target for a preview render.

        Uses the whole input's loudness, exactly like the export does, so a
        preview window is not level-matched to itself and the A/B is comparable
        with what will be written.
        """
        if self._stats_in is None or not self.panel.profile().loudness_match:
            return None
        mode = self.panel.profile().loudness_match_mode
        return self._stats_in.rms_db if mode == "rms" else self._stats_in.lufs

    def _cache_key(self, region: tuple[int, int]) -> str:
        prof = json.dumps(self.panel.profile().to_dict(), sort_keys=True)
        return (f"{hash(prof)}:{region[0]}:{region[1]}:"
                f"{self.audio.sample_rate if self.audio else 0}:"
                f"{self._preview_target_loudness()}")

    def _on_profile_changed(self):
        if self.audio is None:
            return
        self._debounce.start()

    def _on_profile_replaced(self):
        self._cache.clear()
        if self.audio is None:
            return
        self._start_render(play_after=False)

    def _on_whole_toggle(self, on: bool):
        self._whole_file_preview = bool(on)
        if self.audio is not None:
            self._start_render(play_after=False)

    def _start_render(self, play_after: bool = False, force: bool = False):
        if self.audio is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._rerun = True
            self._worker.cancel()
            self._pending_play = self._pending_play or play_after
            return
        region = self._preview_region()
        if region[1] <= region[0]:
            return
        key = self._cache_key(region)
        if not force and key in self._cache:
            self._apply_result(self._cache[key])
            if play_after:
                self._begin_playback()
            return

        self._region = region
        self._orig_clip = self.audio.samples[region[0]:region[1]]
        prof = self.panel.profile()
        worker = RenderWorker(self.audio.samples, self.audio.sample_rate, prof,
                              region[0], region[1], self,
                              target_loudness=self._preview_target_loudness())
        worker.progress.connect(self._on_render_progress)
        worker.done.connect(self._on_render_done)
        worker.failed.connect(self._on_render_failed)
        worker.cancelled.connect(self._on_render_cancelled)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._pending_play = play_after
        self._busy_reason = "render"
        self._set_busy(True, "Rendering preview\u2026")
        worker.start()

    def _on_render_progress(self, frac: float, label: str):
        self._progress.setValue(int(frac * 1000))
        self.statusBar().showMessage(f"{label}\u2026", 2000)

    def _on_render_done(self, result: RenderResult):
        key = self._cache_key((getattr(result, "start_frame", 0),
                               getattr(result, "start_frame", 0) + result.frames))
        self._cache[key] = result
        if len(self._cache) > 8:
            for k in list(self._cache)[:-8]:
                self._cache.pop(k, None)
        self._apply_result(result)
        if self._pending_play:
            self._pending_play = False
            self._begin_playback()

    def _apply_result(self, result: RenderResult):
        start = getattr(result, "start_frame", 0)
        self._result = result
        self._region = (start, start + result.frames)
        self._proc = result.samples
        self.wave.set_processed(result.samples)
        try:
            self._stats_out = measure(result.samples, result.sample_rate)
        except Exception:  # noqa: BLE001
            self._stats_out = None
        self._update_gain_match()
        self._refresh_stats()
        self._refresh_region_label()
        if result.warnings:
            self.meter_note.setText(result.warnings[-1])
        else:
            self.meter_note.setText("")
        self._set_busy(False)
        if self._ab == "processed":
            self.player.set_buffer(self._proc_for_player(), self.audio.sample_rate
                                   if self.audio else 44100)

    def _on_render_failed(self, msg: str):
        self._set_busy(False)
        self._pending_play = False
        QMessageBox.warning(self, "Render failed", msg)
        self.statusBar().showMessage("Render failed.", 8000)

    def _on_render_cancelled(self):
        self._set_busy(False)

    def _on_worker_finished(self):
        self._worker = None
        if self._rerun:
            self._rerun = False
            self._start_render(play_after=self._pending_play)

    def _set_busy(self, busy: bool, text: str = ""):
        self._progress.setVisible(busy)
        self._cancel_btn.setVisible(busy)
        if busy:
            self._progress.setValue(0)
            if text:
                self.statusBar().showMessage(text)

    def _cancel_jobs(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.statusBar().showMessage("Cancelling\u2026", 3000)
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.cancel()

    # ==================================================================
    # playback
    # ==================================================================
    def _proc_for_player(self) -> np.ndarray | None:
        if self._proc is None:
            return None
        if self._ab_match and abs(self._gain_proc) > 0.01:
            return self._proc * (10 ** (self._gain_proc / 20.0))
        return self._proc

    def _orig_for_player(self) -> np.ndarray | None:
        if self._orig_clip is None:
            return None
        if self._ab_match and abs(self._gain_orig) > 0.01:
            return self._orig_clip * (10 ** (self._gain_orig / 20.0))
        return self._orig_clip

    def _update_gain_match(self):
        self._gain_proc = 0.0
        self._gain_orig = 0.0
        if not (self._ab_match and self._proc is not None and self._orig_clip is not None):
            return
        try:
            lo = integrated_loudness(self._orig_clip, self.audio.sample_rate)
            lp = integrated_loudness(self._proc, self.audio.sample_rate)
        except Exception:  # noqa: BLE001
            return
        d = lo - lp
        if d < 0:
            self._gain_proc = d          # processed is louder -> attenuate it
        else:
            self._gain_orig = -d         # input is louder -> attenuate it

    def _set_ab(self, mode: str):
        if mode == self._ab:
            return
        self._ab = mode
        (self.ab_a if mode == "original" else self.ab_b).setChecked(True)
        if self.audio is None:
            return
        if mode == "original":
            if self._orig_clip is None:
                region = self._preview_region()
                self._region = region
                self._orig_clip = self.audio.samples[region[0]:region[1]]
            self.player.set_buffer(self._orig_for_player(), self.audio.sample_rate)
        else:
            if self._proc is None:
                self._start_render(play_after=self.player.is_playing())
                return
            self.player.set_buffer(self._proc_for_player(), self.audio.sample_rate)
        self._refresh_stats()

    def _on_match_toggle(self, on: bool):
        self._ab_match = bool(on)
        self._update_gain_match()
        if self.audio is not None:
            buf = self._proc_for_player() if self._ab == "processed" else self._orig_for_player()
            if buf is not None:
                self.player.set_buffer(buf, self.audio.sample_rate)
        self._refresh_stats()

    def _on_play_pause(self):
        if self.audio is None:
            return
        if self.player.is_playing():
            self.player.pause()
            return
        self._begin_playback()

    def _begin_playback(self):
        if self.audio is None:
            return
        if self._ab == "processed" and self._proc is None:
            self._start_render(play_after=True)
            return
        buf = self._proc_for_player() if self._ab == "processed" else self._orig_for_player()
        if buf is None:
            return
        if not self.player.available:
            QMessageBox.information(
                self, "No audio output",
                "No playback device was found in this environment, so audio monitoring is "
                "disabled.\n\nThe processing, preview rendering and export all still work; "
                "export and listen in any player.\n\nOn Linux install libportaudio2 (or use "
                "a full PySide6 build with Qt Multimedia) to enable monitoring.")
            return
        self.player.set_buffer(buf, self.audio.sample_rate)
        offset = max(0.0, self.wave.playhead() - self._region[0] / self.audio.sample_rate)
        self.player.play(offset)
        self._meter_timer.start()

    def _on_stop(self):
        self.player.stop()
        self._meter_timer.stop()
        self.wave.set_playhead(0.0)
        self._refresh_stats()

    def _stop_all(self):
        self.player.shutdown()
        self._meter_timer.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()

    def _on_seek(self, t: float):
        self.wave.set_playhead(t)
        if self.audio is None:
            return
        region_start = self._region[0] / self.audio.sample_rate
        region_end = self._region[1] / self.audio.sample_rate
        if not (region_start <= t <= region_end) and not self._whole_file_preview:
            self._debounce.stop()
            self._start_render(play_after=self.player.is_playing())
        elif self.player.is_playing() or self.player.position() > 0:
            self.player.seek(max(0.0, t - region_start))

    def _on_selection(self, sel):
        self._refresh_region_label()
        if self.audio is not None:
            self._debounce.start()

    def _on_playback_position(self, pos: float):
        if self.audio is None:
            return
        offset = self._region[0] / self.audio.sample_rate
        self.wave.set_playhead(offset + pos)
        self.time_label.setText(f"{self._fmt(offset + pos)} / {self._fmt(self.audio.duration)}")

    def _on_player_state(self, state: str):
        if state == "playing":
            self.play_btn.setText("\u2016  Pause")
            self._meter_timer.start()
        else:
            self.play_btn.setText("\u25b6  Play")
            if state == "stopped":
                self._meter_timer.stop()

    def _on_player_error(self, msg: str):
        self.statusBar().showMessage(msg, 10000)

    def _on_player_finished(self):
        self._meter_timer.stop()
        self.wave.set_playhead(self.wave.playhead())

    def _tick_meters(self):
        if self.audio is None:
            return
        buf = self._proc_for_player() if self._ab == "processed" else self._orig_for_player()
        if buf is None or not self.player.is_playing():
            return
        sr = self.audio.sample_rate
        t = self.player.position()
        s = int(max(0, (t - 0.6) * sr))
        e = min(buf.shape[0], s + int(0.06 * sr))
        if e <= s:
            return
        self.meter.set_levels(self._channel_levels(buf[s:e]))

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = seconds - 60 * m
        return f"{m}:{s:05.2f}"

    # ==================================================================
    # stats and labels
    # ==================================================================
    def _refresh_stats(self):
        si, so = self._stats_in, self._stats_out
        def put(key, text_in, text_out, delta=""):
            a, b, d = self._stat_labels[key]
            a.setText(text_in)
            b.setText(text_out)
            d.setText(delta)
        if si is None:
            for key in self._stat_labels:
                put(key, "\u2014", "\u2014")
        else:
            put("peak", f"{si.true_peak_db:+.1f}", f"{so.true_peak_db:+.1f}" if so else "\u2014",
                f"{so.true_peak_db - si.true_peak_db:+.1f}" if so else "")
            put("rms", f"{si.rms_db:+.1f}", f"{so.rms_db:+.1f}" if so else "\u2014",
                f"{so.rms_db - si.rms_db:+.1f}" if so else "")
            put("lufs", f"{si.lufs:.1f}", f"{so.lufs:.1f}" if so else "\u2014",
                f"{so.lufs - si.lufs:+.1f}" if so else "")
            put("crest", f"{si.crest_db:.1f}", f"{so.crest_db:.1f}" if so else "\u2014",
                f"{so.crest_db - si.crest_db:+.1f}" if so else "")
            put("corr", f"{si.correlation:.2f}", f"{so.correlation:.2f}" if so else "\u2014",
                f"{so.correlation - si.correlation:+.2f}" if so else "")
        buf = None
        if self._ab == "processed" and self._proc is not None:
            buf = self._proc_for_player()
        elif self._orig_clip is not None:
            buf = self._orig_for_player()
        if buf is not None:
            self.meter.set_levels(self._channel_levels(buf))
        elif si is not None:
            self.meter.set_levels([(si.rms_db, si.peak_db, si.true_peak_db)] * 2)
        self._refresh_region_label()

    @staticmethod
    def _channel_levels(buf: np.ndarray) -> list[tuple[float, float, float]]:
        """Per-channel (RMS, peak, true-peak) in dB, cheap enough for the UI."""
        out = []
        n_ch = 2
        for c in range(n_ch):
            ch = buf[:, min(c, buf.shape[1] - 1)]
            rms = float(np.sqrt(np.mean(ch * ch))) if ch.size else 0.0
            pk = float(np.max(np.abs(ch))) if ch.size else 0.0
            out.append((float(db(rms, -60.0)), float(db(pk, -60.0)), float(db(pk, -60.0))))
        return out

    def _refresh_region_label(self):
        if self.audio is None:
            self.region_label.setText("")
            return
        sr = self.audio.sample_rate
        s, e = self._region
        sel = self.wave.selection()
        txt = (f"Previewing {self._fmt(s / sr)} \u2013 {self._fmt(e / sr)} "
               f"({(e - s) / sr:.1f} s)")
        if sel:
            txt += "  \u00b7 selection"
        elif e - s < self.audio.frames:
            txt += "  \u00b7 30 s window around the playhead for responsiveness"
        else:
            txt += "  \u00b7 whole file"
        stages = self.panel.profile().active_stages()
        txt += "  \u00b7  stages: " + (", ".join(stages) if stages else "none (bypass)")
        if self._gain_proc < -0.01:
            txt += f"  \u00b7  A/B matched: processed {-self._gain_proc:.1f} dB down"
        elif self._gain_orig < -0.01:
            txt += f"  \u00b7  A/B matched: input {-self._gain_orig:.1f} dB down"
        self.region_label.setText(txt)

    # ==================================================================
    # export
    # ==================================================================
    def export_dialog(self):
        if self.audio is None:
            QMessageBox.information(self, "Nothing to export", "Import an audio file first.")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy",
                                    "A preview render is still running. Wait for it to finish "
                                    "or cancel it first.")
            return
        dlg = ExportDialog(self.audio, self._last_dir, self)
        if dlg.exec() != ExportDialog.Accepted:
            return
        settings = dlg.settings()
        self.player.pause()
        worker = ExportWorker(self.audio, self.panel.profile(), settings, self)
        worker.progress.connect(self._on_export_progress)
        worker.done.connect(self._on_export_done)
        worker.failed.connect(self._on_export_failed)
        worker.cancelled.connect(self._on_export_cancelled)
        worker.finished.connect(lambda: setattr(self, "_export_worker", None))
        self._export_worker = worker
        self._busy_reason = "export"
        self._set_busy(True, "Exporting\u2026")
        self.statusBar().showMessage("Exporting\u2026")
        worker.start()

    def _on_export_progress(self, frac: float, label: str):
        self._progress.setValue(int(frac * 1000))
        self.statusBar().showMessage(f"Export: {label}\u2026", 2000)

    def _on_export_cancelled(self):
        self._set_busy(False)
        self.statusBar().showMessage("Export cancelled.", 6000)

    def _on_export_failed(self, msg: str):
        self._set_busy(False)
        QMessageBox.warning(self, "Export failed", msg)
        self.statusBar().showMessage("Export failed.", 8000)

    def _on_export_done(self, facts: dict):
        self._set_busy(False)
        path = facts["path"]
        self._last_dir = os.path.dirname(path)
        report = self._verify_export(path, facts)
        self.statusBar().showMessage(f"Exported to {path}", 10000)
        box = QMessageBox(self)
        box.setWindowTitle("Export complete")
        box.setIcon(QMessageBox.Information)
        box.setText(f"Saved {os.path.basename(path)}")
        box.setDetailedText(report)
        box.setInformativeText(self._export_headline(path, facts))
        box.exec()

    def _export_headline(self, path: str, facts: dict) -> str:
        size = facts["bytes"] / (1024 * 1024)
        return (f"{facts['format']} \u00b7 {facts['sample_rate']} Hz \u00b7 "
                f"{facts['duration']:.2f} s \u00b7 {size:.1f} MB")

    def _verify_export(self, path: str, facts: dict) -> str:
        """Decode the written file and check the invariants on the encoded result."""
        lines = [f"File: {path}", ""]
        try:
            out = load_audio(path)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Could not re-read the exported file: {exc}")
            return "\n".join(lines)
        sr = self.audio.sample_rate if self.audio else out.sample_rate
        try:
            st_out = measure(out.samples, out.sample_rate)
            st_in = self._stats_in or measure(self.audio.samples, sr)
        except Exception:  # noqa: BLE001
            st_out = st_in = None

        dur_in = self.audio.duration if self.audio else 0.0
        lines.append("Timing")
        lines.append(f"  input duration      {dur_in:.6f} s")
        lines.append(f"  exported duration   {out.duration:.6f} s "
                     f"({out.frames} frames @ {out.sample_rate} Hz)")
        lines.append(f"  difference          {out.duration - dur_in:+.6f} s")
        lines.append("  No time-stretch, pitch shift or varispeed is used anywhere in the "
                     "graph; the render length equals the input length exactly before "
                     "container padding.")
        if st_out and st_in:
            lines += [
                "",
                "Encoded measurement",
                f"  sample peak         {st_out.peak_db:+.2f} dBFS",
                f"  true peak           {st_out.true_peak_db:+.2f} dBTP "
                f"(ceiling {self.panel.profile().limiter.ceiling_db:+.1f})",
                f"  integrated loudness {st_out.lufs:.2f} LUFS "
                f"(input {st_in.lufs:.2f}, \u0394 {st_out.lufs - st_in.lufs:+.2f})",
                f"  RMS                 {st_out.rms_db:+.2f} dBFS",
                f"  crest factor        {st_out.crest_db:.2f} dB",
                f"  stereo correlation  {st_out.correlation:.3f}",
                f"  samples at full scale {st_out.clipped_fraction * 100:.4f} %",
            ]
            if st_out.true_peak_db > self.panel.profile().limiter.ceiling_db + 0.1:
                lines.append("  WARNING: encoded true peak exceeds the requested ceiling "
                             "(lossy codec overshoot). Lower the ceiling and re-export.")
            if st_out.lufs > st_in.lufs + 0.5:
                lines.append("  WARNING: the export is louder than the input.")
        if facts.get("render", {}).get("warnings"):
            lines += ["", "Notes"] + [f"  {w}" for w in facts["render"]["warnings"]]
        return "\n".join(lines)

    # ==================================================================
    # analysis
    # ==================================================================
    def open_analysis(self):
        from .analysis_dialog import AnalysisDialog
        dlg = AnalysisDialog(self)
        dlg.profileCreated.connect(self._use_profile_from_analysis)
        dlg.exec()

    def _use_profile_from_analysis(self, profile: Profile, path: str):
        self.panel.set_profile(profile, self.audio.sample_rate if self.audio else 44100)
        self._cache.clear()
        self.statusBar().showMessage(f"Loaded profile '{profile.name}' from analysis "
                                     f"({os.path.basename(path)})", 8000)
        if self.audio is not None:
            self._start_render(play_after=False)

    def show_about(self):
        QMessageBox.information(
            self, f"About {APP_NAME}",
            f"{APP_NAME} {__version__}\n{APP_SUBTITLE}\n\n"
            "Imports an audio file, renders it through one documented processing graph and "
            "exports it at the original pitch, tempo and speed.\n\n"
            "Rules the app enforces:\n"
            "\u2022 no pitch shift, tempo change, time-stretch or varispeed\n"
            "\u2022 no loudness maximisation: the render is matched to the input's loudness, "
            "never above it, unless you raise Output trim yourself\n"
            "\u2022 -1 dBTP true-peak ceiling checked on the encoded export\n"
            "\u2022 stereo width touches only the side signal, so mono fold-down is preserved\n"
            "\u2022 original files are never overwritten\n\n"
            "Preview and export use the same engine (timbreforge.dsp.graph.render).")

    # ==================================================================
    # events
    # ==================================================================
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in (".wav", ".flac", ".ogg", ".oga", ".mp3", ".aiff", ".aif"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self.load_file(path)
                return

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_Space and not isinstance(self.focusWidget(), QLineEdit):
            self._on_play_pause()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.cancel()
            self._export_worker.wait(3000)
        self.player.shutdown()
        super().closeEvent(event)

    def _set_idle_ui(self):
        self.export_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.ab_a.setEnabled(False)
        self.ab_b.setEnabled(False)
        self.meter.set_levels([(-60, -60, -60)] * 2)
