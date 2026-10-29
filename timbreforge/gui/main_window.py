"""Main window: import, audition, adjust, export.

Rules the window enforces
-------------------------
* nothing expensive runs on the UI thread: decoding, measuring, rendering,
  export and the audition fallback all happen in :mod:`timbreforge.gui.workers`;
* a parameter change is debounced and an older render is cancelled rather than
  queued, so dragging a slider cannot build up a backlog of work;
* preview and export call the same ``dsp.graph.render`` - there is no second
  "preview-only" code path that could sound different from the export;
* playback is optional.  If no audio backend exists, the window offers to render
  the preview region and hand it to the system's own player instead.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QSplitter, QToolButton, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, APP_SUBTITLE, __version__
from ..dsp.audio import (IMPORT_FILTER, AudioData, AudioError, format_for_extension,
                         load_audio)
from ..dsp.audio_limits import check_export, check_import, check_loaded, describe_limits
from ..dsp.graph import Profile
from ..dsp.meter import Stats, measure
from .export_dialog import ExportDialog
from .panel import ProcessingPanel
from .player import AudioPlayer
from .widgets import LevelMeter, WaveformWidget
from .workers import (AuditionWorker, ExportWorker, ImportWorker, RenderPayload,
                      RenderWorker)

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
        self._payload: RenderPayload | None = None
        self._proc: np.ndarray | None = None
        self._orig_clip: np.ndarray | None = None
        self._play_array: np.ndarray | None = None
        self._play_array_buf = None
        self._region = (0, 0)
        self._cache: dict[str, RenderPayload] = {}
        self._worker: RenderWorker | None = None
        self._import_worker: ImportWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._audition_worker: AuditionWorker | None = None
        self._audition_tmp: str | None = None
        self._pending_play = False
        self._rerun = False
        # gain applied on top of the rendered buffer while a trim change is still
        # being rendered; always returns to 0 when that render lands
        self._monitor_trim_db = 0.0
        self._ab = "processed"
        self._whole_file_preview = False
        self._busy_reason = ""
        self._last_dir = os.path.expanduser("~")

        self.player = AudioPlayer(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(lambda: self._start_render(play_after=False))

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(50)
        self._meter_timer.timeout.connect(self._tick_meters)

        self._build_ui()
        self._load_startup_profile()
        self._build_menus()
        self._connect()
        self._set_idle_ui()
        self.statusBar().showMessage(
            f"Ready \u2014 audio output: {self.player.backend_name}", 12000)

    # ==================================================================
    # startup
    # ==================================================================
    def _load_startup_profile(self):
        """Load a profile only when one is named explicitly.

        The default start is the **simple** view: five controls that generate a
        profile.  A full profile is loaded when ``TIMBREFORGE_PROFILE`` points at
        one, which is what the CLI and the tests use.
        """
        path = os.environ.get("TIMBREFORGE_PROFILE") or None
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
        self.panel.setMaximumWidth(640)
        split.addWidget(self.panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([860, 520])
        outer.addWidget(split, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setFixedHeight(6)
        self._progress.setVisible(False)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("Ghost")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_jobs)
        self._audition_btn = QPushButton("Audition in player")
        self._audition_btn.setObjectName("Ghost")
        self._audition_btn.setVisible(False)
        self._audition_btn.setToolTip(
            "No audio output backend was found. Render the preview region and open it in "
            "your system's audio player.")
        self._audition_btn.clicked.connect(self._audition)
        self.statusBar().addPermanentWidget(self._progress, 1)
        self.statusBar().addPermanentWidget(self._audition_btn)
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

        self.pair_btn = QPushButton("Pitch/tempo\u2026")
        self.pair_btn.setObjectName("Ghost")
        self.pair_btn.setToolTip(
            "Work out what pitch and tempo change turned one file into another, and get "
            "the Audacity settings that reproduce it (Ctrl+T). Measurement only \u2014 "
            "this app never changes pitch or tempo.")
        lay.addWidget(self.pair_btn)
        lay.addSpacing(6)

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
        self.backend_label = QLabel(self._backend_text())
        self.backend_label.setObjectName("Faint")
        self.backend_label.setToolTip(self.player.backend_name)
        transport.addWidget(self.backend_label)
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
        self.meter_note.setWordWrap(True)
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
        a_tone = QAction("&Test audio output", self)
        a_tone.triggered.connect(self.test_output)
        a_aud = QAction("&Audition preview in system player", self)
        a_aud.setShortcut("Ctrl+Shift+P")
        a_aud.triggered.connect(self._audition)
        m_view.addAction(a_fit)
        m_view.addAction(a_clear)
        m_view.addSeparator()
        m_view.addAction(a_tone)
        m_view.addAction(a_aud)

        m_tools = mb.addMenu("&Analysis")
        a_pt = QAction("&Pitch/tempo of two files\u2026", self)
        a_pt.setShortcut("Ctrl+T")
        a_pt.setStatusTip("Recover the pitch and tempo change between two files")
        a_pt.triggered.connect(self.open_pair_tool)
        a_pair = QAction("&Reference pair analysis\u2026", self)
        a_pair.triggered.connect(self.open_analysis)
        a_about = QAction("&Guide / about\u2026", self)
        a_about.triggered.connect(self.show_about)
        m_tools.addAction(a_pt)
        m_tools.addAction(a_pair)
        m_tools.addSeparator()
        m_tools.addAction(a_about)

    def _connect(self):
        self.import_btn.clicked.connect(self.import_dialog)
        self.export_btn.clicked.connect(self.export_dialog)
        self.pair_btn.clicked.connect(self.open_pair_tool)
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
        self.player.backendChanged.connect(self._on_backend_changed)

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
        """Decode + measure in a worker thread, then render the first preview."""
        verdict = check_import(path)
        if verdict.errors:
            QMessageBox.warning(self, "Could not import", "\n".join(verdict.errors))
            return
        for w in verdict.warnings:
            self.statusBar().showMessage(w, 8000)

        if self._import_worker is not None and self._import_worker.isRunning():
            self._import_worker.cancel()
            self._import_worker.wait(2000)

        self._stop_all()
        self._pending_play = False
        worker = ImportWorker(path, self)
        worker.progress.connect(self._on_import_progress)
        worker.done.connect(self._on_import_done)
        worker.failed.connect(self._on_import_failed)
        worker.finished.connect(lambda: setattr(self, "_import_worker", None))
        self._import_worker = worker
        self._set_busy(True, f"Opening {os.path.basename(path)}\u2026")
        worker.start()

    def _on_import_progress(self, frac: float, label: str):
        self._progress.setValue(int(frac * 1000))
        self.statusBar().showMessage(f"{label}\u2026", 2000)

    def _on_import_failed(self, msg: str):
        self._set_busy(False)
        self.statusBar().showMessage("Import failed.", 8000)
        QMessageBox.warning(self, "Could not import", msg)

    def _on_import_done(self, audio: AudioData, stats: Stats):
        self._set_busy(False)
        verdict = check_loaded(audio.frames, audio.sample_rate, audio.channels)
        if verdict.errors:
            QMessageBox.warning(self, "Could not use this file", "\n".join(verdict.errors))
            return

        self.audio = audio
        self._stats_in = stats
        self._stats_out = None
        self._payload = None
        self._proc = None
        self._orig_clip = None
        self._play_array = None
        self._cache.clear()
        self._region = (0, audio.frames)
        self._last_dir = os.path.dirname(audio.path or "") or self._last_dir

        self.player.configure_source(audio.sample_rate, audio.channels)
        self.backend_label.setText(self._backend_text())
        self.backend_label.setToolTip(self.player.backend_name)
        self._audition_btn.setVisible(not self.player.available)

        self.wave.set_audio(audio.samples, None, audio.sample_rate)
        self.wave.set_playhead(0.0)
        self.panel.set_sample_rate(audio.sample_rate)
        self._update_file_info()
        for w in (self.export_btn, self.play_btn, self.ab_a, self.ab_b):
            w.setEnabled(True)
        self._refresh_stats()

        notes = list(audio.warnings) + list(verdict.warnings)
        if notes:
            QMessageBox.information(self, "Imported with notes", "\n\n".join(notes))

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
        dev = self.player.device_format
        return (f"{hash(prof)}:{region[0]}:{region[1]}:"
                f"{self.audio.sample_rate if self.audio else 0}:"
                f"{dev.sample_rate}:{dev.channels}:{dev.dtype}:"
                f"{self._preview_target_loudness()}")

    def _on_profile_changed(self):
        if self.audio is None:
            return
        # The output trim is the one control whose effect is pure gain, so show it
        # immediately through the player's read-path gain instead of waiting for
        # the render.  The render follows and takes the temporary gain back out.
        p = self._payload
        if p is not None:
            want = float(getattr(self.panel.profile(), "output_trim_db", 0.0))
            delta = want - float(p.trim_db)
            if abs(delta) > 0.05:
                self._monitor_trim_db = delta
                if self.player.is_playing():
                    self.player.set_gain_db(self._gain_for(self._ab) + delta,
                                            allow_boost=True)
        # debounce: a slider drag should produce one render, not thirty
        self._debounce.start()

    def _on_profile_replaced(self):
        self._cache.clear()
        if self.audio is None:
            return
        self._debounce.stop()
        self._start_render(play_after=False)

    def _on_whole_toggle(self, on: bool):
        self._whole_file_preview = bool(on)
        if self.audio is not None:
            self._start_render(play_after=False)

    def _start_render(self, play_after: bool = False, force: bool = False):
        if self.audio is None:
            return
        if self._worker is not None and self._worker.isRunning():
            # supersede: cancel the stale render, run once more when it exits
            self._rerun = True
            self._worker.cancel()
            self._pending_play = self._pending_play or play_after
            return
        region = self._preview_region()
        if region[1] <= region[0]:
            return
        key = self._cache_key(region)
        if not force and key in self._cache:
            self._apply_payload(self._cache[key])
            if play_after:
                self._begin_playback()
            return

        self._region = region
        self._orig_clip = self.audio.samples[region[0]:region[1]]
        prof = self.panel.profile()
        worker = RenderWorker(self.audio.samples, self.audio.sample_rate, prof,
                              region[0], region[1], self,
                              device=self.player.device_format,
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

    def _on_render_done(self, payload: RenderPayload):
        key = self._cache_key((payload.start_frame, payload.start_frame + payload.frames))
        self._cache[key] = payload
        if len(self._cache) > 6:
            for k in list(self._cache)[:-6]:
                self._cache.pop(k, None)
        self._apply_payload(payload)
        if self._pending_play:
            self._pending_play = False
            self._begin_playback()

    def _apply_payload(self, payload: RenderPayload):
        self._payload = payload
        self._monitor_trim_db = 0.0
        self._proc = payload.samples
        self._stats_out = payload.stats_out
        self._region = (payload.start_frame, payload.start_frame + payload.frames)
        if self.audio is not None:
            self._orig_clip = self.audio.samples[self._region[0]:self._region[1]]
        self.wave.set_processed(payload.samples)
        self._refresh_stats()
        if payload.warnings:
            self.meter_note.setText(payload.warnings[-1])
        else:
            self.meter_note.setText("")
        self._set_busy(False)
        if self._ab == "processed":
            self._install_buffer(keep_position=True)

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
        if self._audition_worker is not None and self._audition_worker.isRunning():
            self._audition_worker.cancel()

    # ==================================================================
    # playback
    # ==================================================================
    def _backend_text(self) -> str:
        b = self.player.backend
        if b == "sd":
            return "output: sounddevice"
        if b == "qt":
            return "output: Qt Multimedia"
        return "output: none (export only)"

    def _gain_for(self, mode: str) -> float:
        p = self._payload
        if p is None or not self.match_cb.isChecked():
            return 0.0
        return p.gain_proc_db if mode == "processed" else p.gain_orig_db

    def _install_buffer(self, *, keep_position: bool = True):
        p = self._payload
        if p is None:
            return
        buf = p.play_processed if self._ab == "processed" else p.play_original
        if buf is None:
            return
        # decoding the buffer for the meter is an 11 MB copy per render; keep the
        # last one as long as the buffer it came from is still installed
        if self._play_array_buf is not buf or self._play_array is None:
            self._play_array = buf.as_array()
            self._play_array_buf = buf
        self.player.set_buffer(buf, keep_position=keep_position,
                               gain_db=self._gain_for(self._ab) + self._monitor_trim_db)

    def _set_ab(self, mode: str):
        if mode == self._ab:
            return
        self._ab = mode
        (self.ab_a if mode == "original" else self.ab_b).setChecked(True)
        if self.audio is None or self._payload is None:
            return
        self._install_buffer(keep_position=True)
        self._refresh_stats()

    def _on_match_toggle(self, on: bool):
        if self._payload is not None:
            self.player.set_gain_db(self._gain_for(self._ab) + self._monitor_trim_db,
                                    allow_boost=self._monitor_trim_db > 0)
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
        if self._payload is None:
            self._start_render(play_after=True)
            return
        if not self.player.available:
            self._offer_audition()
            return
        self._install_buffer(keep_position=False)
        offset = max(0.0, self.wave.playhead() - self._region[0] / self.audio.sample_rate)
        self.player.play(offset)
        self._meter_timer.start()

    def _offer_audition(self):
        """No backend: offer to render the region and open it in the OS player."""
        if self._audition_btn.isVisible():
            self._audition()
            return
        box = QMessageBox(self)
        box.setWindowTitle("No audio output backend")
        box.setText("This environment has no audio output that the app can use.")
        box.setInformativeText(
            "You can still audition: the app will render the preview region to a "
            "temporary WAV file and open it in your system's audio player.\n\n"
            "Details: " + self.player.backend_name)
        box.setStandardButtons(QMessageBox.Open | QMessageBox.Close)
        box.button(QMessageBox.Open).setText("Audition in player")
        if box.exec() == QMessageBox.Open:
            self._audition()

    def _audition(self):
        if self.audio is None:
            return
        if self._payload is None:
            self._start_render(play_after=True)
            return
        if self._audition_worker is not None and self._audition_worker.isRunning():
            return
        s, e = self._region
        worker = AuditionWorker(self.audio.samples, self.audio.sample_rate,
                                self.panel.profile(), s, e, self)
        worker.progress.connect(self._on_render_progress)
        worker.done.connect(self._on_audition_ready)
        worker.failed.connect(self._on_render_failed)
        worker.finished.connect(lambda: setattr(self, "_audition_worker", None))
        self._audition_worker = worker
        self._set_busy(True, "Rendering for audition\u2026")
        worker.start()

    def _on_audition_ready(self, path: str):
        self._set_busy(False)
        self._audition_tmp = path
        if self.player.open_in_system_player(path):
            self.statusBar().showMessage(
                "Opened the rendered preview in your system audio player.", 10000)
        else:
            QMessageBox.information(
                self, "Preview rendered",
                f"The preview was rendered to:\n\n{path}\n\nOpen it with any audio player.")

    def test_output(self):
        if self.player.available and self.player.play_test_tone():
            self.statusBar().showMessage(
                f"Playing a short test tone through {self.player.backend_name}.", 8000)
        else:
            QMessageBox.information(
                self, "No audio output",
                "No playback device could be opened.\n\n" + self.player.backend_name +
                "\n\nThe processing, preview rendering and export all still work. Use "
                "\u201cAudition preview in system player\u201d (Ctrl+Shift+P) to listen.")

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
        elif self.player.buffer is not None:
            self.player.seek(max(0.0, t - region_start))

    def _on_selection(self, sel):
        self._refresh_region_label()
        if self.audio is not None:
            self._debounce.start()

    def _on_playback_position(self, pos: float):
        if self.audio is None or self.player.buffer is None:
            return
        offset = self._region[0] / self.audio.sample_rate
        if self.player.buffer.resampled:
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
        self._meter_timer.stop()
        self.statusBar().showMessage(msg, 12000)
        self._audition_btn.setVisible(True)

    def _on_backend_changed(self):
        self._audition_btn.setVisible(not self.player.available)
        if hasattr(self, "backend_label"):
            self.backend_label.setText(self._backend_text())
            self.backend_label.setToolTip(self.player.backend_name)

    def _on_player_finished(self):
        self._meter_timer.stop()

    def _tick_meters(self):
        arr = self._play_array
        buf = self.player.buffer
        if arr is None or buf is None or not self.player.is_playing():
            return
        rate = max(1, buf.format.sample_rate)
        i = int(self.player.position() * rate)
        # a short look-back: the meter should follow the music, not lag 0.6 s
        # behind it (which read as "the volume display is slow")
        s = max(0, i - int(0.20 * rate))
        e = min(arr.shape[0], i + int(0.03 * rate))
        if e <= s:
            return
        self.meter.set_levels(self._channel_levels(arr[s:e]))

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
            put("peak", f"{si.true_peak_db:+.1f}",
                f"{so.true_peak_db:+.1f}" if so else "\u2014",
                f"{so.true_peak_db - si.true_peak_db:+.1f}" if so else "")
            put("rms", f"{si.rms_db:+.1f}", f"{so.rms_db:+.1f}" if so else "\u2014",
                f"{so.rms_db - si.rms_db:+.1f}" if so else "")
            put("lufs", f"{si.lufs:.1f}", f"{so.lufs:.1f}" if so else "\u2014",
                f"{so.lufs - si.lufs:+.1f}" if so else "")
            put("crest", f"{si.crest_db:.1f}", f"{so.crest_db:.1f}" if so else "\u2014",
                f"{so.crest_db - si.crest_db:+.1f}" if so else "")
            put("corr", f"{si.correlation:.2f}", f"{so.correlation:.2f}" if so else "\u2014",
                f"{so.correlation - si.correlation:+.2f}" if so else "")

        p = self._payload
        if not self.player.is_playing():
            if p is not None and self._ab == "processed":
                self.meter.set_levels(p.levels_out)
            elif p is not None:
                self.meter.set_levels(p.levels_in)
            elif si is not None:
                self.meter.set_levels([(si.rms_db, si.peak_db, si.true_peak_db)] * 2)
        self._refresh_region_label()

    @staticmethod
    def _channel_levels(buf: np.ndarray) -> list[tuple[float, float, float]]:
        """Per-channel (RMS, peak, peak-hold) in dB, cheap enough for 20 fps."""
        if buf.ndim == 1:
            buf = buf[:, None]
        out = []
        for c in range(2):
            ch = buf[:, min(c, buf.shape[1] - 1)]
            if ch.size == 0:
                out.append((-60.0, -60.0, -60.0))
                continue
            rms = float(np.sqrt(np.mean(ch * ch)))
            pk = float(np.max(np.abs(ch)))
            r = float(20.0 * np.log10(max(rms, 1e-6)))
            k = float(20.0 * np.log10(max(pk, 1e-6)))
            out.append((max(r, -60.0), max(k, -60.0), max(k, -60.0)))
        return out

    def _refresh_region_label(self):
        if self.audio is None:
            self.region_label.setText("")
            return
        sr = self.audio.sample_rate
        s, e = self._region
        txt = (f"Previewing {self._fmt(s / sr)} \u2013 {self._fmt(e / sr)} "
               f"({(e - s) / sr:.1f} s)")
        sel = self.wave.selection()
        if sel:
            txt += "  \u00b7 selection"
        elif e - s < self.audio.frames:
            txt += "  \u00b7 30 s window around the playhead for responsiveness"
        else:
            txt += "  \u00b7 whole file"
        stages = self.panel.profile().active_stages()
        txt += "  \u00b7  stages: " + (", ".join(stages) if stages else "none (bypass)")
        p = self._payload
        if p is not None:
            if self._ab == "processed" and p.gain_proc_db < -0.01:
                txt += f"  \u00b7  A/B matched: processed {-p.gain_proc_db:.1f} dB down"
            elif self._ab == "original" and p.gain_orig_db < -0.01:
                txt += f"  \u00b7  A/B matched: input {-p.gain_orig_db:.1f} dB down"
            if self._stats_out is not None:
                ceil = self.panel.profile().limiter.ceiling_db
                head = ceil - self._stats_out.true_peak_db
                txt += f"  \u00b7  {head:+.1f} dB below the {ceil:+.0f} dBTP ceiling"
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
        try:
            fmt = settings.format or format_for_extension(settings.path)[0]
        except AudioError as exc:
            QMessageBox.warning(self, "Export settings", str(exc))
            return
        verdict = check_export(
            path=settings.path, fmt=fmt,
            sample_rate=int(settings.sample_rate or self.audio.sample_rate),
            frames=self.audio.frames, channels=self.audio.channels,
            subtype=settings.subtype, source_path=self.audio.path)
        if verdict.errors:
            QMessageBox.warning(self, "Export settings", "\n".join(verdict.errors))
            return
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

    def open_pair_tool(self):
        """Open the read-only pitch/tempo recovery tool.

        Imported here rather than at module scope so the window still starts on a
        Qt build where only the basic widgets exist.
        """
        from .pair_dialog import PairDialog

        current = self.audio.path if self.audio is not None else ""
        dlg = PairDialog(self, loaded=current or "")
        dlg.exec()

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
            "\u2022 \u22121 dBTP true-peak ceiling checked on the encoded export\n"
            "\u2022 stereo width touches only the side signal, so mono fold-down is preserved\n"
            "\u2022 original files are never overwritten\n\n"
            "Preview and export use the same engine (timbreforge.dsp.graph.render).\n\n"
            "Measuring somebody else's edit:\n"
            "Analysis \u203a Pitch/tempo of two files\u2026 (Ctrl+T, or the Pitch/tempo "
            "button in the top bar) takes an original and a modified file and reports "
            "the pitch and tempo change between them, plus the Audacity macro that "
            "reproduces it. That is a measurement tool: nothing it finds is applied to "
            "the audio you export.\n\n"
            "Limits:\n" + describe_limits())

    # ==================================================================
    # events
    # ==================================================================
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in (".wav", ".flac", ".ogg", ".oga", ".mp3", ".aiff", ".aif",
                           ".opus", ".m4a"):
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
        for attr in ("_worker", "_export_worker", "_audition_worker", "_import_worker"):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(3000)
        self.player.shutdown()
        if self._audition_tmp and os.path.exists(self._audition_tmp):
            try:
                os.remove(self._audition_tmp)
            except OSError:
                pass
        super().closeEvent(event)

    def _set_idle_ui(self):
        self.export_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.ab_a.setEnabled(False)
        self.ab_b.setEnabled(False)
        self.meter.set_levels([(-60, -60, -60)] * 2)
        self._audition_btn.setVisible(not self.player.available)
