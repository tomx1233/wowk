"""The processing panel.

Two faces, one engine:

* **Simple** - five controls (Character, Tone, Tightness, Air, Output) that
  generate a complete profile.  This is the default, because the full graph is
  more knobs than most sessions need.
* **Advanced** - every stage of the graph, for when a specific setting matters.

Switching to Advanced loads the profile the simple controls currently describe,
so the two views can never disagree about what is being rendered.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..dsp.graph import (Profile, SimpleSettings, preset_available,
                         profile_from_simple)
from .controls import (DynEqEditor, EqEditor, MultibandEditor, ParamControl,
                       Section, _label, spin)
from .widgets import EqCurveWidget


class ProcessingPanel(QWidget):
    changed = Signal()             # a parameter changed (the window debounces)
    profileReplaced = Signal()     # load / reset / mode switch: caches invalid

    def __init__(self, parent=None):
        super().__init__(parent)
        self._p = Profile()
        self._simple = SimpleSettings()
        self._sr = 44100
        self._suspend = False
        self._simple_mode = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        outer.addWidget(self._build_header())

        self._stack = QStackedWidget()
        self._simple_page = self._build_simple_page()
        self._advanced_page = self._build_advanced_page()
        self._stack.addWidget(self._simple_page)
        self._stack.addWidget(self._advanced_page)
        outer.addWidget(self._stack, 1)

        self._apply_mode()

    # ------------------------------------------------------------------
    # header (always visible)
    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Card")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.mode_btn = QPushButton("Advanced view")
        self.mode_btn.setObjectName("Toggle")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setToolTip("Show every stage of the processing graph. "
                                 "The simple controls are the recommended way in.")
        self.mode_btn.toggled.connect(self._on_mode_toggled)
        top.addWidget(self.mode_btn)
        top.addStretch(1)
        self.bypass_btn = QPushButton("Bypass")
        self.bypass_btn.setObjectName("Toggle")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.setToolTip("Render the input unchanged through the same path "
                                   "so you can compare A and B fairly.")
        self.bypass_btn.toggled.connect(self._on_bypass)
        top.addWidget(self.bypass_btn)
        lay.addLayout(top)

        self.summary = QLabel("")
        self.summary.setObjectName("Faint")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        return wrap

    def _on_mode_toggled(self, advanced: bool):
        if self._suspend:
            return
        if advanced:
            # freeze what the simple controls currently describe
            self._p = profile_from_simple(self._simple)
        self._simple_mode = not advanced
        self._apply_mode()
        self.profileReplaced.emit()

    def _apply_mode(self):
        # guard first: changing the button state re-enters _on_mode_toggled
        self._suspend = True
        self._stack.setCurrentIndex(1 if not self._simple_mode else 0)
        self.mode_btn.setChecked(not self._simple_mode)
        self.mode_btn.setText("Simple view" if not self._simple_mode
                              else "Advanced view")
        if not self._simple_mode:
            self._refresh_advanced()
        self._suspend = False
        self._update_summary()

    def _on_bypass(self, checked: bool):
        if self._suspend:
            return
        self._p.bypass = bool(checked)
        self.changed.emit()

    def _update_summary(self):
        prof = self.profile()
        stages = prof.active_stages()
        self.summary.setText(
            f"{prof.name} \u2014 " + (", ".join(stages).replace("_", " ")
                                     if stages else "no processing (bypass)")
            + ". Output is level-matched to the input (never louder) and capped at "
              f"{prof.limiter.ceiling_db:+.0f} dBTP.")

    # ------------------------------------------------------------------
    # simple page
    # ------------------------------------------------------------------
    def _build_simple_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("Panel")
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        intro = QLabel("Turn up Character until it sounds right, then adjust tone, "
                       "tightness and air.")
        intro.setObjectName("Faint")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        self.s_character = ParamControl(
            "Character", 0, 100, self._simple.character, step=1, decimals=0,
            suffix=" %", default=55,
            tooltip="Overall amount of the processing: tone shaping, level-dependent "
                    "brightness, gentle dynamics and a little harmonic texture. "
                    "0 % is a neutral render.")
        self.s_tone = ParamControl(
            "Tone", -100, 100, self._simple.tone, step=1, decimals=0,
            suffix="", default=0,
            tooltip="Negative is darker and fuller, positive is brighter and leaner.")
        self.s_tightness = ParamControl(
            "Tightness", 0, 100, self._simple.tightness, step=1, decimals=0,
            suffix=" %", default=35,
            tooltip="How much the dynamics are controlled. 0 % leaves them alone.")
        self.s_air = ParamControl(
            "Air", 0, 100, self._simple.air, step=1, decimals=0,
            suffix=" %", default=25,
            tooltip="Extra top-end lift on top of Character.")
        self.s_trim = ParamControl(
            "Output", -12, 12, self._simple.output_trim_db, step=0.5, decimals=1,
            suffix=" dB", default=0,
            tooltip="Extra gain on top of the automatic level match. Leave at 0 to "
                    "keep the export at the input's loudness.")

        for w in (self.s_character, self.s_tone, self.s_tightness, self.s_air,
                  self.s_trim):
            w.valueChanged.connect(self._on_simple_changed)
            lay.addWidget(w)

        lay.addWidget(self._separator())
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(_label("Starting points", width=100))
        for label, vals in (("Subtle", (30, 0, 20, 15)),
                            ("Balanced", (55, 0, 35, 25)),
                            ("Strong", (80, 10, 55, 45)),
                            ("Maximum", (100, 15, 80, 70))):
            b = QPushButton(label)
            b.setObjectName("Small")
            b.clicked.connect(lambda _c=False, v=vals: self._apply_simple_preset(v))
            preset_row.addWidget(b)
        preset_row.addStretch(1)
        lay.addLayout(preset_row)

        lay.addWidget(self._separator())
        self.s_note = QLabel("")
        self.s_note.setObjectName("Faint")
        self.s_note.setWordWrap(True)
        lay.addWidget(self.s_note)

        lay.addStretch(1)
        return scroll

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#2c333c;")
        return line

    def _apply_simple_preset(self, vals):
        c, t, ti, a = vals
        self._suspend = True
        self.s_character.set_value(c)
        self.s_tone.set_value(t)
        self.s_tightness.set_value(ti)
        self.s_air.set_value(a)
        self._suspend = False
        self._on_simple_changed()

    def _on_simple_changed(self, *_):
        if self._suspend:
            return
        self._simple = SimpleSettings(
            character=self.s_character.value(),
            tone=self.s_tone.value(),
            tightness=self.s_tightness.value(),
            air=self.s_air.value(),
            output_trim_db=self.s_trim.value(),
            ceiling_db=self._p.limiter.ceiling_db if self._p else -1.0,
        )
        self._p = profile_from_simple(self._simple)
        self._p.bypass = self.bypass_btn.isChecked()
        self._update_summary()
        self.changed.emit()

    # ------------------------------------------------------------------
    # advanced page
    # ------------------------------------------------------------------
    def _build_advanced_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root = QWidget()
        root.setObjectName("Panel")
        scroll.setWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 10, 8, 10)
        lay.setSpacing(8)

        lay.addWidget(self._build_profile_card())
        lay.addWidget(self._build_tone())
        lay.addWidget(self._build_dynamic_tone())
        lay.addWidget(self._build_dynamics())
        lay.addWidget(self._build_harmonics())
        lay.addWidget(self._build_transient())
        lay.addWidget(self._build_stereo())
        lay.addWidget(self._build_output())
        lay.addStretch(1)
        return scroll

    def _build_profile_card(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Card")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(_label("Profile", width=52))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Profile name")
        self.name_edit.editingFinished.connect(self._on_name)
        row.addWidget(self.name_edit, 1)
        lay.addLayout(row)

        btns = QHBoxLayout()
        btns.setSpacing(6)
        load_btn = QPushButton("Load\u2026")
        load_btn.setObjectName("Ghost")
        load_btn.clicked.connect(self._load_profile)
        save_btn = QPushButton("Save as\u2026")
        save_btn.setObjectName("Ghost")
        save_btn.clicked.connect(self._save_profile)
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("Ghost")
        reset_btn.clicked.connect(self._reset_profile)
        for b in (load_btn, save_btn, reset_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)

        self._presets = QComboBox()
        self._presets.addItem("Load a built-in preset\u2026", "")
        for name in preset_available():
            self._presets.addItem(name, name)
        self._presets.currentIndexChanged.connect(
            lambda _i: self._apply_preset(self._presets.currentData()))
        lay.addWidget(self._presets)
        return wrap

    def _on_name(self):
        if self._suspend:
            return
        self._p.name = self.name_edit.text().strip() or "Untitled profile"
        self._update_summary()
        self.changed.emit()

    def _apply_preset(self, key: str):
        if not key:
            return
        fn = preset_available().get(key)
        if fn is None:
            return
        self.set_profile(fn(), self._sr)
        self._presets.setCurrentIndex(0)
        self.profileReplaced.emit()

    def _load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load processing profile", "",
                                              "Timbreforge profile (*.json)")
        if not path:
            return
        try:
            p = Profile.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not load profile", str(exc))
            return
        errs = p.validate()
        if errs:
            QMessageBox.warning(self, "Profile rejected",
                                "This profile is not valid:\n\n" + "\n".join(errs[:8]))
            return
        self.set_profile(p, self._sr)
        self.profileReplaced.emit()

    def _save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save processing profile",
                                              f"{self._p.name}.json",
                                              "Timbreforge profile (*.json)")
        if not path:
            return
        try:
            self._p.save(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not save profile", str(exc))
            return

    def _reset_profile(self):
        self._simple = SimpleSettings()
        self._suspend = True
        self.s_character.set_value(self._simple.character)
        self.s_tone.set_value(self._simple.tone)
        self.s_tightness.set_value(self._simple.tightness)
        self.s_air.set_value(self._simple.air)
        self.s_trim.set_value(self._simple.output_trim_db)
        self._suspend = False
        self._p = profile_from_simple(self._simple)
        self.bypass_btn.setChecked(False)
        self._p.bypass = False
        if not self._simple_mode:
            self._refresh_advanced()
        self._update_summary()
        self.profileReplaced.emit()

    # -- advanced sections ----------------------------------------------
    def _build_tone(self) -> Section:
        sec = Section("Tonal shaping",
                      "One static EQ cascade. Minimum phase is the default; "
                      "linear phase is available when pre-ringing must be avoided.")
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(_label("Phase", width=44))
        self.eq_mode = QComboBox()
        self.eq_mode.addItem("Minimum phase", "minimum")
        self.eq_mode.addItem("Linear phase (FIR)", "linear")
        self.eq_mode.setFixedWidth(150)
        self.eq_mode.currentIndexChanged.connect(self._on_eq_mode)
        top.addWidget(self.eq_mode)
        top.addStretch(1)
        sec.body().addLayout(top)
        self.eq_curve = EqCurveWidget()
        sec.body().addWidget(self.eq_curve)
        self.eq_editor = EqEditor()
        self.eq_editor.changed.connect(self._on_eq_changed)
        sec.body().addWidget(self.eq_editor)
        return sec

    def _on_eq_mode(self, *_):
        if self._suspend:
            return
        self._p.eq_mode = self.eq_mode.currentData()
        self.changed.emit()

    def _on_eq_changed(self):
        if self._suspend:
            return
        self._p.eq_bands = self.eq_editor.bands()
        self.eq_curve.set_bands(self._p.eq_bands, self._sr)
        self._update_summary()
        self.changed.emit()

    def _build_dynamic_tone(self) -> Section:
        sec = Section("Dynamic tone (level-dependent EQ)", collapsed=True,
                      subtitle="Peaking bands whose amount follows input level.")
        self.dyn_enable = QCheckBox("Enable dynamic EQ")
        self.dyn_enable.toggled.connect(self._on_dyn_enable)
        sec.body().addWidget(self.dyn_enable)
        self.dyn_editor = DynEqEditor()
        self.dyn_editor.changed.connect(self._on_dyn_changed)
        sec.body().addWidget(self.dyn_editor)
        return sec

    def _on_dyn_enable(self, on: bool):
        if self._suspend:
            return
        self._p.dynamic_eq.enabled = bool(on)
        self.changed.emit()

    def _on_dyn_changed(self):
        if self._suspend:
            return
        self._p.dynamic_eq.bands = self.dyn_editor.bands()
        if self._p.dynamic_eq.bands and not self.dyn_enable.isChecked():
            self.dyn_enable.setChecked(True)
            self._p.dynamic_eq.enabled = True
        self.changed.emit()

    def _build_dynamics(self) -> Section:
        sec = Section("Dynamics",
                      "Feed-forward compression. Makeup is manual; the output stage "
                      "keeps the render at the input's loudness.")
        self.comp_enable = QCheckBox("Enable broadband compressor")
        self.comp_enable.toggled.connect(self._on_comp_enable)
        sec.body().addWidget(self.comp_enable)

        self.comp_thr = ParamControl("Threshold", -60, 0, -18, step=0.5, suffix=" dB")
        self.comp_ratio = ParamControl("Ratio", 1.0, 12.0, 2.0, step=0.1, decimals=2,
                                       suffix=" :1")
        self.comp_att = ParamControl("Attack", 0.5, 200, 15, step=1, suffix=" ms")
        self.comp_rel = ParamControl("Release", 20, 2000, 180, step=5, decimals=0,
                                     suffix=" ms")
        self.comp_knee = ParamControl("Knee", 0, 24, 8, step=1, suffix=" dB")
        self.comp_mix = ParamControl("Parallel mix", 0, 100, 100, step=5, decimals=0,
                                     suffix=" %")
        self.comp_makeup = ParamControl("Makeup", -12, 6, 0, step=0.5, suffix=" dB",
                                        tooltip="Manual makeup only. A positive value is "
                                                "an explicit loudness decision.")
        self.comp_sc = ParamControl("Sidechain HP", 0, 300, 0, step=5, decimals=0,
                                    suffix=" Hz", tooltip="High-pass the detector "
                                                          "(0 = off).")
        for w in (self.comp_thr, self.comp_ratio, self.comp_att, self.comp_rel,
                  self.comp_knee, self.comp_mix, self.comp_makeup, self.comp_sc):
            w.valueChanged.connect(self._on_comp_changed)
            sec.body().addWidget(w)

        det = QHBoxLayout()
        det.setSpacing(6)
        det.addWidget(_label("Detector", width=88))
        self.comp_det = QComboBox()
        self.comp_det.addItem("RMS (smooth)", "rms")
        self.comp_det.addItem("Peak (fast)", "peak")
        self.comp_det.currentIndexChanged.connect(lambda _i: self._on_comp_changed())
        det.addWidget(self.comp_det)
        det.addStretch(1)
        det.addWidget(_label("Channels", width=60))
        self.comp_stereo = QComboBox()
        self.comp_stereo.addItem("Linked", "linked")
        self.comp_stereo.addItem("Independent", "independent")
        self.comp_stereo.currentIndexChanged.connect(lambda _i: self._on_comp_changed())
        det.addWidget(self.comp_stereo)
        sec.body().addLayout(det)

        sec.body().addWidget(self._separator())
        mb_head = QHBoxLayout()
        self.mb_enable = QCheckBox("Enable multiband dynamics")
        self.mb_enable.toggled.connect(self._on_mb_changed)
        mb_head.addWidget(self.mb_enable)
        mb_head.addStretch(1)
        sec.body().addLayout(mb_head)
        self.mb_editor = MultibandEditor()
        self.mb_editor.changed.connect(self._on_mb_changed)
        sec.body().addWidget(self.mb_editor)
        note = QLabel("Crossovers are complementary linear-phase FIR: the bands sum back "
                      "to the input exactly and stay time aligned.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        sec.body().addWidget(note)
        return sec

    def _on_comp_enable(self, on: bool):
        if self._suspend:
            return
        self._p.compressor.enabled = bool(on)
        self.changed.emit()

    def _on_comp_changed(self, *_):
        if self._suspend:
            return
        c = self._p.compressor
        c.threshold_db = self.comp_thr.value()
        c.ratio = self.comp_ratio.value()
        c.attack_ms = self.comp_att.value()
        c.release_ms = self.comp_rel.value()
        c.knee_db = self.comp_knee.value()
        c.mix = self.comp_mix.value() / 100.0
        c.makeup_db = self.comp_makeup.value()
        c.sidechain_hp_hz = self.comp_sc.value()
        c.detector = self.comp_det.currentData()
        c.stereo = self.comp_stereo.currentData()
        self.changed.emit()

    def _on_mb_changed(self, *_):
        if self._suspend:
            return
        p = self.mb_editor.params(self._p.multiband)
        p.enabled = self.mb_enable.isChecked()
        self._p.multiband = p
        self.changed.emit()

    def _build_harmonics(self) -> Section:
        sec = Section("Harmonics (saturation)", collapsed=True,
                      subtitle="Oversampled waveshaping. Level is compensated by "
                               "attenuation only, so this cannot become a loudness boost.")
        self.sat_enable = QCheckBox("Enable harmonic stage")
        self.sat_enable.toggled.connect(self._on_sat_changed)
        sec.body().addWidget(self.sat_enable)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(_label("Curve", width=88))
        self.sat_shaper = QComboBox()
        for key, label in (("tanh", "Tanh (soft)"), ("arctan", "Arctan (gentle)"),
                           ("soft", "Soft clip"), ("asym", "Asymmetric")):
            self.sat_shaper.addItem(label, key)
        self.sat_shaper.currentIndexChanged.connect(lambda _i: self._on_sat_changed())
        row.addWidget(self.sat_shaper)
        row.addStretch(1)
        row.addWidget(_label("Oversample", width=74))
        self.sat_os = QComboBox()
        for v in (1, 2, 4, 8):
            self.sat_os.addItem(f"{v}x", v)
        self.sat_os.currentIndexChanged.connect(lambda _i: self._on_sat_changed())
        row.addWidget(self.sat_os)
        sec.body().addLayout(row)
        self.sat_drive = ParamControl("Drive", -12, 18, 0, step=0.5, suffix=" dB")
        self.sat_mix = ParamControl("Mix", 0, 100, 0, step=1, decimals=0, suffix=" %")
        for w in (self.sat_drive, self.sat_mix):
            w.valueChanged.connect(self._on_sat_changed)
            sec.body().addWidget(w)
        return sec

    def _on_sat_changed(self, *_):
        if self._suspend:
            return
        s = self._p.saturation
        s.enabled = self.sat_enable.isChecked()
        s.shaper = self.sat_shaper.currentData()
        s.oversample = int(self.sat_os.currentData())
        s.drive_db = self.sat_drive.value()
        s.mix = self.sat_mix.value() / 100.0
        s.compensate = "rms"
        self.changed.emit()

    def _build_transient(self) -> Section:
        sec = Section("Transients", collapsed=True,
                      subtitle="Attack/sustain shaping ahead of the dynamics stages.")
        self.tr_enable = QCheckBox("Enable transient shaping")
        self.tr_enable.toggled.connect(self._on_tr_changed)
        sec.body().addWidget(self.tr_enable)
        self.tr_attack = ParamControl("Attack", -12, 12, 0, step=0.5, suffix=" dB")
        self.tr_sustain = ParamControl("Sustain", -12, 12, 0, step=0.5, suffix=" dB")
        self.tr_sens = ParamControl("Sensitivity", 1, 18, 6, step=0.5, suffix=" dB")
        for w in (self.tr_attack, self.tr_sustain, self.tr_sens):
            w.valueChanged.connect(self._on_tr_changed)
            sec.body().addWidget(w)
        return sec

    def _on_tr_changed(self, *_):
        if self._suspend:
            return
        t = self._p.transient
        t.enabled = self.tr_enable.isChecked()
        t.attack_db = self.tr_attack.value()
        t.sustain_db = self.tr_sustain.value()
        t.sensitivity_db = self.tr_sens.value()
        self.changed.emit()

    def _build_stereo(self) -> Section:
        sec = Section("Stereo", collapsed=True,
                      subtitle="Width changes the side signal only, so mono fold-down is "
                               "untouched. No phase rotation is offered.")
        self.st_enable = QCheckBox("Enable width control")
        self.st_enable.toggled.connect(self._on_stereo_changed)
        sec.body().addWidget(self.st_enable)
        self.st_width = ParamControl("Width", 0, 200, 100, step=1, decimals=0,
                                     suffix=" %", tooltip="100% = unchanged, 0% = mono")
        self.st_mono = ParamControl("Mono below", 0, 400, 0, step=10, decimals=0,
                                    suffix=" Hz")
        for w in (self.st_width, self.st_mono):
            w.valueChanged.connect(self._on_stereo_changed)
            sec.body().addWidget(w)
        return sec

    def _on_stereo_changed(self, *_):
        if self._suspend:
            return
        s = self._p.stereo
        s.enabled = self.st_enable.isChecked()
        s.width = self.st_width.value() / 100.0
        s.mono_below_hz = self.st_mono.value()
        self.changed.emit()

    def _build_output(self) -> Section:
        sec = Section("Output",
                      "The render is matched to the input's loudness (never above it) and "
                      "capped at a true-peak ceiling before encoding.")
        self.match_enable = QCheckBox("Match the input's loudness (attenuation-biased)")
        self.match_enable.setChecked(True)
        self.match_enable.toggled.connect(self._on_output_changed)
        sec.body().addWidget(self.match_enable)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(_label("Match on", width=88))
        self.match_mode = QComboBox()
        self.match_mode.addItem("Integrated loudness (LUFS)", "lufs")
        self.match_mode.addItem("RMS", "rms")
        self.match_mode.currentIndexChanged.connect(lambda _i: self._on_output_changed())
        row.addWidget(self.match_mode)
        row.addStretch(1)
        sec.body().addLayout(row)

        self.trim = ParamControl("Output trim", -24, 24, 0, step=0.5, suffix=" dB")
        self.trim.valueChanged.connect(self._on_output_changed)
        sec.body().addWidget(self.trim)

        lim_row = QHBoxLayout()
        lim_row.setSpacing(6)
        self.lim_enable = QCheckBox("True-peak limiter")
        self.lim_enable.setChecked(True)
        self.lim_enable.toggled.connect(self._on_output_changed)
        lim_row.addWidget(self.lim_enable)
        lim_row.addStretch(1)
        lim_row.addWidget(_label("Ceiling", width=48))
        self.lim_ceiling = spin(-6.0, -0.1, -1.0, 0.1, 1, " dBTP", 84)
        self.lim_ceiling.valueChanged.connect(lambda _v: self._on_output_changed())
        lim_row.addWidget(self.lim_ceiling)
        lim_row.addWidget(_label("Look-ahead", width=66))
        self.lim_look = spin(1.0, 20.0, 6.0, 0.5, 1, " ms", 74)
        self.lim_look.valueChanged.connect(lambda _v: self._on_output_changed())
        lim_row.addWidget(self.lim_look)
        sec.body().addLayout(lim_row)
        note = QLabel("Limiting is offline with look-ahead: the output length is identical "
                      "to the input (zero added latency).")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        sec.body().addWidget(note)
        return sec

    def _on_output_changed(self, *_):
        if self._suspend:
            return
        self._p.loudness_match = self.match_enable.isChecked()
        self._p.loudness_match_mode = self.match_mode.currentData()
        self._p.output_trim_db = self.trim.value()
        self._p.limiter.enabled = self.lim_enable.isChecked()
        self._p.limiter.ceiling_db = self.lim_ceiling.value()
        self._p.limiter.lookahead_ms = self.lim_look.value()
        self.changed.emit()

    # ------------------------------------------------------------------
    # profile in / out
    # ------------------------------------------------------------------
    def profile(self) -> Profile:
        if self._simple_mode:
            p = profile_from_simple(self._simple)
            p.bypass = self.bypass_btn.isChecked()
            return p
        p = copy.deepcopy(self._p)
        p.bypass = self.bypass_btn.isChecked()
        return p

    def simple_settings(self) -> SimpleSettings:
        return copy.deepcopy(self._simple)

    def is_simple(self) -> bool:
        return self._simple_mode

    def set_profile(self, p: Profile, sr: int, *, refresh: bool = True):
        """Load a full profile: always switches to the advanced view."""
        self._p = copy.deepcopy(p)
        self._sr = int(sr)
        self.bypass_btn.setChecked(bool(self._p.bypass))
        if self._simple_mode:
            self._simple_mode = False
        if refresh:
            self._stack.setCurrentIndex(1)
            self.mode_btn.setChecked(True)
            self.mode_btn.setText("Simple view")
            self._refresh_advanced()
        self._update_summary()

    def set_sample_rate(self, sr: int):
        self._sr = int(sr)
        if hasattr(self, "eq_curve"):
            self.eq_curve.set_bands(self._p.eq_bands, self._sr)

    def _refresh_advanced(self):
        self._suspend = True
        p = self._p
        self.name_edit.setText(p.name)
        i = self.eq_mode.findData(p.eq_mode)
        self.eq_mode.setCurrentIndex(max(0, i))
        self.eq_editor.set_bands(p.eq_bands)
        self.eq_curve.set_bands(p.eq_bands, self._sr)

        self.dyn_enable.setChecked(p.dynamic_eq.enabled)
        self.dyn_editor.set_bands(p.dynamic_eq.bands)

        c = p.compressor
        self.comp_enable.setChecked(c.enabled)
        self.comp_thr.set_value(c.threshold_db)
        self.comp_ratio.set_value(c.ratio)
        self.comp_att.set_value(c.attack_ms)
        self.comp_rel.set_value(c.release_ms)
        self.comp_knee.set_value(c.knee_db)
        self.comp_mix.set_value(c.mix * 100)
        self.comp_makeup.set_value(c.makeup_db)
        self.comp_sc.set_value(c.sidechain_hp_hz)
        self.comp_det.setCurrentIndex(max(0, self.comp_det.findData(c.detector)))
        self.comp_stereo.setCurrentIndex(max(0, self.comp_stereo.findData(c.stereo)))

        self.mb_enable.setChecked(p.multiband.enabled)
        self.mb_editor.set_params(p.multiband)

        s = p.saturation
        self.sat_enable.setChecked(s.enabled)
        self.sat_shaper.setCurrentIndex(max(0, self.sat_shaper.findData(s.shaper)))
        self.sat_os.setCurrentIndex(max(0, self.sat_os.findData(int(s.oversample))))
        self.sat_drive.set_value(s.drive_db)
        self.sat_mix.set_value(s.mix * 100)

        t = p.transient
        self.tr_enable.setChecked(t.enabled)
        self.tr_attack.set_value(t.attack_db)
        self.tr_sustain.set_value(t.sustain_db)
        self.tr_sens.set_value(t.sensitivity_db)

        st = p.stereo
        self.st_enable.setChecked(st.enabled)
        self.st_width.set_value(st.width * 100)
        self.st_mono.set_value(st.mono_below_hz)

        self.match_enable.setChecked(p.loudness_match)
        self.match_mode.setCurrentIndex(max(0, self.match_mode.findData(
            p.loudness_match_mode)))
        self.trim.set_value(p.output_trim_db)
        self.lim_enable.setChecked(p.limiter.enabled)
        self.lim_ceiling.setValue(p.limiter.ceiling_db)
        self.lim_look.setValue(p.limiter.lookahead_ms)
        self._suspend = False
        self._update_summary()
