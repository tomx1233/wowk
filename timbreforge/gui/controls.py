"""Reusable parameter widgets (slider + spin pairs, sections, band editors)."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QSizePolicy, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from ..dsp.biquad import Band
from ..dsp.dynamics import CompressorParams, DynamicEqBand, MultibandParams
from .theme import COLORS


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _label(text: str, dim: bool = True, width: int | None = None) -> QLabel:
    lab = QLabel(text)
    if dim:
        lab.setObjectName("Subtle")
    if width:
        lab.setFixedWidth(width)
    return lab


def spin(lo: float, hi: float, value: float, step: float = 1.0, decimals: int = 1,
         suffix: str = "", width: int = 68) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setFixedWidth(width)
    box.setKeyboardTracking(False)
    box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return box


class ParamControl(QWidget):
    """Label + horizontal slider + numeric box, kept in sync."""

    valueChanged = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, value: float, *,
                 step: float = 0.5, decimals: int = 1, suffix: str = "",
                 default: float | None = None, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._lo, self._hi = float(lo), float(hi)
        self._scale = float(10 ** decimals)
        self._default = value if default is None else default
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self._lab = _label(label, width=88)
        self._lab.setToolTip(tooltip or label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(lo * self._scale)), int(round(hi * self._scale)))
        self.slider.setSingleStep(max(1, int(round(step * self._scale))))
        self.slider.setPageStep(max(1, int(round((hi - lo) * self._scale / 10))))
        self.slider.setValue(int(round(value * self._scale)))
        self.box = spin(lo, hi, value, step, decimals, suffix)
        lay.addWidget(self._lab)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.box)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)
        if tooltip:
            self.setToolTip(tooltip)

    def _from_slider(self, ival: int):
        v = ival / self._scale
        with QSignalBlocker(self.box):
            self.box.setValue(v)
        self.valueChanged.emit(v)

    def _from_box(self, v: float):
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(v * self._scale)))
        self.valueChanged.emit(float(v))

    def value(self) -> float:
        return float(self.box.value())

    def set_value(self, v: float):
        v = float(min(max(v, self._lo), self._hi))
        with QSignalBlocker(self.box), QSignalBlocker(self.slider):
            self.box.setValue(v)
            self.slider.setValue(int(round(v * self._scale)))

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.set_value(self._default)
        self.valueChanged.emit(self._default)


class Section(QWidget):
    """Collapsible titled section."""

    toggled = Signal(bool)

    def __init__(self, title: str, subtitle: str = "", *, collapsed: bool = False,
                 parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QToolButton()
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._header.setText(f"  {title.upper()}")
        self._header.setStyleSheet(
            f"QToolButton {{ text-align:left; padding:7px 6px; border:none;"
            f" color:{COLORS['text_dim']}; font-size:11px; font-weight:600;"
            f" letter-spacing:1px; }}"
            f"QToolButton:hover {{ color:{COLORS['text']}; }}")
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(self._header)

        self._sub = QLabel(subtitle)
        self._sub.setObjectName("Faint")
        self._sub.setWordWrap(True)
        self._sub.setContentsMargins(24, 0, 8, 4)
        if subtitle:
            outer.addWidget(self._sub)

        self._body = QFrame()
        self._body.setObjectName("Card")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 8, 10, 10)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)
        self._body.setVisible(not collapsed)
        self._sub.setVisible(bool(subtitle) and not collapsed)

        self._header.toggled.connect(self._on_toggle)
        self._collapsed = collapsed

    def _on_toggle(self, checked: bool):
        self._body.setVisible(checked)
        self._sub.setVisible(checked and bool(self._sub.text()))
        self._header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._collapsed = not checked
        self.toggled.emit(checked)

    def set_collapsed(self, collapsed: bool):
        self._header.setChecked(not collapsed)

    def body(self) -> QVBoxLayout:
        return self._body_layout


# ---------------------------------------------------------------------------
# static EQ editor
# ---------------------------------------------------------------------------


class _Row(QWidget):
    """One editable band row."""

    changed = Signal()
    removed = Signal(object)

    FIELDS = ()

    def __init__(self, band: Band, parent=None):
        super().__init__(parent)
        self._band = band
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.on = QCheckBox()
        self.on.setChecked(band.enabled)
        self.on.setToolTip("Enable this band")
        self.kind = QComboBox()
        for t, lbl in (("peaking", "Bell"), ("lowshelf", "Low shelf"),
                       ("highshelf", "High shelf"), ("highpass", "Low cut"),
                       ("lowpass", "High cut"), ("notch", "Notch")):
            self.kind.addItem(lbl, t)
        idx = self.kind.findData(band.type)
        self.kind.setCurrentIndex(max(0, idx))
        self.kind.setFixedWidth(82)
        self.freq = spin(10, 22000, band.freq, 10, 0, "", 72)
        self.gain = spin(-24, 24, band.gain_db, 0.1, 1, "", 60)
        self.q = spin(0.1, 18.0, band.q, 0.05, 2, "", 58)
        self.del_btn = QPushButton("\u00d7")
        self.del_btn.setObjectName("Small")
        self.del_btn.setFixedWidth(24)
        self.del_btn.setToolTip("Remove band")
        lay.addWidget(self.on)
        lay.addWidget(self.kind)
        lay.addWidget(self.freq)
        lay.addWidget(self.gain)
        lay.addWidget(self.q)
        lay.addWidget(self.del_btn)

        self.on.toggled.connect(lambda _: self._emit())
        self.kind.currentIndexChanged.connect(lambda _: self._type_changed())
        for w in (self.freq, self.gain, self.q):
            w.valueChanged.connect(lambda _v: self._emit())
        self.del_btn.clicked.connect(lambda: self.removed.emit(self))

    def _type_changed(self):
        t = self.kind.currentData()
        # cut filters do not use gain
        self.gain.setEnabled(t not in ("highpass", "lowpass", "notch"))
        self._emit()

    def _emit(self):
        b = self.band()
        self._band = b
        self.changed.emit()

    def band(self) -> Band:
        return Band(type=self.kind.currentData(), freq=float(self.freq.value()),
                    gain_db=float(self.gain.value()), q=float(self.q.value()),
                    enabled=self.on.isChecked())


class EqEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_Row] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addSpacing(20)
        head.addWidget(_label("Type", width=82))
        head.addWidget(_label("Frequency", width=72))
        head.addWidget(_label("Gain", width=60))
        head.addWidget(_label("Q", width=58))
        head.addStretch(1)
        lay.addLayout(head)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(3)
        lay.addLayout(self._rows_box)
        btns = QHBoxLayout()
        self.add_btn = QPushButton("+ Add band")
        self.add_btn.setObjectName("Ghost")
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setObjectName("Ghost")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.clear_btn)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.add_btn.clicked.connect(lambda: self._add(Band("peaking", 1000.0, 0.0, 1.0)))
        self.clear_btn.clicked.connect(self._clear)

    def _add(self, band: Band):
        row = _Row(band)
        row.changed.connect(self._on_row_changed)
        row.removed.connect(self._remove)
        row._type_changed()
        self._rows.append(row)
        self._rows_box.addWidget(row)
        self.changed.emit()

    def _remove(self, row: _Row):
        if row in self._rows:
            self._rows.remove(row)
            self._rows_box.removeWidget(row)
            row.deleteLater()
            self.changed.emit()

    def _clear(self):
        for row in list(self._rows):
            self._rows_box.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self.changed.emit()

    def _on_row_changed(self):
        self.changed.emit()

    def set_bands(self, bands: Iterable[Band]):
        for row in list(self._rows):
            self._rows_box.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        for b in bands:
            row = _Row(b)
            row.changed.connect(self._on_row_changed)
            row.removed.connect(self._remove)
            row._type_changed()
            self._rows.append(row)
            self._rows_box.addWidget(row)

    def bands(self) -> list[Band]:
        return [r.band() for r in self._rows]


# ---------------------------------------------------------------------------
# dynamic EQ editor
# ---------------------------------------------------------------------------


class DynEqRow(QFrame):
    changed = Signal()
    removed = Signal(object)

    def __init__(self, band: DynamicEqBand, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.on = QCheckBox()
        self.on.setChecked(band.enabled)
        self.on.setToolTip("Enable this band")
        top.addWidget(self.on)
        top.addWidget(_label("Band", width=34))
        self.freq = spin(30, 20000, band.freq, 10, 0, "", 70)
        self.gain = spin(-18, 18, band.gain_db, 0.5, 1, "", 60)
        self.q = spin(0.2, 10, band.q, 0.05, 2, "", 56)
        self.mode = QComboBox()
        self.mode.addItem("Cut when loud", "above")
        self.mode.addItem("Lift when quiet", "below")
        self.mode.setCurrentIndex(0 if band.direction == "above" else 1)
        self.mode.setFixedWidth(120)
        top.addWidget(self.freq)
        top.addWidget(self.gain)
        top.addWidget(self.q)
        top.addWidget(self.mode)
        top.addStretch(1)
        self.del_btn = QPushButton("\u00d7")
        self.del_btn.setObjectName("Small")
        self.del_btn.setFixedWidth(24)
        top.addWidget(self.del_btn)
        lay.addLayout(top)

        bot = QHBoxLayout()
        bot.setSpacing(6)
        bot.addWidget(_label("Threshold", width=58))
        self.thr = spin(-60, 0, band.threshold_db, 1, 1, "", 56)
        bot.addWidget(self.thr)
        bot.addWidget(_label("Ratio", width=32))
        self.ratio = spin(1.0, 10, band.ratio, 0.1, 1, "", 50)
        bot.addWidget(self.ratio)
        bot.addWidget(_label("Attack", width=38))
        self.att = spin(0.5, 200, band.attack_ms, 1, 0, "", 56)
        bot.addWidget(self.att)
        bot.addWidget(_label("Release", width=46))
        self.rel = spin(10, 2000, band.release_ms, 10, 0, "", 60)
        bot.addWidget(self.rel)
        bot.addWidget(_label("Max", width=26))
        self.maxred = spin(0.5, 18, band.max_reduction_db, 0.5, 1, "", 52)
        bot.addWidget(self.maxred)
        bot.addStretch(1)
        lay.addLayout(bot)

        for w in (self.freq, self.gain, self.q, self.thr, self.ratio, self.att,
                  self.rel, self.maxred):
            w.valueChanged.connect(lambda _v: self.changed.emit())
        self.on.toggled.connect(lambda _c: self.changed.emit())
        self.mode.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self.del_btn.clicked.connect(lambda: self.removed.emit(self))

    def band(self) -> DynamicEqBand:
        return DynamicEqBand(
            enabled=self.on.isChecked(), freq=float(self.freq.value()),
            q=float(self.q.value()), gain_db=float(self.gain.value()),
            threshold_db=float(self.thr.value()), ratio=float(self.ratio.value()),
            attack_ms=float(self.att.value()), release_ms=float(self.rel.value()),
            direction=self.mode.currentData(), max_reduction_db=float(self.maxred.value()),
        )


class DynEqEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[DynEqRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._rows_box = QVBoxLayout()
        self._rows_box.setSpacing(4)
        lay.addLayout(self._rows_box)
        btn = QHBoxLayout()
        add = QPushButton("+ Add dynamic band")
        add.setObjectName("Ghost")
        add.clicked.connect(lambda: self._add(DynamicEqBand(enabled=True)))
        btn.addWidget(add)
        btn.addStretch(1)
        lay.addLayout(btn)

    def _add(self, band: DynamicEqBand):
        row = DynEqRow(band)
        row.changed.connect(self.changed)
        row.removed.connect(self._remove)
        self._rows.append(row)
        self._rows_box.addWidget(row)
        self.changed.emit()

    def _remove(self, row: DynEqRow):
        if row in self._rows:
            self._rows.remove(row)
            self._rows_box.removeWidget(row)
            row.deleteLater()
            self.changed.emit()

    def set_bands(self, bands: Sequence[DynamicEqBand]):
        for row in list(self._rows):
            self._rows_box.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        for b in bands:
            row = DynEqRow(b)
            row.changed.connect(self.changed)
            row.removed.connect(self._remove)
            self._rows.append(row)
            self._rows_box.addWidget(row)

    def bands(self) -> list[DynamicEqBand]:
        return [r.band() for r in self._rows]


# ---------------------------------------------------------------------------
# multiband editor
# ---------------------------------------------------------------------------


class MultibandEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bands: list[CompressorParams] = [CompressorParams(), CompressorParams(),
                                               CompressorParams()]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        xo = QHBoxLayout()
        xo.setSpacing(6)
        xo.addWidget(_label("Low / mid", width=64))
        self.xo1 = spin(40, 800, 180, 5, 0, " Hz", 68)
        xo.addWidget(self.xo1)
        xo.addSpacing(8)
        xo.addWidget(_label("Mid / high", width=64))
        self.xo2 = spin(400, 12000, 1800, 20, 0, " Hz", 74)
        xo.addWidget(self.xo2)
        xo.addStretch(1)
        lay.addLayout(xo)
        head = QHBoxLayout()
        head.setSpacing(6)
        for txt, w in (("Band", 56), ("On", 24), ("Threshold", 62), ("Ratio", 54),
                       ("Attack", 58), ("Release", 62), ("Mix", 54)):
            head.addWidget(_label(txt, width=w))
        head.addStretch(1)
        lay.addLayout(head)
        self._rows = []
        names = ["Low", "Mid", "High"]
        for i, name in enumerate(names):
            row = QHBoxLayout()
            row.setSpacing(6)
            on = QCheckBox()
            on.setChecked(False)
            row.addWidget(_label(name, width=56))
            row.addWidget(on)
            thr = spin(-60, 0, -24, 1, 1, "", 58)
            ratio = spin(1, 10, 2, 0.1, 1, "", 50)
            att = spin(1, 200, 20, 1, 0, "", 54)
            rel = spin(20, 2000, 200, 10, 0, "", 58)
            mix = spin(0, 100, 100, 5, 0, "", 50)
            for w in (thr, ratio, att, rel, mix):
                row.addWidget(w)
            row.addStretch(1)
            lay.addLayout(row)
            self._rows.append({"on": on, "thr": thr, "ratio": ratio, "att": att,
                               "rel": rel, "mix": mix})
            on.toggled.connect(lambda _c: self.changed.emit())
            for w in (thr, ratio, att, rel, mix):
                w.valueChanged.connect(lambda _v: self.changed.emit())
        self.xo1.valueChanged.connect(lambda _v: self.changed.emit())
        self.xo2.valueChanged.connect(lambda _v: self.changed.emit())

    def set_params(self, p: MultibandParams):
        self.xo1.setValue(p.crossovers_hz[0] if p.crossovers_hz else 180)
        self.xo2.setValue(p.crossovers_hz[1] if len(p.crossovers_hz) > 1 else 1800)
        for i, row in enumerate(self._rows):
            b = p.bands[i] if i < len(p.bands) else CompressorParams()
            row["on"].setChecked(b.enabled)
            row["thr"].setValue(b.threshold_db)
            row["ratio"].setValue(b.ratio)
            row["att"].setValue(b.attack_ms)
            row["rel"].setValue(b.release_ms)
            row["mix"].setValue(b.mix * 100)

    def params(self, base: MultibandParams | None = None) -> MultibandParams:
        bands = []
        for i, row in enumerate(self._rows):
            prev = base.bands[i] if base and i < len(base.bands) else CompressorParams()
            bands.append(CompressorParams(
                enabled=row["on"].isChecked(), threshold_db=row["thr"].value(),
                ratio=row["ratio"].value(), attack_ms=max(1.0, row["att"].value()),
                release_ms=max(10.0, row["rel"].value()), knee_db=prev.knee_db,
                detector=prev.detector, window_ms=prev.window_ms,
                stereo=prev.stereo, mix=row["mix"].value() / 100.0,
                sidechain_hp_hz=prev.sidechain_hp_hz, makeup_db=0.0,
            ))
        return MultibandParams(enabled=True,
                               crossovers_hz=[self.xo1.value(), self.xo2.value()],
                               bands=bands)
