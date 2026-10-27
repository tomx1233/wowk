"""Custom painted widgets: waveform, level meter, EQ curve."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPolygonF)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..dsp.biquad import Band, magnitude_db
from .theme import COLORS


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}:{s:05.2f}" if m else f"{s:.2f}s"


# ---------------------------------------------------------------------------
# waveform
# ---------------------------------------------------------------------------


class WaveformWidget(QWidget):
    """Overlaid input/processed waveforms with click-to-seek and drag-select."""

    seekRequested = Signal(float)
    selectionChanged = Signal(object)   # (start_s, end_s) or None
    viewChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self._orig: np.ndarray | None = None
        self._proc: np.ndarray | None = None
        self._sr = 44100
        self._dur = 1.0
        self._t0 = 0.0
        self._t1 = 1.0
        self._sel: tuple[float, float] | None = None
        self._playhead = 0.0
        self._drag_anchor: float | None = None
        self._dragging = False
        self._mindrag = 3
        self._show_input = True

    # -- data -------------------------------------------------------------
    def set_audio(self, orig: np.ndarray | None, proc: np.ndarray | None, sr: int):
        self._orig = orig
        self._proc = proc
        self._sr = int(sr) if sr else 44100
        dur = 0.0
        for arr in (orig, proc):
            if arr is not None and len(arr):
                dur = max(dur, len(arr) / self._sr)
        self._dur = max(dur, 0.001)
        self._t0, self._t1 = 0.0, self._dur
        self.update()

    def set_processed(self, proc: np.ndarray | None):
        if proc is not None and self._orig is not None and len(proc) != len(self._orig):
            # keep the widget honest about length rather than silently stretching
            self._proc = proc
        else:
            self._proc = proc
        self.update()

    def show_input(self, on: bool):
        self._show_input = bool(on)
        self.update()

    @property
    def duration(self) -> float:
        return self._dur

    def selection(self) -> tuple[float, float] | None:
        return self._sel

    def set_selection(self, sel: tuple[float, float] | None):
        if sel is not None:
            a, b = sorted(sel)
            a = max(0.0, a)
            b = min(self._dur, b)
            sel = (a, b) if b - a > 0.01 else None
        self._sel = sel
        self.selectionChanged.emit(sel)
        self.update()

    def set_playhead(self, t: float):
        t = float(np.clip(t, 0.0, self._dur))
        if abs(t - self._playhead) > 1e-4:
            self._playhead = t
            self.update()

    def playhead(self) -> float:
        return self._playhead

    def fit(self):
        self._t0, self._t1 = 0.0, self._dur
        self.viewChanged.emit(self._t0, self._t1)
        self.update()

    def set_view(self, t0: float, t1: float):
        span = max(0.02, t1 - t0)
        span = min(span, self._dur)
        t0 = float(np.clip(t0, 0.0, max(0.0, self._dur - span)))
        self._t0, self._t1 = t0, t0 + span
        self.viewChanged.emit(self._t0, self._t1)
        self.update()

    def zoom(self, factor: float, center: float | None = None):
        c = self._playhead if center is None else center
        span = (self._t1 - self._t0) / max(factor, 1e-6)
        span = float(np.clip(span, 0.05, self._dur))
        self.set_view(c - span / 2, c + span / 2)

    # -- geometry ---------------------------------------------------------
    def _plot_rect(self) -> QRectF:
        r = self.rect()
        return QRectF(10, 18, max(10, r.width() - 20), max(10, r.height() - 34))

    def _time_to_x(self, t: float) -> float:
        r = self._plot_rect()
        span = max(self._t1 - self._t0, 1e-9)
        return r.left() + (t - self._t0) / span * r.width()

    def _x_to_time(self, x: float) -> float:
        r = self._plot_rect()
        span = max(self._t1 - self._t0, 1e-9)
        return float(np.clip(self._t0 + (x - r.left()) / r.width() * span, 0.0, self._dur))

    def _envelope(self, x: np.ndarray, width: int) -> np.ndarray:
        n = x.shape[0]
        i0 = int(max(0, self._t0 * self._sr))
        i1 = int(min(n, self._t1 * self._sr))
        if i1 <= i0 or width < 2:
            return np.zeros(max(2, width))
        seg = x[i0:i1]
        mono = np.abs(seg).max(axis=1) if seg.ndim > 1 else np.abs(seg)
        if mono.size <= width:
            return np.interp(np.linspace(0, mono.size - 1, width), np.arange(mono.size), mono)
        edges = np.linspace(0, mono.size, width + 1).astype(np.int64)
        edges = np.unique(np.clip(edges, 0, mono.size - 1))
        if edges.size < 2:
            return np.zeros(width)
        peaks = np.maximum.reduceat(mono, edges[:-1])
        if peaks.size != width:
            peaks = np.interp(np.linspace(0, 1, width), np.linspace(0, 1, peaks.size), peaks)
        return peaks

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(COLORS["panel"]))
        r = self._plot_rect()
        mid = r.center().y()

        # grid
        p.setPen(QPen(QColor(COLORS["grid"]), 1))
        p.drawLine(QPointF(r.left(), mid), QPointF(r.right(), mid))
        for frac in (0.25, 0.75):
            y = r.top() + r.height() * frac
            p.setPen(QPen(QColor(COLORS["grid"]), 1, Qt.DotLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        # time ruler
        p.setPen(QPen(QColor(COLORS["border"]), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        span = max(self._t1 - self._t0, 1e-9)
        step = self._nice_step(span / max(1.0, r.width() / 90.0))
        t = math.ceil(self._t0 / step) * step
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        p.setFont(font)
        while t <= self._t1:
            x = self._time_to_x(t)
            p.setPen(QPen(QColor(COLORS["border"]), 1))
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 4))
            p.setPen(QColor(COLORS["text_faint"]))
            p.drawText(QRectF(x - 30, r.bottom() + 4, 60, 14),
                       Qt.AlignHCenter | Qt.AlignTop, _fmt_time(t))
            t += step

        if self._orig is None and self._proc is None:
            p.setPen(QColor(COLORS["text_faint"]))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Drop an audio file here, or use File \u203a Import\u2026")
            p.end()
            return

        width = max(2, int(r.width()))
        p.save()
        p.setClipRect(r)

        if self._show_input and self._orig is not None:
            env = self._envelope(self._orig, width)
            self._draw_shape(p, r, env, QColor(COLORS["wave_in"]))

        if self._proc is not None:
            env = self._envelope(self._proc, width)
            col = QColor(COLORS["wave_out"])
            col.setAlpha(235)
            self._draw_shape(p, r, env, col)

        # selection
        if self._sel is not None:
            x0, x1 = self._time_to_x(self._sel[0]), self._time_to_x(self._sel[1])
            overlay = QColor(COLORS["accent"])
            overlay.setAlpha(28)
            p.fillRect(QRectF(x0, r.top(), max(1.0, x1 - x0), r.height()), overlay)
            p.setPen(QPen(QColor(COLORS["accent_dim"]), 1))
            p.drawLine(QPointF(x0, r.top()), QPointF(x0, r.bottom()))
            p.drawLine(QPointF(x1, r.top()), QPointF(x1, r.bottom()))

        # playhead
        if self._t0 <= self._playhead <= self._t1:
            x = self._time_to_x(self._playhead)
            p.setPen(QPen(QColor(COLORS["accent2"]), 1.5))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.restore()

        # labels
        p.setPen(QColor(COLORS["text_faint"]))
        p.setFont(font)
        p.drawText(QRectF(r.left(), 2, 200, 14), Qt.AlignLeft | Qt.AlignVCenter,
                   "OUTPUT" if self._proc is not None else "")
        if self._show_input and self._orig is not None:
            p.drawText(QRectF(r.left() + 60, 2, 200, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       "INPUT")
        p.end()

    def _draw_shape(self, p: QPainter, r: QRectF, env: np.ndarray, color: QColor):
        n = env.size
        half = r.height() / 2.0
        mid = r.center().y()
        top = QPolygonF()
        bottom = QPolygonF()
        xs = np.linspace(r.left(), r.right(), n)
        amp = np.clip(env, 0.0, 1.2)
        ys = mid - amp * half * 0.94
        for x, y in zip(xs, ys):
            top.append(QPointF(float(x), float(y)))
        for x, y in zip(xs[::-1], (mid + (mid - ys))[::-1]):
            bottom.append(QPointF(float(x), float(y)))
        path = QPainterPath()
        path.addPolygon(top)
        path.addPolygon(bottom)
        path.closeSubpath()
        p.setPen(QPen(color.darker(115), 0.6))
        p.setBrush(QBrush(color))
        p.drawPath(path)

    @staticmethod
    def _nice_step(target: float) -> float:
        if target <= 0:
            return 1.0
        exp = math.floor(math.log10(target))
        base = target / (10 ** exp)
        for m in (1, 2, 5, 10):
            if base <= m:
                return m * 10 ** exp
        return 10 ** (exp + 1)

    # -- interaction ------------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        t = self._x_to_time(event.position().x())
        self._drag_anchor = t
        self._dragging = False
        self._press_x = event.position().x()
        if event.modifiers() & Qt.ShiftModifier and self._sel is not None:
            self.set_selection((self._sel[0], t))
            self._drag_anchor = self._sel[0]

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_anchor is None:
            return
        if abs(event.position().x() - self._press_x) > self._mindrag:
            self._dragging = True
        if self._dragging:
            self.set_selection((self._drag_anchor, self._x_to_time(event.position().x())))

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_anchor is None:
            return
        if not self._dragging:
            self.seekRequested.emit(float(self._drag_anchor))
        self._drag_anchor = None
        self._dragging = False

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.set_selection(None)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.zoom(factor, self._x_to_time(event.position().x()))
        else:
            span = self._t1 - self._t0
            shift = span * (0.12 if event.angleDelta().y() < 0 else -0.12)
            self.set_view(self._t0 + shift, self._t1 + shift)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_F:
            self.fit()
        elif event.key() == Qt.Key_Escape:
            self.set_selection(None)
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# level meter
# ---------------------------------------------------------------------------


class LevelMeter(QWidget):
    """Horizontal stereo meter with RMS bar, peak marker and ceiling line."""

    DB_MIN = -48.0
    DB_MAX = 0.0

    def __init__(self, channels: int = 2, parent=None):
        super().__init__(parent)
        self._channels = channels
        self._vals = [(self.DB_MIN, self.DB_MIN, self.DB_MIN)] * channels  # (rms, peak, true peak)
        self._hold = [self.DB_MIN] * channels
        self.setMinimumHeight(14 + 22 * channels)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_levels(self, values: Sequence[tuple[float, float, float]]):
        self._vals = list(values)[: self._channels]
        for i, v in enumerate(self._vals):
            self._hold[i] = max(self._hold[i], v[1])
        self.update()

    def clear_hold(self):
        self._hold = [self.DB_MIN] * self._channels
        self.update()

    def _x_of(self, dbv: float, r: QRectF) -> float:
        f = (dbv - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        return r.left() + float(np.clip(f, 0.0, 1.0)) * r.width()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(COLORS["panel"]))
        bar_h = 14
        pad = 6
        width = self.width() - 2 * pad - 10
        top = 4
        for i in range(self._channels):
            r = QRectF(pad + 10, top + i * (bar_h + 4), width, bar_h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(COLORS["panel_alt"]))
            p.drawRoundedRect(r, 3, 3)
            rms_db, peak_db, tp_db = self._vals[i]
            # RMS bar
            x0 = r.left()
            x1 = self._x_of(rms_db, r)
            grad = QLinearGradient(r.left(), 0, r.right(), 0)
            grad.setColorAt(0.0, QColor("#3d8a6b"))
            grad.setColorAt(0.72, QColor(COLORS["ok"]))
            grad.setColorAt(0.90, QColor(COLORS["warn"]))
            grad.setColorAt(1.0, QColor(COLORS["danger"]))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(x0, r.top(), max(2.0, x1 - x0), r.height()), 3, 3)
            # peak marker
            px = self._x_of(peak_db, r)
            p.setPen(QPen(QColor(COLORS["text"]), 1.5))
            p.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
            # true-peak marker
            if tp_db > peak_db + 0.05:
                qx = self._x_of(tp_db, r)
                p.setPen(QPen(QColor(COLORS["accent2"]), 1.0, Qt.DotLine))
                p.drawLine(QPointF(qx, r.top() - 1), QPointF(qx, r.bottom() + 1))
            # hold
            hx = self._x_of(self._hold[i], r)
            p.setPen(QPen(QColor(COLORS["text_faint"]), 1))
            p.drawLine(QPointF(hx, r.top() + 1), QPointF(hx, r.bottom() - 1))
            f = QFont(self.font())
            f.setPointSizeF(max(7.0, f.pointSizeF() - 2))
            p.setFont(f)
            p.setPen(QColor(COLORS["text_faint"]))
            label = "L" if self._channels > 1 and i == 0 else ("R" if i == 1 else "")
            p.drawText(QRectF(0, r.top(), pad - 1, bar_h), Qt.AlignRight | Qt.AlignVCenter,
                       label)
        # scale
        scale_y = top + self._channels * (bar_h + 4) + 2
        p.setPen(QColor(COLORS["text_faint"]))
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 2))
        p.setFont(f)
        ruler = QRectF(pad, scale_y, width, 12)
        for dbv in (-40, -24, -12, -6, -3, -1):
            x = self._x_of(dbv, ruler)
            p.setPen(QPen(QColor(COLORS["border"]), 1))
            p.drawLine(QPointF(x, scale_y), QPointF(x, scale_y + 3))
            p.setPen(QColor(COLORS["text_faint"]))
            p.drawText(QRectF(x - 14, scale_y + 3, 28, 10),
                       Qt.AlignHCenter | Qt.AlignTop, f"{dbv}")
        p.end()


# ---------------------------------------------------------------------------
# EQ curve
# ---------------------------------------------------------------------------


class EqCurveWidget(QWidget):
    DB_RANGE = 18.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._bands: list[Band] = []
        self._sr = 44100
        self._freqs = np.logspace(math.log10(20), math.log10(20000), 320)
        self._curve = np.zeros_like(self._freqs)
        self._target: tuple[np.ndarray, np.ndarray] | None = None
        self._title = "Static EQ"

    def set_bands(self, bands: Iterable[Band], sr: int):
        self._bands = list(bands)
        self._sr = int(sr)
        self._curve = magnitude_db(self._bands, self._freqs, self._sr)
        self.update()

    def set_target(self, freqs: np.ndarray | None, db: np.ndarray | None,
                   title: str = "Reference (apparent)"):
        if freqs is None or db is None:
            self._target = None
        else:
            self._target = (np.asarray(freqs), np.asarray(db))
            self._title = title
        self.update()

    def _xy(self, f: np.ndarray, mag_db: np.ndarray, r: QRectF):
        lx = (np.log10(np.maximum(f, 1.0)) - math.log10(20)) / (math.log10(20000) - math.log10(20))
        x = r.left() + lx * r.width()
        y = r.center().y() - (mag_db / self.DB_RANGE) * (r.height() / 2.0)
        return x, y

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(COLORS["panel_alt"]))
        r = QRectF(8, 12, self.width() - 16, self.height() - 24)
        p.setPen(QPen(QColor(COLORS["grid"]), 1))
        for f in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
            x, _ = self._xy(np.array([f]), np.zeros(1), r)
            p.setPen(QPen(QColor(COLORS["grid"]), 1, Qt.DotLine))
            p.drawLine(QPointF(float(x[0]), r.top()), QPointF(float(x[0]), r.bottom()))
            p.setPen(QColor(COLORS["text_faint"]))
            fnt = QFont(self.font())
            fnt.setPointSizeF(max(6.5, fnt.pointSizeF() - 3))
            p.setFont(fnt)
            label = f"{f // 1000}k" if f >= 1000 else f"{f}"
            p.drawText(QRectF(float(x[0]) - 18, r.bottom() + 1, 36, 12),
                       Qt.AlignHCenter | Qt.AlignTop, label)
        for dbv in (-12, -6, 0, 6, 12):
            y = r.center().y() - (dbv / self.DB_RANGE) * (r.height() / 2)
            p.setPen(QPen(QColor(COLORS["border"] if dbv == 0 else COLORS["grid"]),
                          1, Qt.SolidLine if dbv == 0 else Qt.DotLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(COLORS["text_faint"]))
            p.drawText(QRectF(r.left() + 2, y - 12, 40, 11), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{dbv:+d}" if dbv else "0")

        if self._target is not None:
            tf, td = self._target
            xs, ys = self._xy(tf, td, r)
            path = QPainterPath()
            path.moveTo(float(xs[0]), float(ys[0]))
            for x, y in zip(xs[1:], ys[1:]):
                path.lineTo(float(x), float(y))
            p.setPen(QPen(QColor(COLORS["accent2"]), 1.4, Qt.DashLine))
            p.drawPath(path)

        xs, ys = self._xy(self._freqs, self._curve, r)
        path = QPainterPath()
        path.moveTo(float(xs[0]), float(ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(float(x), float(y))
        # fill to 0 dB
        fill = QPainterPath(path)
        zero_y = r.center().y()
        fill.lineTo(float(xs[-1]), zero_y)
        fill.lineTo(float(xs[0]), zero_y)
        fill.closeSubpath()
        col = QColor(COLORS["accent"])
        col.setAlpha(38)
        p.fillPath(fill, col)
        p.setPen(QPen(QColor(COLORS["accent"]), 1.8))
        p.drawPath(path)

        # band handles
        for b in self._bands:
            if not b.is_active():
                continue
            x, y = self._xy(np.array([b.freq]), np.array([b.gain_db]), r)
            p.setPen(QPen(QColor(COLORS["text"]), 1))
            p.setBrush(QColor(COLORS["panel"]))
            p.drawEllipse(QPointF(float(x[0]), float(y[0])), 3.2, 3.2)
        p.end()
