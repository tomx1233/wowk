"""Application entry point."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from .. import APP_NAME, APP_SUBTITLE, __version__
from .main_window import MainWindow
from .theme import COLORS, apply_theme


def _make_icon() -> QIcon:
    """Draw the app icon (no external asset needed)."""
    pm = QPixmap(64, 64)
    pm.fill(QColor(COLORS["panel"]))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(COLORS["accent"]))
    heights = [10, 20, 34, 46, 28, 40, 22, 14]
    for i, h in enumerate(heights):
        x = 8 + i * 6
        p.drawRoundedRect(x, 32 - h / 2, 4, h, 2, 2)
    p.setPen(QPen(QColor(COLORS["accent2"]), 1.5))
    p.drawLine(6, 32, 58, 32)
    p.end()
    return QIcon(pm)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:  # noqa: BLE001 - older Qt
        pass
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(f"{APP_NAME} \u2014 {APP_SUBTITLE}")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Timbreforge")
    apply_theme(app)
    icon = _make_icon()
    app.setWindowIcon(icon)

    win = MainWindow()
    win.setWindowIcon(icon)
    win.show()

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        if os.path.isfile(arg):
            win.load_file(arg)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
