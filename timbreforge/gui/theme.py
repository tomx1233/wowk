"""Dark studio theme for Timbreforge."""

from __future__ import annotations

COLORS = {
    "bg": "#14161a",
    "panel": "#1b1f26",
    "panel_alt": "#222833",
    "panel_hi": "#28303c",
    "border": "#2c333f",
    "border_hi": "#3b4553",
    "text": "#e6eaf0",
    "text_dim": "#9aa6b6",
    "text_faint": "#6b7686",
    "accent": "#57c8d8",
    "accent_dim": "#2f7f8b",
    "accent2": "#f0a45c",
    "ok": "#6ecf97",
    "warn": "#f0b95c",
    "danger": "#e5695b",
    "wave_in": "#4d5766",
    "wave_out": "#57c8d8",
    "grid": "#242a34",
}

STYLESHEET = f"""
* {{
    font-family: "Inter", "Segoe UI", "Helvetica Neue", "Noto Sans", sans-serif;
    font-size: 12px;
}}
QWidget#Root {{ background: {COLORS['bg']}; }}
QMainWindow, QDialog {{ background: {COLORS['bg']}; }}
QWidget {{ color: {COLORS['text']}; }}

QFrame#Panel, QWidget#Panel {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}
QFrame#Card {{
    background: {COLORS['panel_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
}}
QLabel#Title {{ font-size: 15px; font-weight: 600; color: {COLORS['text']}; }}
QLabel#Subtle {{ color: {COLORS['text_dim']}; }}
QLabel#Faint {{ color: {COLORS['text_faint']}; font-size: 11px; }}
QLabel#SectionTitle {{
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    color: {COLORS['text_dim']}; text-transform: uppercase;
}}
QLabel#Value {{ font-family: "JetBrains Mono", "SF Mono", Consolas, monospace; }}
QLabel#Mono {{ font-family: "JetBrains Mono", "SF Mono", Consolas, monospace;
               color: {COLORS['text_dim']}; font-size: 11px; }}
QLabel#Accent {{ color: {COLORS['accent']}; }}
QLabel#Chip {{
    background: {COLORS['panel_hi']}; border: 1px solid {COLORS['border_hi']};
    border-radius: 9px; padding: 1px 8px; color: {COLORS['text_dim']}; font-size: 11px;
}}

QPushButton {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border_hi']};
    border-radius: 6px; padding: 6px 12px; color: {COLORS['text']};
}}
QPushButton:hover {{ background: {COLORS['panel_hi']}; }}
QPushButton:pressed {{ background: {COLORS['border']}; }}
QPushButton:disabled {{ color: {COLORS['text_faint']}; border-color: {COLORS['border']}; }}
QPushButton#Primary {{
    background: {COLORS['accent']}; color: #08262b; border: 1px solid {COLORS['accent']};
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #6ed6e4; }}
QPushButton#Primary:disabled {{ background: {COLORS['accent_dim']}; border-color: {COLORS['accent_dim']};
                               color: #0d3338; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid {COLORS['border']}; }}
QPushButton#Danger {{ color: {COLORS['danger']}; border-color: #4a2d2b; }}
QPushButton#Toggle:checked {{
    background: {COLORS['accent_dim']}; border-color: {COLORS['accent']};
    color: {COLORS['text']};
}}
QPushButton#Small {{ padding: 3px 8px; font-size: 11px; }}

QGroupBox {{
    background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
    border-radius: 8px; margin-top: 14px; padding: 10px 10px 8px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 6px; color: {COLORS['text_dim']};
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border_hi']};
    border-radius: 5px; padding: 3px 6px; selection-background-color: {COLORS['accent_dim']};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: #47525f; }}
QComboBox QAbstractItemView {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border_hi']};
    selection-background-color: {COLORS['accent_dim']};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{ width: 12px; background: transparent; }}

QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {COLORS['border_hi']}; background: {COLORS['panel_alt']};
}}
QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}

QSlider::groove:horizontal {{
    height: 4px; background: {COLORS['border']}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {COLORS['accent_dim']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
    background: {COLORS['text']}; border: 1px solid {COLORS['border_hi']};
}}
QSlider::handle:horizontal:hover {{ background: {COLORS['accent']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {COLORS['border_hi']}; border-radius: 4px; min-height: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {COLORS['border_hi']}; border-radius: 4px; min-width: 24px;
}}

QProgressBar {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border']};
    border-radius: 4px; height: 6px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 3px; }}

QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 8px;
                    background: {COLORS['panel']}; }}
QTabBar::tab {{
    background: transparent; color: {COLORS['text_dim']}; padding: 6px 14px;
    border: 1px solid transparent; border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ color: {COLORS['text']}; background: {COLORS['panel']};
                          border-color: {COLORS['border']}; border-bottom-color: {COLORS['panel']}; }}
QTabBar::tab:hover {{ color: {COLORS['text']}; }}

QMenuBar {{ background: {COLORS['panel']}; color: {COLORS['text']};
            border-bottom: 1px solid {COLORS['border']}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; color: {COLORS['text']}; }}
QMenuBar::item:selected {{ background: {COLORS['panel_hi']}; border-radius: 4px; }}
QMenu {{ background: {COLORS['panel_alt']}; color: {COLORS['text']};
         border: 1px solid {COLORS['border_hi']}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 18px; border-radius: 4px; color: {COLORS['text']}; }}
QMenu::item:selected {{ background: {COLORS['accent_dim']}; }}
QMenu::separator {{ height: 1px; background: {COLORS['border']}; margin: 4px 8px; }}

QToolTip {{
    background: {COLORS['panel_hi']}; color: {COLORS['text']};
    border: 1px solid {COLORS['border_hi']}; padding: 4px;
}}

QSplitter::handle {{ background: {COLORS['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QTextEdit, QPlainTextEdit {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-family: "JetBrains Mono", "SF Mono", Consolas, monospace; font-size: 11px;
}}
QTableWidget {{
    background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']}; border-radius: 6px;
}}
QHeaderView::section {{
    background: {COLORS['panel_hi']}; color: {COLORS['text_dim']};
    border: none; padding: 4px; font-size: 11px;
}}
QStatusBar {{ background: {COLORS['panel']}; border-top: 1px solid {COLORS['border']};
              color: {COLORS['text_dim']}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 5px;
               padding: 3px; }}
QToolButton:hover {{ background: {COLORS['panel_hi']}; }}
QMessageBox {{ background: {COLORS['panel']}; }}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(STYLESHEET)
    try:
        from PySide6.QtGui import QColor, QPalette
        pal = app.palette()
        pal.setColor(QPalette.Window, QColor(COLORS["bg"]))
        pal.setColor(QPalette.WindowText, QColor(COLORS["text"]))
        pal.setColor(QPalette.Base, QColor(COLORS["panel_alt"]))
        pal.setColor(QPalette.Text, QColor(COLORS["text"]))
        pal.setColor(QPalette.Button, QColor(COLORS["panel_alt"]))
        pal.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
        pal.setColor(QPalette.Highlight, QColor(COLORS["accent_dim"]))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        app.setPalette(pal)
    except Exception:  # pragma: no cover - palette is cosmetic
        pass
