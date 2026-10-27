"""Export options dialog (format, rate, quality, destination)."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)

from ..dsp.audio import AudioData, ExportSettings, format_for_extension
from .controls import _label
from .theme import COLORS

#: (label, format, subtype, extension, kind)
FORMATS = [
    ("WAV \u00b7 24-bit PCM", "WAV", "PCM_24", ".wav", "pcm"),
    ("WAV \u00b7 16-bit PCM", "WAV", "PCM_16", ".wav", "pcm16"),
    ("WAV \u00b7 32-bit float", "WAV", "FLOAT", ".wav", "pcm"),
    ("FLAC \u00b7 24-bit", "FLAC", "PCM_24", ".flac", "pcm"),
    ("FLAC \u00b7 16-bit", "FLAC", "PCM_16", ".flac", "pcm16"),
    ("Ogg Vorbis", "OGG", "VORBIS", ".ogg", "vorbis"),
    ("MP3 \u00b7 320 kbps", "MP3", "MPEG_LAYER_III", ".mp3", "mp3"),
]


class ExportDialog(QDialog):
    def __init__(self, audio: AudioData, directory: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export processed audio")
        self.setMinimumWidth(520)
        self._audio = audio
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        head = QLabel("Export")
        head.setObjectName("Title")
        lay.addWidget(head)
        self.summary = QLabel("")
        self.summary.setObjectName("Faint")
        lay.addWidget(self.summary)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        self.format = QComboBox()
        for label, *_ in FORMATS:
            self.format.addItem(label)
        self.format.setCurrentIndex(0)
        self.format.currentIndexChanged.connect(self._update_quality_visibility)
        form.addRow(_label("Format", width=110), self.format)

        self.rate = QComboBox()
        self.rate.addItem(f"Keep source rate ({audio.sample_rate} Hz)", 0)
        for sr in (44100, 48000, 96000):
            if sr != audio.sample_rate:
                self.rate.addItem(f"{sr} Hz", sr)
        form.addRow(_label("Sample rate", width=110), self.rate)
        note = QLabel("Rate conversion is band-limited and preserves duration, so pitch "
                      "and tempo do not change.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        form.addRow("", note)

        self.quality_box = QWidget()
        ql = QHBoxLayout(self.quality_box)
        ql.setContentsMargins(0, 0, 0, 0)
        self.vorbis_quality = QComboBox()
        for v, label in ((0.4, "q4 \u00b7 ~128 kbps"), (0.6, "q6 \u00b7 ~192 kbps"),
                         (0.8, "q8 \u00b7 ~256 kbps"), (0.95, "q9 \u00b7 ~320 kbps")):
            self.vorbis_quality.addItem(label, v)
        self.vorbis_quality.setCurrentIndex(2)
        ql.addWidget(self.vorbis_quality)
        form.addRow(_label("Vorbis quality", width=110), self.quality_box)

        self.dither = QCheckBox("TPDF dither (16-bit targets only)")
        form.addRow("", self.dither)

        path_row = QWidget()
        pl = QHBoxLayout(path_row)
        pl.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit()
        base = os.path.splitext(os.path.basename(audio.path or "audio"))[0]
        directory = directory or (os.path.dirname(audio.path or "") or os.getcwd())
        self.path_edit.setText(os.path.join(directory, f"{base}_timbreforge.wav"))
        browse = QPushButton("Browse\u2026")
        browse.clicked.connect(self._browse)
        pl.addWidget(self.path_edit, 1)
        pl.addWidget(browse)
        form.addRow(_label("Destination", width=110), path_row)
        lay.addLayout(form)

        self.warn = QLabel("")
        self.warn.setObjectName("Faint")
        self.warn.setWordWrap(True)
        lay.addWidget(self.warn)

        buttons = QDialogButtonBox()
        self.ok = buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        self.ok.setObjectName("Primary")
        buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._update_summary()
        self._update_quality_visibility()

    def _update_summary(self):
        self.summary.setText(
            f"{self._audio.filename} \u00b7 {self._audio.sample_rate} Hz \u00b7 "
            f"{self._audio.channels} ch \u00b7 {self._audio.duration:.2f} s")

    def _update_quality_visibility(self):
        kind = FORMATS[self.format.currentIndex()][4]
        self.quality_box.setVisible(kind == "vorbis")
        self.dither.setVisible(kind == "pcm16")
        ext = FORMATS[self.format.currentIndex()][3]
        current = self.path_edit.text()
        if current:
            root, _ = os.path.splitext(current)
            self.path_edit.setText(root + ext)
        if kind == "mp3":
            self.warn.setText("MP3 encoding uses the libsndfile default bitrate; the exact "
                              "bitrate of the encoder is not controlled here.")
        elif kind == "vorbis":
            self.warn.setText("Vorbis is lossy. Validate the encoded file, not just the "
                              "rendered float buffer.")
        else:
            self.warn.setText("")

    def _browse(self):
        current = self.path_edit.text()
        ext = FORMATS[self.format.currentIndex()][3]
        filt = {".wav": "WAV (*.wav)", ".flac": "FLAC (*.flac)",
                ".ogg": "Ogg Vorbis (*.ogg)", ".mp3": "MP3 (*.mp3)"}[ext]
        path, _ = QFileDialog.getSaveFileName(self, "Save processed audio", current, filt)
        if path:
            if not path.lower().endswith(ext):
                path = os.path.splitext(path)[0] + ext
            self.path_edit.setText(path)

    def accept(self):  # noqa: D102
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No destination", "Choose where to save the file.")
            return
        try:
            format_for_extension(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Unsupported extension", str(exc))
            return
        if not os.path.isdir(os.path.dirname(os.path.abspath(path)) or "."):
            QMessageBox.warning(self, "Folder missing",
                                "The destination folder does not exist.")
            return
        if self._audio.path and os.path.abspath(path) == os.path.abspath(self._audio.path):
            QMessageBox.warning(self, "Protected file",
                                "The destination is the imported file. Choose another name "
                                "so the original is never overwritten.")
            return
        if os.path.exists(path):
            resp = QMessageBox.question(
                self, "Overwrite file?",
                f"'{os.path.basename(path)}' already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if resp != QMessageBox.Yes:
                return
        super().accept()

    def settings(self) -> ExportSettings:
        _, fmt, subtype, _ext, kind = FORMATS[self.format.currentIndex()]
        rate = int(self.rate.currentData()) or None
        return ExportSettings(
            path=self.path_edit.text().strip(),
            format=fmt,
            subtype=subtype,
            sample_rate=rate,
            ogg_quality=float(self.vorbis_quality.currentData() or 0.8),
            dither=bool(self.dither.isChecked() and kind == "pcm16"),
            overwrite=True,
            source_path=self._audio.path,
        )
