"""Pitch / tempo of two files: a read-only measurement dialog.

The app deliberately has no pitch or tempo control in its processing chain.  This
dialog is the exception that proves the rule: it does not *apply* anything, it
*discovers* what somebody else applied to a pair of files and prints the Audacity
settings that reproduce it.

It exists because the answer to "what did you do to this song?" is easy to lose:
it was originally a command-line tool (``tools/pair_match.py``) and a report in
``docs/PAIR_MATCH.md``, which is not where anybody looks while the app is open.
"""

from __future__ import annotations

import os
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QGridLayout,
                               QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QVBoxLayout)

from ..dsp.audio import IMPORT_FILTER
from ..dsp.pairmatch import Cancelled, macro_text, measure_pair


class PairWorker(QThread):
    """Runs the measurement off the UI thread so the window stays usable."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, original: str, modified: str, parent=None, *,
                 verify: bool = True):
        super().__init__(parent)
        self._original = original
        self._modified = modified
        self._verify = bool(verify)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):  # noqa: D401
        try:
            report = measure_pair(
                self._original, self._modified,
                progress=lambda f, l: self.progress.emit(float(f), l),
                should_cancel=lambda: self._cancel,
                selftest=self._verify, chain_control=self._verify)
            if self._cancel:
                self.cancelled.emit()
                return
            self.done.emit(report)
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


# ---------------------------------------------------------------------------
# report text
# ---------------------------------------------------------------------------
def macro_for(report: dict) -> str:
    p = report["pitch"]["percent"]
    t = report["tempo"]["percent"]
    return macro_text(p, t)


def format_report(r: dict) -> str:
    """Human-readable report.  Every number here was measured, not assumed."""
    o, m = r["original"], r["modified"]
    p, t = r["pitch"], r["tempo"]
    span, bw = r["content_span"], r["bandwidth"]
    L = []
    L.append("FILES")
    L.append(f"  original  {o['path']}  {o['sample_rate']} Hz  {o['channels']} ch  "
             f"{o['duration']:.3f} s  ({o['codec']}, {o['bitrate_kbps']:.0f} kbps)")
    L.append(f"  modified  {m['path']}  {m['sample_rate']} Hz  {m['channels']} ch  "
             f"{m['duration']:.3f} s  ({m['codec']}, {m['bitrate_kbps']:.0f} kbps)")
    L.append("")
    L.append("WHAT CHANGED")
    L.append(f"  pitch   {p['ratio']:.6f}  ({p['median_cents']:+.2f} cents, "
             f"{p['percent']:+.3f} %), spread over {p['windows']} windows "
             f"{p['spread_cents']:.2f} cents")
    L.append(f"  speed   {t['ratio']:.6f}  ({t['percent']:+.3f} %), correlation "
             f"{t['corr']:.3f}, constant offset {t['offset_ms']:+.1f} ms, "
             f"quarter lags {t['quarter_lags']}")
    L.append("")
    L.append("AUDACITY")
    for line in macro_for(r).strip().splitlines():
        L.append("  " + line)
    L.append("  (Tools > Macros > New, then Import a file with these lines, or use"
             " Insert; apply to the whole track)")
    L.append("")
    L.append("LENGTH")
    L.append(f"  audio span  {span['original']['start_s']:.3f}-{span['original']['end_s']:.3f} s"
             f"  ->  {span['modified']['start_s']:.3f}-{span['modified']['end_s']:.3f} s")
    L.append(f"  span ratio  {span['span_ratio']:.6f}   (file-length ratio "
             f"{r['duration_ratio']:.6f} - silence is excluded here)")
    L.append("")
    L.append("LEVEL AND TONE AFTER REMOVING THE PITCH SHIFT")
    lv = r.get("level") or {}
    if lv:
        L.append(f"  level change {lv['median_db']:+.2f} dB median "
                 f"(p5..p95 {lv['p5_db']:+.2f}..{lv['p95_db']:+.2f} dB, "
                 f"sd {lv['std_db']:.2f} dB over {lv['windows']} windows)")
        L.append(f"  level-vs-level slope {lv['slope_vs_level']:+.2f} "
                 f"(near 0 = no compression, clearly negative = level dependent)")
    tone = r["tone"]
    L.append(f"  band differences, normalised to 1-4 kHz (total offset "
             f"{tone['level_offset_db']:+.2f} dB, range {tone['range_db']:.2f} dB):")
    row = "    "
    for c, d in zip(tone["centres"], tone["normalised_db"]):
        row += f"{c:.0f} Hz {d:+.1f}  "
        if len(row) > 90:
            L.append(row)
            row = "    "
    if row.strip():
        L.append(row)
    warn = bandwidth_warning(r)
    if warn:
        L.append("")
        L.append("SOURCE BANDWIDTH")
        L.append("  " + warn)
    L.append("")
    L.append("CHECKS")
    for row in r.get("selftest") or []:
        L.append(f"  control {row['true_ratio']:.6f} -> measured "
                 f"{row['measured_ratio']:.6f} ({row['error_ppm']:+d} ppm)")
    ch = r.get("selftest_through_stretch")
    if ch:
        L.append(f"  control through a time-stretch {ch['true_ratio']:.6f} -> "
                 f"{ch['measured_ratio']:.6f} ({ch['error_ppm']:+d} ppm)")
    L.append("")
    L.append("CAVEATS")
    L.append("  - This measures the pair you supplied. It is not a claim that the")
    L.append("    original processing chain has been identified.")
    L.append("  - A single speed change and a separate pitch+tempo change can be")
    L.append("    indistinguishable when the two percentages are close.")
    L.append("  - Nothing here is applied to the audio this app exports.")
    return "\n".join(L)


def bandwidth_warning(r: dict) -> str:
    """Explain a top-octave difference that no pitch/tempo/gain stage can make."""
    bw = r.get("bandwidth") or {}
    eo = ((bw.get("original") or {}).get("edge_hz"))
    em = ((bw.get("modified") or {}).get("edge_hz"))
    if not eo or not em:
        return ""
    if em > eo + 800:
        return (f"the modified file carries real content up to about {em} Hz while the "
                f"original stops at about {eo} Hz, so part of the difference is a "
                f"source/encode difference rather than an effect.")
    if eo > em + 800:
        return (f"the original carries content up to about {eo} Hz and the modified "
                f"file only to about {em} Hz, i.e. the modified file is the "
                f"band-limited one.")
    return ""


# ---------------------------------------------------------------------------
# dialog
# ---------------------------------------------------------------------------
class PairDialog(QDialog):
    def __init__(self, parent=None, original: str = "", loaded: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Pitch / tempo of two files")
        self.resize(940, 720)
        self._original = original or ""
        self._modified = ""
        self._report: dict | None = None
        self._worker: PairWorker | None = None
        self._loaded = loaded or ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        head = QLabel("Pitch / tempo of two files")
        head.setObjectName("Title")
        lay.addWidget(head)
        sub = QLabel(
            "Measures what pitch and tempo change turned one file into another and "
            "prints the Audacity settings that reproduce it. This only measures: it "
            "never touches the audio this app exports, and the processing chain has no "
            "pitch or tempo control.")
        sub.setObjectName("Faint")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.orig_label = QLabel("(none)")
        self.mod_label = QLabel("(none)")
        for lab in (self.orig_label, self.mod_label):
            lab.setObjectName("Mono")
        b1 = QPushButton("Choose\u2026")
        b2 = QPushButton("Choose\u2026")
        b1.clicked.connect(lambda: self._pick("original"))
        b2.clicked.connect(lambda: self._pick("modified"))
        self.use_loaded_btn = QPushButton("Use the file I have open")
        self.use_loaded_btn.setObjectName("Ghost")
        self.use_loaded_btn.setEnabled(bool(self._loaded))
        self.use_loaded_btn.clicked.connect(self._use_loaded)
        grid.addWidget(QLabel("Original"), 0, 0)
        grid.addWidget(self.orig_label, 0, 1)
        grid.addWidget(b1, 0, 2)
        grid.addWidget(QLabel("Modified"), 1, 0)
        grid.addWidget(self.mod_label, 1, 1)
        grid.addWidget(b2, 1, 2)
        grid.addWidget(self.use_loaded_btn, 0, 3)
        lay.addLayout(grid)
        if self._original:
            self.orig_label.setText(os.path.basename(self._original))

        self.macro_label = QLabel("")
        self.macro_label.setObjectName("Mono")
        self.macro_label.setWordWrap(True)
        self.macro_label.setStyleSheet(
            "QLabel { background:#1d232b; border:1px solid #2c333c; "
            "border-radius:6px; padding:10px; font-size:13px; }")
        self.macro_label.setVisible(False)
        lay.addWidget(self.macro_label)

        self.verify_cb = QCheckBox("Verify the estimators against known controls "
                                   "(slower, but proves the numbers)")
        self.verify_cb.setChecked(True)
        self.verify_cb.setToolTip(
            "Resamples the original by known ratios and re-measures them. If those "
            "controls do not come back, the measurement of your pair is not reported "
            "as trustworthy.")
        lay.addWidget(self.verify_cb)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Measure pitch and tempo")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Ghost")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.copy_btn = QPushButton("Copy macro")
        self.copy_btn.setObjectName("Ghost")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._copy_macro)
        self.save_macro_btn = QPushButton("Save macro\u2026")
        self.save_macro_btn.setObjectName("Ghost")
        self.save_macro_btn.setEnabled(False)
        self.save_macro_btn.clicked.connect(self._save_macro)
        self.save_btn = QPushButton("Save report\u2026")
        self.save_btn.setObjectName("Ghost")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        for b in (self.run_btn, self.cancel_btn, self.copy_btn,
                  self.save_macro_btn, self.save_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlaceholderText(
            "Choose the original and the modified file, then press Measure.\n\n"
            "Both files should be the same recording. A minute of audio takes a few "
            "seconds; the self-check at the end adds a little more.")
        lay.addWidget(self.view, 1)

    # -- inputs ---------------------------------------------------------
    def _pick(self, which: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose an audio file", "",
                                              IMPORT_FILTER)
        if not path:
            return
        if which == "original":
            self._original = path
            self.orig_label.setText(os.path.basename(path))
        else:
            self._modified = path
            self.mod_label.setText(os.path.basename(path))

    def _use_loaded(self):
        if not self._loaded:
            return
        self._original = self._loaded
        self.orig_label.setText(os.path.basename(self._loaded))

    # -- run ------------------------------------------------------------
    def _run(self):
        if not (self._original and self._modified):
            QMessageBox.information(
                self, "Two files needed",
                "Choose the original and the modified file. Both must be the same "
                "recording, one of them processed from the other.")
            return
        if os.path.abspath(self._original) == os.path.abspath(self._modified):
            QMessageBox.information(self, "Same file",
                                    "Those are the same file - there is nothing to measure.")
            return
        self._set_running(True)
        self.view.setPlainText("Measuring\u2026")
        worker = PairWorker(self._original, self._modified, self,
                            verify=self.verify_cb.isChecked())
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(lambda: self._set_running(False))
        self._worker = worker
        worker.start()

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)
        else:
            self._worker = None

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, frac: float, label: str):
        self.progress.setValue(int(frac * 1000))
        if label:
            self.view.setPlainText(f"{label}\u2026")

    def _on_done(self, report: dict):
        self._report = report
        self.view.setPlainText(format_report(report))
        self.macro_label.setText(macro_for(report).strip().replace("\n", "      "))
        self.macro_label.setVisible(True)
        for b in (self.copy_btn, self.save_macro_btn, self.save_btn):
            b.setEnabled(True)

    def _on_failed(self, msg: str):
        self.view.setPlainText(f"Measurement failed:\n{msg}")
        QMessageBox.warning(self, "Measurement failed", msg)

    def _on_cancelled(self):
        self.view.setPlainText("Cancelled.")

    # -- outputs --------------------------------------------------------
    def _copy_macro(self):
        if not self._report:
            return
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(macro_for(self._report))
        self.view.appendPlainText("\nMacro copied to the clipboard.")

    def _save_macro(self):
        if not self._report:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the Audacity macro",
            os.path.splitext(os.path.basename(self._modified))[0] + "_macro.txt",
            "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(macro_for(self._report))
        self.view.appendPlainText(f"\nMacro saved to {path}")

    def _save(self):
        if not self._report:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the report",
                                              "pitch_tempo_report.txt", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(format_report(self._report))
        self.view.appendPlainText(f"\nReport saved to {path}")
