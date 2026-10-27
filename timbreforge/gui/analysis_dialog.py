"""Optional developer panel: compare a processed reference with its original.

Nothing here runs during normal use.  It exists so the processing profile can be
derived from evidence rather than guesswork, and it produces a *starting point*
profile - never an automatic "match".
"""

from __future__ import annotations

import json
import os
import traceback

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QDialog, QFileDialog, QGridLayout, QHBoxLayout,
                               QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                               QVBoxLayout)

from ..dsp.analyze import analyze_pair
from ..dsp.audio import IMPORT_FILTER, load_audio
from ..dsp.biquad import Band, magnitude_db
from ..dsp.dynamics import CompressorParams, LimiterParams
from ..dsp.graph import Profile
from ..dsp.meter import measure
from .controls import _label
from .theme import COLORS


class PairWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, orig: str, ref: str, parent=None):
        super().__init__(parent)
        self._orig, self._ref = orig, ref

    def run(self):
        try:
            rep = analyze_pair(self._orig, self._ref)
            self.done.emit(rep)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(exc))


def format_report(r: dict) -> str:
    o, ref = r["files"]["original"], r["files"]["reference"]
    g = r["geometry"]
    al = r["alignment"]
    seg = al["segments"]
    tun = r["tuning"]
    lo, lr, lm = r["levels_original"], r["levels_reference"], r["levels_reference_matched"]
    tr = r["level_stratified"]
    rel = r["level_relation"]
    ver = r["same_recording_verdict"]

    L = []
    L.append("FILES")
    L.append(f"  original   {o['filename']}  {o['format']}/{o['subtype']}  "
             f"{o['sample_rate']} Hz  {o['channels']} ch  {o['duration']:.6f} s")
    L.append(f"  reference  {ref['filename']}  {ref['format']}/{ref['subtype']}  "
             f"{ref['sample_rate']} Hz  {ref['channels']} ch  {ref['duration']:.6f} s")
    L.append(f"  frame delta {g['frame_delta']} samples "
             f"({g['duration_delta_s'] * 1000:+.2f} ms)")
    L.append("")
    L.append("SAME RECORDING?  (verified, not assumed)")
    for k, v in ver["checks"].items():
        L.append(f"  [{'ok' if v else '--'}] {k.replace('_', ' ')}")
    L.append(f"  best lag {al['lag_samples']} samples ({al['lag_ms']:+.3f} ms), "
             f"normalised correlation {al['correlation']:.3f}")
    L.append(f"  per-segment lag span {seg['lag_span_samples']:.1f} samples, "
             f"drift {seg['drift_ppm']:+.1f} ppm, "
             f"consistent offset: {seg['consistent_offset']}")
    L.append(f"  verdict: {ver['verdict']}")
    L.append("")
    L.append("TUNING")
    L.append(f"  best spectral ratio {tun['ratio']:.6f} "
             f"({tun.get('cents', 0):+.2f} cents) over +/-{tun.get('searched_cents', 0):.0f} "
             f"cents; correlation at best {tun.get('correlation_at_best', 0):.3f}, "
             f"at unity {tun.get('correlation_at_unity', 0):.3f}")
    L.append("  (a genuine pitch shift or stretch would show a systematic ratio; "
             "chroma similarity alone is not used)")
    L.append("")
    L.append("LEVELS (raw)")
    headers = ("", "original", "reference", "ref @ matched")
    L.append("  {:12s} {:>10s} {:>10s} {:>12s}".format(*headers))
    rows = [("sample peak dBFS", "peak_db"), ("true peak dBTP", "true_peak_db"),
            ("RMS dBFS", "rms_db"), ("LUFS", "lufs"), ("LRA", "lra"),
            ("crest dB", "crest_db"), ("env range dB", "envelope_range_db"),
            ("flatness", "flatness"), ("centroid Hz", "centroid_hz"),
            ("corr", "correlation"), ("S/M dB", "side_mid_db")]
    for label, key in rows:
        L.append(f"  {label:20s} {lo.get(key, 0):10.3f} {lr.get(key, 0):10.3f} "
                 f"{lm.get(key, 0):12.3f}")
    L.append(f"  reference was {r['level_match_gain_db']:+.2f} dB from the original's "
             f"loudness; comparisons below use that matched version")
    L.append("")
    L.append("SPECTRUM AT MATCHED LEVEL (band energy difference, dB)")
    for k, v in r["band_delta_db_at_match"].items():
        L.append(f"  {k:20s} {v:+7.2f}")
    L.append("")
    L.append("IS THE DIFFERENCE STATIC OR LEVEL-DEPENDENT?")
    if tr.get("ok"):
        L.append("  per loudness quartile of the original (transfer shape, gain removed):")
        for entry in tr["level_bins"]:
            rel_txt = "  ".join(f"{k.split('-')[0]}:{v:+.1f}"
                                for k, v in entry["relative"].items())
            L.append(f"  {entry['label']:14s} mean {entry['mean_level_db']:+.1f} dBFS, "
                     f"{entry['frames']} frames, overall {entry['overall_gain_db']:+.1f} dB")
            L.append(f"      {rel_txt}")
        L.append("  loudest minus quietest shape (level dependence):")
        for k, v in tr.get("level_dependence_db", {}).items():
            L.append(f"      {k:12s} {v:+6.2f} dB")
        L.append(f"  -> {tr.get('verdict', 'unknown')} "
                 f"(max deviation {tr.get('max_level_dependence_db', 0.0):.1f} dB; "
                 "above 2 dB the difference is not a static filter)")
        L.append("  Note: a broadband compressor shifts every band by the same amount, so it "
                 "does not show up here; it shows up in the level-relation section.")
    else:
        L.append(f"  not available: {tr.get('reason', 'unknown')}")
    L.append("")
    if rel.get("ok"):
        L.append("SHORT-TERM LEVEL RELATION")
        L.append(f"  slope {rel['level_slope']:.3f} (1.0 = no compression), implied ratio "
                 f"{rel['implied_ratio']:.2f}:1, correlation {rel['level_correlation']:.3f}")
        L.append(f"  original 200 ms level range {rel['orig_range_db']:.1f} dB, reference "
                 f"{rel['ref_range_db']:.1f} dB")
    L.append("")
    L.append("MODULATION (envelope spectrum peaks)")
    for name, m in (("original", r["modulation_original"]), ("reference", r["modulation_reference"])):
        if m.get("ok"):
            peaks = ", ".join(f"{p['freq_hz']:.2f} Hz" for p in m["peaks"][:6])
            L.append(f"  {name:10s} {peaks}")
        else:
            L.append(f"  {name:10s} unavailable")
    L.append("")
    L.append("CAUTION")
    L.append("  A reference with very high density and flat envelope (crest ~2.6 dB, many "
             "samples at full scale) cannot be reproduced while keeping the quality rules "
             "(no loudness maximising, no clipping, natural dynamics, -1 dBTP). The app "
             "reproduces tonal/textural character and rejects the crushed level.")
    return "\n".join(L)


def fit_eq_from_transfer(freqs: np.ndarray, target_db: np.ndarray, sr: int,
                         n_bands: int = 6) -> tuple[list[Band], float]:
    """Fit a small bank of broad peaking filters to a target magnitude curve."""
    from scipy.optimize import least_squares

    freqs = np.asarray(freqs, dtype=float)
    target = np.asarray(target_db, dtype=float)
    # fit where loudspeakers/ears are: below ~40 Hz the fit latches onto the
    # high-pass skirt and produces redundant low bells
    sel = (freqs >= 45) & (freqs <= min(18000, sr * 0.45))
    sel &= np.isfinite(target)
    f_sel, t_sel = freqs[sel], target[sel]
    if f_sel.size < 24:
        raise ValueError("not enough spectrum to fit")

    # start positions: the biggest deviations, spaced apart in log frequency
    order = np.argsort(-np.abs(t_sel))
    picks: list[float] = []
    for i in order:
        f = f_sel[i]
        if all(abs(np.log2(f / p)) > 0.6 for p in picks):
            picks.append(float(f))
        if len(picks) >= n_bands:
            break
    while len(picks) < n_bands:
        picks.append(float(np.geomspace(70, 12000, n_bands)[len(picks)]))

    x0 = []
    for f in picks:
        g = float(np.interp(np.log(f), np.log(f_sel), t_sel))
        x0 += [np.log10(f), np.clip(g, -12, 12), 0.0]  # q = 10**0 = 1
    lo_b = []
    hi_b = []
    for _ in range(n_bands):
        lo_b += [np.log10(50), -15.0, np.log10(0.35)]
        hi_b += [np.log10(18000), 15.0, np.log10(4.0)]

    def model(x):
        bands = [Band("peaking", 10 ** x[i], x[i + 1], 10 ** x[i + 2])
                 for i in range(0, len(x), 3)]
        return magnitude_db(bands, f_sel, sr)

    def residual(x):
        return model(x) - t_sel

    res = least_squares(residual, np.array(x0), bounds=(np.array(lo_b), np.array(hi_b)),
                        max_nfev=400, xtol=1e-4, ftol=1e-4)
    bands = []
    for i in range(0, len(res.x), 3):
        f = float(10 ** res.x[i])
        g = float(res.x[i + 1])
        q = float(10 ** res.x[i + 2])
        if abs(g) < 0.5 or f < 45.0:
            continue
        bands.append(Band("peaking", round(f, 1), round(g, 2), round(q, 2)))
    bands.sort(key=lambda b: b.freq)
    rms_err = float(np.sqrt(np.mean(res.fun ** 2)))
    return bands, rms_err


class AnalysisDialog(QDialog):
    profileCreated = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference pair analysis (developer utility)")
        self.resize(1000, 720)
        self._orig = ""
        self._ref = ""
        self._report: dict | None = None
        self._worker: PairWorker | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        head = QLabel("Reference pair analysis")
        head.setObjectName("Title")
        lay.addWidget(head)
        sub = QLabel(
            "Compares a processed reference with its unprocessed original and measures what "
            "actually differs. It reports timbre/dynamics/stereo facts, tells static from "
            "level-dependent changes, and can generate a starting-point EQ profile. It does "
            "not copy audio and does not optimise loudness.")
        sub.setObjectName("Faint")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.orig_label = QLabel("(none)")
        self.ref_label = QLabel("(none)")
        for lab in (self.orig_label, self.ref_label):
            lab.setObjectName("Mono")
        b1 = QPushButton("Choose original\u2026")
        b2 = QPushButton("Choose reference\u2026")
        b1.clicked.connect(lambda: self._pick("orig"))
        b2.clicked.connect(lambda: self._pick("ref"))
        grid.addWidget(_label("Original", width=80), 0, 0)
        grid.addWidget(self.orig_label, 0, 1)
        grid.addWidget(b1, 0, 2)
        grid.addWidget(_label("Reference", width=80), 1, 0)
        grid.addWidget(self.ref_label, 1, 1)
        grid.addWidget(b2, 1, 2)
        lay.addLayout(grid)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Analyse")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._run)
        self.save_btn = QPushButton("Save report\u2026")
        self.save_btn.setObjectName("Ghost")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.eq_btn = QPushButton("Create EQ starting point")
        self.eq_btn.setObjectName("Ghost")
        self.eq_btn.setEnabled(False)
        self.eq_btn.clicked.connect(self._make_eq_profile)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.save_btn)
        btns.addWidget(self.eq_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlaceholderText("Choose the two files and press Analyse.")
        lay.addWidget(self.view, 1)

        note = QLabel("Measurements describe the supplied pair. They are not a claim that "
                      "the original processing chain has been identified.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        lay.addWidget(note)

    def _pick(self, which: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio file", "", IMPORT_FILTER)
        if not path:
            return
        if which == "orig":
            self._orig = path
            self.orig_label.setText(os.path.basename(path))
        else:
            self._ref = path
            self.ref_label.setText(os.path.basename(path))

    def _run(self):
        if not (self._orig and self._ref):
            QMessageBox.information(self, "Files needed",
                                    "Choose both the original and the processed reference.")
            return
        self.run_btn.setEnabled(False)
        self.view.setPlainText("Analysing\u2026 this can take a while for long files.")
        worker = PairWorker(self._orig, self._ref, self)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._worker = worker
        worker.start()

    def _on_done(self, report: dict):
        self._report = report
        self.view.setPlainText(format_report(report))
        self.save_btn.setEnabled(True)
        self.eq_btn.setEnabled(True)

    def _on_failed(self, msg: str):
        self.view.setPlainText(f"Analysis failed:\n{msg}")
        QMessageBox.warning(self, "Analysis failed", msg)

    def _save(self):
        if not self._report:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save analysis report",
                                              "reference_analysis.txt", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(format_report(self._report))
            fh.write("\n\nRAW METRICS\n")
            fh.write(json.dumps({k: v for k, v in self._report.items()
                                 if k != "transfer_curve"}, indent=2, default=str))
        base = os.path.splitext(path)[0]
        try:
            with open(base + "_transfer.csv", "w", encoding="utf-8") as fh:
                fh.write("freq_hz,apparent_transfer_db\n")
                tc = self._report["transfer_curve"]
                for f, d in zip(tc["freqs_hz"], tc["db"]):
                    fh.write(f"{f:.3f},{d:.4f}\n")
        except Exception:  # noqa: BLE001
            pass
        self.view.appendPlainText(f"\nSaved to {path}")

    def _make_eq_profile(self):
        if not self._report:
            return
        tc = self._report["transfer_curve"]
        f = np.array(tc["freqs_hz"])
        d = np.array(tc["db"])
        mid = (f >= 250) & (f <= 2500)
        if mid.any():
            d = d - np.median(d[mid])  # normalise: the level match handles overall gain
        try:
            bands, err = fit_eq_from_transfer(f, d, 44100, n_bands=6)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Fit failed", str(exc))
            return
        prof = Profile()
        prof.name = os.path.splitext(os.path.basename(self._ref))[0] + " (EQ starting point)"
        prof.description = (
            f"Static EQ fitted to the apparent transfer of {os.path.basename(self._ref)} vs "
            f"{os.path.basename(self._orig)} (fit residual {err:.1f} dB). This is a starting "
            "point from measurement, not a validated reproduction.")
        prof.eq_bands = [Band("highpass", 30.0, 0.0, 0.707)] + bands
        rel = self._report.get("level_relation", {})
        notes = []
        if rel.get("ok") and rel["level_slope"] < 0.85:
            implied = float(np.clip(rel["implied_ratio"], 1.0, 6.0))
            prof.compressor = CompressorParams(
                enabled=True, threshold_db=-20.0, ratio=round(implied, 2),
                attack_ms=20.0, release_ms=200.0, knee_db=10.0, detector="rms",
                mix=0.7, makeup_db=0.0)
            notes.append(f"gentle {implied:.2f}:1 compressor from the measured level slope")
        prof.loudness_match = True
        prof.limiter = LimiterParams(enabled=True, ceiling_db=-1.0)
        prof.description += (" Includes " + ", ".join(notes) + "." if notes else "")
        self.profileCreated.emit(prof, self._ref)
        QMessageBox.information(
            self, "Profile created",
            f"Loaded a {len(bands)}-band EQ starting point (fit residual {err:.1f} dB).\n"
            "Audition it, adjust it, and save it as a profile. Compression is only added "
            "when the level statistics support it.")
