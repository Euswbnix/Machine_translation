"""Main window for the Transformer-MT desktop GUI.

Threads:
  - DownloadWorker  : fetches HF model files
  - TranslateWorker : runs inference

Both communicate progress to the main thread via Qt signals so the UI
stays responsive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QStatusBar, QVBoxLayout, QWidget,
)

from gui.downloader import cache_dir_for, download_model, is_cached
from gui.inference import CancelledError, TransformerMTRuntime, TranslationResult
from gui.model_registry import (
    DEFAULT_MODEL_KEY, ModelEntry, find_by_key, get_models,
)


# ============================================================ workers

class DownloadWorker(QThread):
    # file_idx, file_total, bytes_done, bytes_total, filename
    progress = Signal(int, int, int, int, str)
    finished_ok = Signal(str)                # local cache dir
    failed = Signal(str)                     # error message

    def __init__(self, model: ModelEntry):
        super().__init__()
        self.model = model

    def run(self):
        try:
            local = download_model(
                self.model.hf_repo, self.model.files,
                progress_cb=self._on_progress,
            )
            self.finished_ok.emit(str(local))
        except Exception as e:
            self.failed.emit(str(e))

    def _on_progress(self, i: int, total: int, bdone: int, btotal: int,
                     fname: str):
        self.progress.emit(i, total, bdone, btotal, fname)


class TranslateWorker(QThread):
    progress = Signal(int, int, str)         # current, total, preview
    finished_ok = Signal(int, object)        # req_id, TranslationResult
    cancelled = Signal(int)                  # req_id
    failed = Signal(str)

    def __init__(self, runtime: TransformerMTRuntime, text: str, beam: int,
                 req_id: int, sentence_cache: Optional[dict] = None):
        super().__init__()
        self.runtime = runtime
        self.text = text
        self.beam = beam
        self.req_id = req_id
        self.sentence_cache = sentence_cache

    def run(self):
        try:
            result = self.runtime.translate(
                self.text,
                beam=self.beam,
                progress_cb=self._on_progress,
                cancel_check=self.isInterruptionRequested,
                sentence_cache=self.sentence_cache,
            )
            self.finished_ok.emit(self.req_id, result)
        except CancelledError:
            self.cancelled.emit(self.req_id)
        except Exception as e:
            self.failed.emit(str(e))

    def _on_progress(self, i: int, total: int, preview: str):
        self.progress.emit(i, total, preview)


# ============================================================ main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformer MT")
        self.resize(960, 640)

        self._runtime: Optional[TransformerMTRuntime] = None
        self._current_model: Optional[ModelEntry] = None
        self._download_worker: Optional[DownloadWorker] = None
        self._translate_worker: Optional[TranslateWorker] = None
        self._translate_req_id = 0
        self._latest_applied_req_id = 0
        self._pending_live_rerun = False
        self._sentence_cache: dict = {}
        self._live_enabled = True
        self._auto_download_attempted: set[str] = set()

        self._build_ui()
        self._populate_models()
        self._on_model_change(0)             # auto-select default

    # ---------------------------------------------------------- UI build

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # --- top: model selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        top.addWidget(self.model_combo, stretch=1)
        self.download_btn = QPushButton("Download model")
        self.download_btn.clicked.connect(self._on_download_clicked)
        top.addWidget(self.download_btn)
        self.live_checkbox = QCheckBox("Live")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.toggled.connect(self._on_live_toggled)
        top.addWidget(self.live_checkbox)
        self.live_hint = QLabel("Tip: Big models can be slow for Live mode.")
        self.live_hint.setStyleSheet("color: #666;")
        top.addWidget(self.live_hint)
        outer.addLayout(top)

        # --- middle: input / output
        io = QHBoxLayout()
        in_col = QVBoxLayout()
        in_col.addWidget(QLabel("Input (English):"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type or paste English text here…")
        self.input_edit.textChanged.connect(self._on_input_changed)
        in_col.addWidget(self.input_edit)
        io.addLayout(in_col)

        out_col = QVBoxLayout()
        out_col.addWidget(QLabel("Output:"))
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        out_col.addWidget(self.output_edit)
        io.addLayout(out_col)
        outer.addLayout(io, stretch=1)

        # --- bottom: translate button + progress
        bot = QHBoxLayout()
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.clicked.connect(self._on_translate_clicked)
        bot.addWidget(self.translate_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bot.addWidget(self.progress, stretch=1)
        outer.addLayout(bot)

        # status bar
        self.setStatusBar(QStatusBar())
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(400)  # debounce
        self._live_timer.timeout.connect(self._run_live_translate)

        # advanced menu
        menu = self.menuBar().addMenu("Advanced")
        ft_action = QAction("Show fine-tuned models", self, checkable=True)
        ft_action.toggled.connect(self._on_toggle_ft)
        menu.addAction(ft_action)

    # ---------------------------------------------------------- model list

    def _populate_models(self, include_ft: bool = False):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in get_models(include_ft=include_ft):
            tag = " ✓" if is_cached(m.hf_repo, m.files) else ""
            self.model_combo.addItem(f"{m.label}{tag}", userData=m.key)
        # default to enfr_base_v11 if present
        default_idx = max(0, self.model_combo.findData(DEFAULT_MODEL_KEY))
        self.model_combo.setCurrentIndex(default_idx)
        self.model_combo.blockSignals(False)

    def _on_toggle_ft(self, checked: bool):
        self._populate_models(include_ft=checked)

    def _on_model_change(self, _idx: int):
        key = self.model_combo.currentData()
        m = find_by_key(key)
        if m is None:
            return
        self._current_model = m
        self._runtime = None  # invalidate cached runtime
        self._sentence_cache.clear()
        cached = is_cached(m.hf_repo, m.files)
        self.download_btn.setEnabled(not cached)
        self.translate_btn.setEnabled(cached)
        self.statusBar().showMessage(
            "Ready — click Translate" if cached
            else f"Click Download (~{m.size_mb_estimate} MB)"
        )
        if not cached:
            self._maybe_auto_download_model(m)

    def _maybe_auto_download_model(self, m: ModelEntry):
        if m.key in self._auto_download_attempted:
            return
        if self._download_worker is not None and self._download_worker.isRunning():
            return
        self._auto_download_attempted.add(m.key)
        QTimer.singleShot(0, lambda: self._start_download(m, user_initiated=False))

    # ---------------------------------------------------------- download

    def _on_download_clicked(self):
        m = self._current_model
        if m is None:
            return
        self._start_download(m, user_initiated=True)

    def _start_download(self, m: ModelEntry, *, user_initiated: bool):
        size = m.size_mb_estimate
        if user_initiated:
            ans = QMessageBox.question(
                self, "Download model",
                f"Download {m.label} (~{size} MB) from HuggingFace Hub?\n"
                "This may take 1–5 minutes depending on your connection.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if ans != QMessageBox.Yes:
                return

        self._set_busy(True, indeterminate=False)
        self.progress.setRange(0, 1000)             # use 0..1000 for fractional %
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        if user_initiated:
            self.statusBar().showMessage(f"Downloading {m.label}…")
        else:
            self.statusBar().showMessage(
                f"First-time setup: downloading {m.label} (~{size} MB)…"
            )

        self._download_worker = DownloadWorker(m)
        self._download_worker.progress.connect(self._on_dl_progress)
        self._download_worker.finished_ok.connect(self._on_dl_done)
        self._download_worker.failed.connect(self._on_dl_failed)
        self._download_worker.start()

    def _on_dl_progress(self, i: int, total_files: int,
                         bdone: int, btotal: int, fname: str):
        # Combine per-file progress into a single overall percentage
        # (assume files are equally weighted by btotal where known, or by
        # file-count where btotal is unknown for some pre-current file).
        if btotal > 0:
            file_frac = bdone / btotal
        else:
            file_frac = 0.0
        overall = (i + file_frac) / max(total_files, 1)
        self.progress.setValue(int(overall * 1000))
        self.progress.setFormat(f"{overall * 100:.1f}%")
        if btotal > 0:
            mb_done = bdone / 1024 / 1024
            mb_total = btotal / 1024 / 1024
            self.statusBar().showMessage(
                f"Downloading file {i + 1}/{total_files}: "
                f"{fname}  —  {mb_done:.1f} / {mb_total:.1f} MB"
            )
        else:
            self.statusBar().showMessage(
                f"Downloading file {i + 1}/{total_files}: {fname}"
            )

    def _on_dl_done(self, _local_dir: str):
        self._set_busy(False)
        self.statusBar().showMessage("Download complete.")
        self._populate_models(include_ft=False)
        self._on_model_change(self.model_combo.currentIndex())

    def _on_dl_failed(self, err: str):
        self._set_busy(False)
        QMessageBox.critical(self, "Download failed", err)
        self.statusBar().showMessage("Download failed.")

    # ---------------------------------------------------------- translate

    def _ensure_runtime(self) -> bool:
        if self._runtime is not None:
            return True
        m = self._current_model
        if m is None or not is_cached(m.hf_repo, m.files):
            return False
        self.statusBar().showMessage("Loading model…")
        QApplication.processEvents()
        try:
            self._runtime = TransformerMTRuntime(cache_dir_for(m.hf_repo))
        except Exception as e:
            QMessageBox.critical(self, "Model load failed", str(e))
            return False
        return True

    def _on_translate_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("Type or paste some English text first.")
            return
        if not self._ensure_runtime():
            return

        self._start_translation(text=text, live=False)

    def _on_input_changed(self):
        if not self._live_enabled:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            self._live_timer.stop()
            self._pending_live_rerun = False
            # Invalidate any in-flight result immediately.
            self._latest_applied_req_id = self._translate_req_id + 1
            if self._translate_worker is not None and self._translate_worker.isRunning():
                self._translate_worker.requestInterruption()
            self.output_edit.clear()
            self.statusBar().showMessage("Ready.")
            self._set_busy(False)
            return
        self._live_timer.start()

    def _on_live_toggled(self, checked: bool):
        self._live_enabled = checked
        if not checked:
            self._live_timer.stop()
            self._pending_live_rerun = False
            if self._translate_worker is not None and self._translate_worker.isRunning():
                self._translate_worker.requestInterruption()
            self.statusBar().showMessage("Live mode off. Click Translate to run.")
            return
        self.statusBar().showMessage("Live mode on.")
        text = self.input_edit.toPlainText().strip()
        if text:
            self._live_timer.start()

    def _run_live_translate(self):
        if not self._live_enabled:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        if not self._ensure_runtime():
            return
        if self._translate_worker is not None and self._translate_worker.isRunning():
            self._pending_live_rerun = True
            self._translate_worker.requestInterruption()
            return
        self._start_translation(text=text, live=True)

    def _start_translation(self, *, text: str, live: bool):
        self._translate_req_id += 1
        req_id = self._translate_req_id
        self._set_busy(True, indeterminate=True)
        self.progress.setRange(0, 0)
        if live:
            self.statusBar().showMessage("Live translating…")
        else:
            self.statusBar().showMessage("Translating…")
            self.output_edit.clear()

        self._translate_worker = TranslateWorker(
            self._runtime,
            text,
            beam=5,
            req_id=req_id,
            sentence_cache=self._sentence_cache,
        )
        self._translate_worker.progress.connect(self._on_tr_progress)
        self._translate_worker.finished_ok.connect(self._on_tr_done)
        self._translate_worker.cancelled.connect(self._on_tr_cancelled)
        self._translate_worker.failed.connect(self._on_tr_failed)
        self._translate_worker.start()

    def _on_tr_progress(self, i: int, total: int, _preview: str):
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(i)
            self.statusBar().showMessage(f"Translating sentence {i}/{total}…")

    def _on_tr_done(self, req_id: int, result: TranslationResult):
        if req_id < self._latest_applied_req_id:
            return
        current_text = self.input_edit.toPlainText().strip()
        if not current_text:
            self._set_busy(False)
            self.output_edit.clear()
            self.statusBar().showMessage("Ready.")
            return
        self._latest_applied_req_id = req_id
        self._set_busy(False)
        self.output_edit.setPlainText(result.output_text)
        n_flagged = sum(result.flagged)
        if n_flagged:
            self.statusBar().showMessage(
                f"Done. {n_flagged} sentence(s) flagged as possibly mistranslated "
                "(unusual length ratio); inspect carefully."
            )
        else:
            self.statusBar().showMessage(
                f"Done. {len(result.sentences_tgt)} sentence(s) translated."
            )
        if self._pending_live_rerun:
            self._pending_live_rerun = False
            if self._live_enabled:
                self._run_live_translate()

    def _on_tr_cancelled(self, _req_id: int):
        self._set_busy(False)
        if not self.input_edit.toPlainText().strip():
            self.output_edit.clear()
            self.statusBar().showMessage("Ready.")
            return
        if self._pending_live_rerun:
            self._pending_live_rerun = False
            if self._live_enabled:
                self._run_live_translate()

    def _on_tr_failed(self, err: str):
        self._set_busy(False)
        QMessageBox.critical(self, "Translation failed", err)
        self.statusBar().showMessage("Translation failed.")

    # ---------------------------------------------------------- helpers

    def _set_busy(self, busy: bool, *, indeterminate: bool = False):
        self.translate_btn.setEnabled(not busy)
        self.download_btn.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy and indeterminate:
            self.progress.setRange(0, 0)
