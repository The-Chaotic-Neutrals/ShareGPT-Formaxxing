"""
RefusalMancer - Binary classification tool for filtering refusals from conversation datasets.
"""

import os

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QProgressBar, QGroupBox, QListWidget,
    QComboBox, QSlider,
    QListWidgetItem, QSizePolicy, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDropEvent

from App.Other.BG import GalaxyBackgroundWidget


APP_TITLE = "RefusalMancer"
GARAK_MAX_MICROBATCH = 64

GARAK_UI_INFO = {
    "max_tokens": 8192,
    "default_split_tokens": 8192,
    "name": "Garak Classifier",
    "strategy": "Entry-level + sentence fallback",
    "threshold_tip": "Refusal confidence cutoff (0.0 - 1.0). Entries at or above this are removed as refusals.",
}


class FileListWidget(QListWidget):
    """QListWidget that accepts drag & drop of .jsonl files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event):
        if self._has_valid_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._has_valid_urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if local_path.lower().endswith(".jsonl"):
                    paths.append(local_path)

        if paths:
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_files"):
                parent = parent.parent()
            if parent is not None:
                parent._add_files(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _has_valid_urls(self, event):
        if not event.mimeData().hasUrls():
            return False
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".jsonl"):
                return True
        return False


class FilterThread(QThread):
    status_update = pyqtSignal(str)
    counts_update = pyqtSignal(int, int)
    progress_update = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def __init__(self, input_files, threshold, batch_size, conversation_batch_size, precision_mode, split_token_limit):
        super().__init__()
        self.input_files = input_files
        self.threshold = threshold
        self.batch_size = batch_size
        self.conversation_batch_size = conversation_batch_size
        self.precision_mode = precision_mode
        self.split_token_limit = split_token_limit
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def should_stop(self):
        return self._stop_requested

    def run(self):
        self.status_update.emit("Preparing classifier runtime...")

        from App.RefusalMancer.binary_classification import (
            filter_conversations as fc,
            initialize_models as init_models,
        )

        state = {
            "total_refusals": 0,
            "total_compliance": 0,
        }
        had_error = False

        self.status_update.emit("Loading garak classifier model...")
        init_models(status_update_callback=self.status_update.emit)
        self.status_update.emit("Counts represent entries (Refusals/Compliance), not GPU microbatch size.")

        total_files = max(1, len(self.input_files))

        for idx, input_file in enumerate(self.input_files):
            if self.should_stop():
                self.status_update.emit("Streaming cancelled by user.")
                had_error = True
                break

            per_file = {"refusals": 0, "compliance": 0}
            file_name = os.path.basename(input_file)
            self.status_update.emit(f"Processing file {idx + 1}/{total_files}: {file_name}")

            def status_callback(msg, file_idx=idx):
                nonlocal had_error
                if msg.startswith("Streaming error"):
                    had_error = True
                if msg.startswith("Streaming cancelled"):
                    had_error = True

                if msg.startswith("Filtering complete"):
                    overall = int(round(((file_idx + 1) / total_files) * 100))
                    self.progress_update.emit(max(0, min(100, overall)))
                    self.status_update.emit(f"Completed file {file_idx + 1}/{total_files}: {file_name}")
                else:
                    self.status_update.emit(msg)

            def progress_callback(percent, file_idx=idx):
                overall = int(round(((file_idx + (percent / 100.0)) / total_files) * 100))
                self.progress_update.emit(max(0, min(99, overall)))

            def counts_callback(refusals, compliance):
                delta_refusals = max(0, refusals - per_file["refusals"])
                delta_compliance = max(0, compliance - per_file["compliance"])
                per_file["refusals"] = refusals
                per_file["compliance"] = compliance

                state["total_refusals"] += delta_refusals
                state["total_compliance"] += delta_compliance
                self.counts_update.emit(state["total_refusals"], state["total_compliance"])

            class DummyEntry:
                def __init__(self, val):
                    self._val = val
                def get(self):
                    return self._val

            fc(
                input_file_entry=DummyEntry(input_file),
                threshold_entry=DummyEntry(str(self.threshold)),
                batch_size_entry=DummyEntry(str(self.batch_size)),
                conversation_batch_size_entry=DummyEntry(str(self.conversation_batch_size)),
                precision_entry=DummyEntry(self.precision_mode),
                split_tokens_entry=DummyEntry(str(self.split_token_limit)),
                status_update_callback=status_callback,
                counts_update_callback=counts_callback,
                progress_update_callback=progress_callback,
                stop_requested_callback=self.should_stop,
            )

            if had_error:
                break

        if not had_error:
            self.progress_update.emit(100)
        self.finished_signal.emit(not had_error)

class BinaryClassificationApp(QWidget):
    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.input_files = []
        self.thread = None
        
        self.setWindowTitle(f"{APP_TITLE} 🛡️")
        self.setMinimumSize(800, 600)
        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #F9FAFB;
                font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
                font-size: 11pt;
            }
            QLabel {
                color: #E5E7EB;
                background-color: transparent;
            }
            QGroupBox {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                margin-top: 18px;
                padding: 10px;
                background-color: rgba(5, 5, 15, 180);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #9CA3AF;
                font-weight: 600;
                font-size: 13pt;
            }
            QLineEdit {
                background-color: rgba(5, 5, 15, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 4px;
                padding: 5px 8px;
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QLineEdit:focus {
                border: 1px solid #2563EB;
            }
            QLineEdit::placeholder {
                color: #6B7280;
            }
            QListWidget {
                background-color: rgba(2, 2, 10, 220);
                color: #D1D5DB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #2563EB;
                color: #F9FAFB;
            }
            QListWidget::item:hover {
                background-color: rgba(37, 99, 235, 0.3);
            }
            QPushButton {
                background-color: rgba(2, 6, 23, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(17, 24, 39, 220);
                border-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: rgba(3, 7, 18, 240);
            }
            QPushButton:disabled {
                color: #6B7280;
                border-color: rgba(17, 24, 39, 200);
                background-color: rgba(2, 2, 2, 200);
            }
            QProgressBar {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                background-color: rgba(2, 2, 10, 220);
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 7px;
            }
        """)

    def resizeEvent(self, event):
        """Handle window resize to update background widget"""
        super().resizeEvent(event)
        if hasattr(self, 'galaxy_bg'):
            self.galaxy_bg.resize(self.size())

    def _build_ui(self):
        # Create galaxy background widget
        self.galaxy_bg = GalaxyBackgroundWidget(self)
        self.galaxy_bg.lower()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Initial resize
        QTimer.singleShot(100, lambda: self.galaxy_bg.resize(self.size()) if hasattr(self, 'galaxy_bg') else None)

        # Header
        header_row = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #F9FAFB;")

        subtitle_label = QLabel("Binary Classification for Refusal Filtering")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 11pt;")

        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_row.addLayout(title_container)
        header_row.addStretch()
        main_layout.addLayout(header_row)

        # Input files group
        input_group = QGroupBox("📁 Input Files")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        input_group.setLayout(input_layout)

        input_label = QLabel("Drag .jsonl files here or use Add Files button:")
        input_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        input_layout.addWidget(input_label)

        list_row = QHBoxLayout()
        list_row.setSpacing(10)

        self.file_list = FileListWidget(self)
        self.file_list.setMinimumHeight(80)
        self.file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_row.addWidget(self.file_list, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        add_btn = QPushButton("Add Files")
        add_btn.setFixedWidth(110)
        add_btn.clicked.connect(self.browse_input_file)
        btn_col.addWidget(add_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(110)
        clear_btn.setStyleSheet(self._red_button_style())
        clear_btn.clicked.connect(self.clear_files)
        btn_col.addWidget(clear_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        input_layout.addLayout(list_row)

        main_layout.addWidget(input_group)

        # Settings group
        settings_group = QGroupBox("⚙️ Classification Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(12)
        settings_group.setLayout(settings_layout)

        model_label = QLabel("Classifier Model: Garak Classifier")
        model_label.setStyleSheet("font-weight: 500;")
        settings_layout.addWidget(model_label)

        # Classification logic hint
        self.class_logic_label = QLabel(self.get_classification_logic_text())
        self.class_logic_label.setStyleSheet("color: #6B7280; font-size: 10pt; font-style: italic;")
        settings_layout.addWidget(self.class_logic_label)

        self.backend_capability_label = QLabel("")
        self.backend_capability_label.setStyleSheet("color: #9CA3AF; font-size: 9pt;")
        settings_layout.addWidget(self.backend_capability_label)

        # Threshold and batching
        params_row = QHBoxLayout()
        params_row.setSpacing(20)

        params_row.addWidget(QLabel("Threshold:"))
        self.threshold_entry = QLineEdit("0.75")
        self.threshold_entry.setFixedWidth(80)
        self.threshold_entry.setToolTip(GARAK_UI_INFO["threshold_tip"])
        params_row.addWidget(self.threshold_entry)

        params_row.addWidget(QLabel("GPU Microbatch:"))
        self.batch_size_entry = QLineEdit(str(GARAK_MAX_MICROBATCH))
        self.batch_size_entry.setFixedWidth(80)
        self.batch_size_entry.setToolTip("Target inference microbatch size on GPU (higher is faster until VRAM limit)")
        params_row.addWidget(self.batch_size_entry)

        params_row.addWidget(QLabel("CPU Queue:"))
        self.conversation_batch_entry = QLineEdit("512")
        self.conversation_batch_entry.setFixedWidth(80)
        self.conversation_batch_entry.setToolTip("Maximum prepared entries queued between CPU preparation and GPU scoring")
        params_row.addWidget(self.conversation_batch_entry)

        params_row.addWidget(QLabel("Precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["fp16", "bf16", "fp32"])
        self.precision_combo.setCurrentText("fp16")
        self.precision_combo.setToolTip("Inference precision on GPU")
        self.precision_combo.setFixedWidth(90)
        params_row.addWidget(self.precision_combo)

        params_row.addStretch()
        settings_layout.addLayout(params_row)

        split_row = QHBoxLayout()
        split_row.setSpacing(12)
        split_row.addWidget(QLabel("Split At Tokens:"))
        self.split_tokens_slider = QSlider(Qt.Horizontal)
        self.split_tokens_slider.setMinimum(32)
        self.split_tokens_slider.setSingleStep(32)
        self.split_tokens_slider.setPageStep(256)
        self.split_tokens_slider.valueChanged.connect(self._on_split_slider_changed)
        split_row.addWidget(self.split_tokens_slider, stretch=1)
        self.split_tokens_value_label = QLabel("0")
        self.split_tokens_value_label.setFixedWidth(58)
        self.split_tokens_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        split_row.addWidget(self.split_tokens_value_label)
        settings_layout.addLayout(split_row)

        main_layout.addWidget(settings_group)

        # Run controls
        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        self.filter_button = QPushButton("⚡ Filter Conversations")
        self.filter_button.setFixedHeight(42)
        self.filter_button.setStyleSheet(self._primary_button_style())
        self.filter_button.clicked.connect(self.start_filtering)
        self.filter_button.setEnabled(False)
        run_row.addWidget(self.filter_button, stretch=1)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setFixedHeight(42)
        self.stop_button.setFixedWidth(120)
        self.stop_button.setStyleSheet(self._red_button_style())
        self.stop_button.clicked.connect(self.stop_filtering)
        self.stop_button.setEnabled(False)
        run_row.addWidget(self.stop_button)

        main_layout.addLayout(run_row)

        # Progress group
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        progress_group.setLayout(progress_layout)

        # Status
        self.status_bar = QLabel("Status: Ready")
        self.status_bar.setStyleSheet("color: #9CA3AF;")
        progress_layout.addWidget(self.status_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(20)
        progress_layout.addWidget(self.progress_bar)

        # Counts
        counts_row = QHBoxLayout()
        counts_row.setSpacing(30)

        self.positive_count_label = QLabel("🚫 Refusals: 0")
        self.positive_count_label.setStyleSheet("color: #EF4444; font-weight: 600;")
        counts_row.addWidget(self.positive_count_label)

        self.negative_count_label = QLabel("✅ Compliance: 0")
        self.negative_count_label.setStyleSheet("color: #10B981; font-weight: 600;")
        counts_row.addWidget(self.negative_count_label)

        counts_row.addStretch()
        progress_layout.addLayout(counts_row)

        main_layout.addWidget(progress_group)
        main_layout.addStretch()

        self._sync_backend_ui()

    def _primary_button_style(self):
        return """
            QPushButton {
                background-color: #2563EB;
                color: #F9FAFB;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: rgba(37, 99, 235, 0.3);
                color: #6B7280;
            }
        """

    def _red_button_style(self):
        return """
            QPushButton {
                background-color: rgba(220, 38, 38, 0.8);
                color: #F9FAFB;
                border: 1px solid rgba(220, 38, 38, 0.5);
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(185, 28, 28, 0.9);
            }
        """

    def _current_backend_info(self):
        return GARAK_UI_INFO

    def _on_split_slider_changed(self, value):
        self.split_tokens_value_label.setText(str(int(value)))

    def _sync_backend_ui(self):
        info = self._current_backend_info()
        max_tokens = int(info["max_tokens"])
        default_split_tokens = int(info.get("default_split_tokens", max_tokens))
        default_split_tokens = max(32, min(default_split_tokens, max_tokens))
        self.split_tokens_slider.blockSignals(True)
        self.split_tokens_slider.setMaximum(max_tokens)
        self.split_tokens_slider.setValue(default_split_tokens)
        self.split_tokens_slider.blockSignals(False)
        self.split_tokens_value_label.setText(str(int(self.split_tokens_slider.value())))
        self.backend_capability_label.setText(
            f"{info['name']} supports up to {max_tokens} tokens ({info['strategy']} scoring). "
            "Lower 'Split At Tokens' values split earlier."
        )
        self.threshold_entry.setToolTip(info["threshold_tip"])
        try:
            current_microbatch = int(self.batch_size_entry.text())
        except ValueError:
            current_microbatch = GARAK_MAX_MICROBATCH
        if current_microbatch > GARAK_MAX_MICROBATCH:
            self.batch_size_entry.setText(str(GARAK_MAX_MICROBATCH))

    def _add_files(self, file_paths):
        """Add files from drag-drop or browse."""
        added = False
        for fp in file_paths:
            if fp.endswith('.jsonl') and fp not in self.input_files:
                self.input_files.append(fp)
                added = True
        if added:
            self._refresh_file_list()

    def _refresh_file_list(self):
        """Refresh the file list widget."""
        self.file_list.clear()
        import os
        for fp in self.input_files:
            folder = os.path.basename(os.path.dirname(fp))
            fname = os.path.basename(fp)
            display = f"{folder}/{fname}" if folder else fname
            item = QListWidgetItem(f"📄 {display}")
            item.setToolTip(fp)
            self.file_list.addItem(item)
        is_running = self.thread is not None and self.thread.isRunning()
        self.filter_button.setEnabled(bool(self.input_files) and not is_running)

    def browse_input_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select JSONL Files", filter="JSONL Files (*.jsonl)")
        if files:
            self._add_files(files)

    def clear_files(self):
        self.input_files.clear()
        self._refresh_file_list()
        self.update_status("Ready")
        self.progress_bar.setValue(0)
        self.positive_count_label.setText("🚫 Refusals: 0")
        self.negative_count_label.setText("✅ Compliance: 0")
        self.filter_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def start_filtering(self):
        if self.thread is not None and self.thread.isRunning():
            self.update_status("⚠️ Filtering already running.")
            return

        try:
            threshold = float(self.threshold_entry.text())
            batch_size = int(self.batch_size_entry.text())
            conversation_batch_size = int(self.conversation_batch_entry.text())
            precision_mode = self.precision_combo.currentText().strip().lower()
            split_token_limit = int(self.split_tokens_slider.value())
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
            if batch_size <= 0:
                raise ValueError("GPU microbatch must be positive")
            if batch_size > GARAK_MAX_MICROBATCH:
                raise ValueError(f"Garak Classifier supports a maximum GPU microbatch of {GARAK_MAX_MICROBATCH}")
            if conversation_batch_size <= 0:
                raise ValueError("CPU queue must be positive")
            if split_token_limit <= 0:
                raise ValueError("Split token limit must be positive")
            if precision_mode not in {"fp16", "bf16", "fp32"}:
                raise ValueError("Precision must be fp16, bf16, or fp32")
        except ValueError as e:
            self.update_status(f"⚠️ Invalid settings: {e}")
            return

        if not self.input_files:
            self.update_status("⚠️ No input files selected.")
            return

        self.filter_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.positive_count_label.setText("🚫 Refusals: 0")
        self.negative_count_label.setText("✅ Compliance: 0")
        self.thread = FilterThread(
            self.input_files,
            threshold,
            batch_size,
            conversation_batch_size,
            precision_mode,
            split_token_limit,
        )
        self.thread.status_update.connect(self.update_status)
        self.thread.counts_update.connect(self.update_counts)
        self.thread.progress_update.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.filtering_finished)
        self.thread.start()

    def filtering_finished(self, ok):
        if ok:
            self.update_status("✅ Filtering complete! Check Outputs/")
            self.progress_bar.setValue(100)
        self.thread = None
        self.filter_button.setEnabled(bool(self.input_files))
        self.stop_button.setEnabled(False)

    def stop_filtering(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.request_stop()
            self.stop_button.setEnabled(False)
            self.update_status("Stopping... finishing current chunk")

    def update_status(self, message):
        self.status_bar.setText(f"Status: {message}")

    def update_counts(self, refusal_count, clean_count):
        self.positive_count_label.setText(f"🚫 Refusals: {refusal_count}")
        self.negative_count_label.setText(f"✅ Compliance: {clean_count}")

    def get_classification_logic_text(self):
        return "Garak classifier selected. Scores full entries unless they exceed the split limit, then removes if any sentence is a refusal."


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    from App.Other.Theme import Theme

    window = BinaryClassificationApp(Theme.DARK)
    window.show()
    sys.exit(app.exec_())
