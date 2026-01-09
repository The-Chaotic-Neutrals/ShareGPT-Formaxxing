"""
RefusalMancer - Binary classification tool for filtering refusals from conversation datasets.
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QButtonGroup, QProgressBar, QGroupBox, QListWidget,
    QListWidgetItem, QSizePolicy, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDropEvent

from App.RefusalMancer.binary_classification import (
    initialize_models,
    filter_conversations,
    set_filter_mode,
    FILTER_MODE_RP,
    FILTER_MODE_NORMAL
)
from App.Other.BG import GalaxyBackgroundWidget


APP_TITLE = "RefusalMancer"


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
    finished_signal = pyqtSignal()

    def __init__(self, input_files, threshold, batch_size):
        super().__init__()
        self.input_files = input_files
        self.threshold = threshold
        self.batch_size = batch_size

    def run(self):
        from App.RefusalMancer.binary_classification import filter_conversations as fc

        def status_callback(msg):
            if "%" in msg:
                try:
                    percent = int(float(msg.split("(")[-1].split("%")[0]))
                    self.progress_update.emit(percent)
                except ValueError:
                    pass
            self.status_update.emit(msg)

        def counts_callback(pos, neg):
            self.counts_update.emit(pos, neg)

        for input_file in self.input_files:
            class DummyEntry:
                def __init__(self, val):
                    self._val = val
                def get(self):
                    return self._val

            fc(
                input_file_entry=DummyEntry(input_file),
                threshold_entry=DummyEntry(str(self.threshold)),
                batch_size_entry=DummyEntry(str(self.batch_size)),
                status_update_callback=status_callback,
                counts_update_callback=counts_callback,
            )

        self.finished_signal.emit()


class BinaryClassificationApp(QWidget):
    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.input_files = []

        initialize_models()
        
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
            QRadioButton {
                spacing: 8px;
                color: #E5E7EB;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #4B5563;
                background-color: transparent;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #2563EB;
                background-color: #2563EB;
                border-radius: 8px;
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

        # Filter mode
        mode_row = QHBoxLayout()
        mode_row.setSpacing(25)
        mode_label = QLabel("Filter Mode:")
        mode_label.setStyleSheet("font-weight: 500;")
        
        self.mode_group = QButtonGroup(self)
        self.rp_mode_radio = QRadioButton("RP Filter (detect refusals)")
        self.rp_mode_radio.setChecked(True)
        self.rp_mode_radio.toggled.connect(self.update_filter_mode)
        self.mode_group.addButton(self.rp_mode_radio)

        self.normal_mode_radio = QRadioButton("Normal Filter (keep safe only)")
        self.normal_mode_radio.toggled.connect(self.update_filter_mode)
        self.mode_group.addButton(self.normal_mode_radio)

        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.rp_mode_radio)
        mode_row.addWidget(self.normal_mode_radio)
        mode_row.addStretch()
        settings_layout.addLayout(mode_row)

        # Classification logic hint
        self.class_logic_label = QLabel(self.get_classification_logic_text())
        self.class_logic_label.setStyleSheet("color: #6B7280; font-size: 10pt; font-style: italic;")
        settings_layout.addWidget(self.class_logic_label)

        # Threshold and batch size
        params_row = QHBoxLayout()
        params_row.setSpacing(20)

        params_row.addWidget(QLabel("Threshold:"))
        self.threshold_entry = QLineEdit("0.75")
        self.threshold_entry.setFixedWidth(80)
        self.threshold_entry.setToolTip("Classification threshold (0.0 - 1.0)")
        params_row.addWidget(self.threshold_entry)

        params_row.addWidget(QLabel("Batch Size:"))
        self.batch_size_entry = QLineEdit("64")
        self.batch_size_entry.setFixedWidth(80)
        self.batch_size_entry.setToolTip("Number of items to process at once")
        params_row.addWidget(self.batch_size_entry)

        params_row.addStretch()
        settings_layout.addLayout(params_row)

        main_layout.addWidget(settings_group)

        # Run button
        self.filter_button = QPushButton("⚡ Filter Conversations")
        self.filter_button.setFixedHeight(42)
        self.filter_button.setStyleSheet(self._primary_button_style())
        self.filter_button.clicked.connect(self.start_filtering)
        main_layout.addWidget(self.filter_button)

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

        self.positive_count_label = QLabel("✅ Positive (Refusal): 0")
        self.positive_count_label.setStyleSheet("color: #10B981; font-weight: 500;")
        counts_row.addWidget(self.positive_count_label)

        self.negative_count_label = QLabel("❌ Negative (Safe): 0")
        self.negative_count_label.setStyleSheet("color: #EF4444; font-weight: 500;")
        counts_row.addWidget(self.negative_count_label)

        counts_row.addStretch()
        progress_layout.addLayout(counts_row)

        main_layout.addWidget(progress_group)
        main_layout.addStretch()

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

    def browse_input_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select JSONL Files", filter="JSONL Files (*.jsonl)")
        if files:
            self._add_files(files)

    def clear_files(self):
        self.input_files.clear()
        self._refresh_file_list()
        self.update_status("Ready")
        self.progress_bar.setValue(0)
        self.positive_count_label.setText("✅ Positive (Refusal): 0")
        self.negative_count_label.setText("❌ Negative (Safe): 0")

    def start_filtering(self):
        try:
            threshold = float(self.threshold_entry.text())
            batch_size = int(self.batch_size_entry.text())
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
        except ValueError as e:
            self.update_status(f"⚠️ Invalid settings: {e}")
            return

        if not self.input_files:
            self.update_status("⚠️ No input files selected.")
            return

        self.filter_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.thread = FilterThread(self.input_files, threshold, batch_size)
        self.thread.status_update.connect(self.update_status)
        self.thread.counts_update.connect(self.update_counts)
        self.thread.progress_update.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.filtering_finished)
        self.thread.start()

    def filtering_finished(self):
        self.update_status("✅ Filtering complete! Check outputs/refusalmancer/")
        self.progress_bar.setValue(100)
        self.filter_button.setEnabled(True)

    def update_status(self, message):
        self.status_bar.setText(f"Status: {message}")

    def update_counts(self, positive_count, negative_count):
        self.positive_count_label.setText(f"✅ Positive (Refusal): {positive_count}")
        self.negative_count_label.setText(f"❌ Negative (Safe): {negative_count}")

    def update_filter_mode(self):
        mode = FILTER_MODE_RP if self.rp_mode_radio.isChecked() else FILTER_MODE_NORMAL
        set_filter_mode(mode)
        self.positive_count_label.setText("✅ Positive (Refusal): 0")
        self.negative_count_label.setText("❌ Negative (Safe): 0")
        self.class_logic_label.setText(self.get_classification_logic_text())
        self.update_status("Filter mode changed")

    def get_classification_logic_text(self):
        if self.rp_mode_radio.isChecked():
            return "RP Filter: Class 0 = Refusal (positive), Class 1 = Safe — Keeps conversations with refusals"
        else:
            return "Normal Filter: Class 1 = Refusal (positive), Class 0 = Safe — Keeps only safe conversations"


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    from App.Other.Theme import Theme

    window = BinaryClassificationApp(Theme.DARK)
    window.show()
    sys.exit(app.exec_())
