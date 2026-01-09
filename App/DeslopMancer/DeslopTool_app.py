"""
DeslopMancer - Tool for filtering slop from conversation datasets.
Supports string matching and classifier-based removal methods.
"""

import sys
import json
from pathlib import Path

import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QRadioButton, QCheckBox,
    QGroupBox, QSizePolicy, QPlainTextEdit, QProgressBar, QListWidget,
    QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QIcon, QDropEvent

from App.DeslopMancer.DeslopTool import filter_dataset, load_filter_criteria
from App.DeslopMancer.DeslopTool_classifier import CharacterSlopFilter
from App.Other.BG import GalaxyBackgroundWidget


APP_TITLE = "DeslopMancer"


def load_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line. Error: {e}")
    return data


class FileListWidget(QListWidget):
    """QListWidget that accepts drag & drop of .jsonl/.json files."""

    def __init__(self, parent=None, extensions=None):
        super().__init__(parent)
        self.extensions = extensions or {".jsonl", ".json"}
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
                if any(local_path.lower().endswith(ext) for ext in self.extensions):
                    paths.append(local_path)

        if paths:
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_dataset_files"):
                parent = parent.parent()
            if parent is not None:
                parent._add_dataset_files(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _has_valid_urls(self, event):
        if not event.mimeData().hasUrls():
            return False
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if any(local_path.lower().endswith(ext) for ext in self.extensions):
                    return True
        return False


class SlopFilterWorker(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal(str)

    def __init__(self, slop_filter, dataset_file: str, output_path: Path):
        super().__init__()
        self.slop_filter = slop_filter
        self.dataset_file = dataset_file
        self.output_path = output_path

    def run(self):
        try:
            self.status_update.emit(f"Starting classifier filtering...")
            
            def progress_callback(current, total):
                self.progress_update.emit(current, total)

            filtered_count, total = self.slop_filter.filter_conversations(
                self.dataset_file,
                self.output_path,
                progress_callback=progress_callback
            )
            self.status_update.emit(f"Classifier filtering complete.")
            self.finished.emit(
                f"✅ Saved to {self.output_path}\n"
                f"   Kept {filtered_count} of {total} conversations."
            )
        except Exception as e:
            self.finished.emit(f"❌ Classifier filtering failed: {e}")


class StringMatchWorker(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    finished = pyqtSignal(str)

    def __init__(self, dataset_file: str, filter_files: list, threshold_multiplier: float):
        super().__init__()
        self.dataset_file = dataset_file
        self.filter_files = filter_files
        self.threshold_multiplier = threshold_multiplier

    def run(self):
        try:
            self.status_update.emit(f"Starting string matching filtering...")
            
            def progress_callback(current_percent: int, total: int):
                self.progress_update.emit(current_percent, total)

            output_message = filter_dataset(
                self.dataset_file,
                self.filter_files,
                self.threshold_multiplier,
                progress_callback=progress_callback
            )
            self.status_update.emit(f"String matching filtering complete.")
            self.finished.emit(output_message)
        except Exception as e:
            self.finished.emit(f"❌ String matching filtering failed: {e}")


class DeslopToolApp(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or {}
        
        self.filter_files: list = []
        self.dataset_files: list = []
        self.last_filter_file_path = Path(__file__).parent / 'last_filter_file.txt'

        self.selected_filter_method = 1
        self.batch_size = 256
        self.force_gpu = True

        self.slop_filter = None
        self.slop_worker = None
        self.string_worker = None
        self.current_file_index = 0

        self.setWindowTitle(f"{APP_TITLE} ⚒️")
        self.setMinimumSize(850, 700)
        
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._setup_style()
        self._build_ui()
        self.load_last_filter_file()
        self.update_device_status()

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
            QPlainTextEdit {
                background-color: rgba(2, 2, 10, 220);
                color: #D1D5DB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                font-family: Consolas, "Fira Code", monospace;
                font-size: 10pt;
                padding: 6px;
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
            QCheckBox {
                spacing: 8px;
                color: #E5E7EB;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #4B5563;
                background-color: #000000;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2563EB;
                background-color: #2563EB;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                background-color: rgba(2, 2, 10, 220);
                text-align: center;
                color: #F9FAFB;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 7px;
            }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'galaxy_bg'):
            self.galaxy_bg.resize(self.size())

    def _build_ui(self):
        # Galaxy background
        self.galaxy_bg = GalaxyBackgroundWidget(self)
        self.galaxy_bg.lower()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        QTimer.singleShot(100, lambda: self.galaxy_bg.resize(self.size()) if hasattr(self, 'galaxy_bg') else None)

        # Header
        header_row = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #F9FAFB;")

        subtitle_label = QLabel("Remove slop from conversation datasets")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 11pt;")

        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_row.addLayout(title_container)
        header_row.addStretch()
        
        # Device status in header
        self.device_label = QLabel("")
        self.device_label.setStyleSheet("color: #10B981; font-size: 10pt;")
        header_row.addWidget(self.device_label)
        
        main_layout.addLayout(header_row)

        # Dataset files group
        dataset_group = QGroupBox("📁 Dataset Files")
        dataset_layout = QVBoxLayout()
        dataset_layout.setSpacing(10)
        dataset_group.setLayout(dataset_layout)

        dataset_label = QLabel("Drag .jsonl files here or use Add Datasets button:")
        dataset_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        dataset_layout.addWidget(dataset_label)

        list_row = QHBoxLayout()
        list_row.setSpacing(10)

        self.dataset_list = FileListWidget(self)
        self.dataset_list.setMinimumHeight(80)
        self.dataset_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_row.addWidget(self.dataset_list, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        add_btn = QPushButton("Add Datasets")
        add_btn.setFixedWidth(120)
        add_btn.clicked.connect(self.select_files)
        btn_col.addWidget(add_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(120)
        clear_btn.setStyleSheet(self._red_button_style())
        clear_btn.clicked.connect(self.clear_dataset_list)
        btn_col.addWidget(clear_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        dataset_layout.addLayout(list_row)

        main_layout.addWidget(dataset_group)

        # Settings group
        settings_group = QGroupBox("⚙️ Filtering Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(12)
        settings_group.setLayout(settings_layout)

        # Method selection
        method_row = QHBoxLayout()
        method_row.setSpacing(25)
        method_label = QLabel("Method:")
        method_label.setStyleSheet("font-weight: 500;")

        self.string_match_radio = QRadioButton("String Matching")
        self.string_match_radio.setChecked(True)
        self.string_match_radio.toggled.connect(lambda: self.set_filter_method(1))

        self.classifier_radio = QRadioButton("Classifier Removal")
        self.classifier_radio.toggled.connect(lambda: self.set_filter_method(2))

        method_row.addWidget(method_label)
        method_row.addWidget(self.string_match_radio)
        method_row.addWidget(self.classifier_radio)
        method_row.addStretch()
        settings_layout.addLayout(method_row)

        # Filter files (for string matching)
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        
        self.filter_button = QPushButton("Select Filter Files...")
        self.filter_button.clicked.connect(self.select_filter_files)
        filter_row.addWidget(self.filter_button)

        self.last_filter_label = QLabel("No filter files selected")
        self.last_filter_label.setStyleSheet("color: #6B7280; font-size: 10pt;")
        filter_row.addWidget(self.last_filter_label, stretch=1)

        settings_layout.addLayout(filter_row)

        # Parameters
        params_row = QHBoxLayout()
        params_row.setSpacing(20)

        params_row.addWidget(QLabel("Threshold (× avg):"))
        self.threshold_entry = QLineEdit("0.69")
        self.threshold_entry.setFixedWidth(80)
        self.threshold_entry.setToolTip("Multiplier of average score for filtering")
        params_row.addWidget(self.threshold_entry)

        params_row.addWidget(QLabel("Batch Size:"))
        self.batch_entry = QLineEdit(str(self.batch_size))
        self.batch_entry.setFixedWidth(80)
        self.batch_entry.setToolTip("Number of items to process at once")
        params_row.addWidget(self.batch_entry)

        self.force_gpu_checkbox = QCheckBox("Force GPU")
        self.force_gpu_checkbox.setChecked(True)
        self.force_gpu_checkbox.stateChanged.connect(self.toggle_force_gpu)
        params_row.addWidget(self.force_gpu_checkbox)

        params_row.addStretch()
        settings_layout.addLayout(params_row)

        main_layout.addWidget(settings_group)

        # Process button
        self.process_button = QPushButton("⚙️ Process Datasets")
        self.process_button.setFixedHeight(42)
        self.process_button.setStyleSheet(self._primary_button_style())
        self.process_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_button)

        # Output group
        output_group = QGroupBox("📊 Status & Output")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(8)
        output_group.setLayout(output_layout)

        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Processing logs will appear here...")
        self.output_text.setMinimumHeight(120)
        output_layout.addWidget(self.output_text, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setTextVisible(True)
        output_layout.addWidget(self.progress_bar)

        main_layout.addWidget(output_group, stretch=1)

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

    def _add_dataset_files(self, file_paths):
        """Add files from drag-drop or browse."""
        for fp in file_paths:
            if fp not in self.dataset_files:
                self.dataset_files.append(fp)
        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        self.dataset_list.clear()
        import os
        for fp in self.dataset_files:
            folder = os.path.basename(os.path.dirname(fp))
            fname = os.path.basename(fp)
            display = f"{folder}/{fname}" if folder else fname
            item = QListWidgetItem(f"📄 {display}")
            item.setToolTip(fp)
            self.dataset_list.addItem(item)

    def set_filter_method(self, method: int):
        self.selected_filter_method = method

    def toggle_force_gpu(self):
        self.force_gpu = self.force_gpu_checkbox.isChecked()
        self.update_device_status()

    def update_device_status(self):
        try:
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if self.force_gpu and has_cuda:
            device_info = "🟢 GPU"
        elif has_cuda:
            device_info = "🟢 GPU Available"
        else:
            device_info = "🟡 CPU"
        self.device_label.setText(device_info)

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Dataset Files", "", "JSON/JSON Lines (*.json *.jsonl)"
        )
        if file_paths:
            self._add_dataset_files(file_paths)

    def clear_dataset_list(self):
        self.dataset_files.clear()
        self._refresh_dataset_list()
        self.output_text.clear()
        self.progress_bar.setValue(0)

    def select_filter_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Filter Files", "", "Text files (*.txt)"
        )
        if file_paths:
            self.filter_files = list(file_paths)
            last_filter_file = self.filter_files[-1]
            self.save_last_filter_file(last_filter_file)
            import os
            display = os.path.basename(last_filter_file)
            self.last_filter_label.setText(f"📝 {display} (+{len(self.filter_files)-1} more)" if len(self.filter_files) > 1 else f"📝 {display}")
            self.last_filter_label.setStyleSheet("color: #10B981; font-size: 10pt;")

    def load_last_filter_file(self):
        if self.last_filter_file_path.exists():
            with open(self.last_filter_file_path, 'r', encoding='utf-8', errors='replace') as f:
                last_filter_file = f.read().strip()
                if last_filter_file and Path(last_filter_file).exists():
                    self.filter_files.append(last_filter_file)
                    import os
                    display = os.path.basename(last_filter_file)
                    self.last_filter_label.setText(f"📝 {display}")
                    self.last_filter_label.setStyleSheet("color: #10B981; font-size: 10pt;")

    def save_last_filter_file(self, filter_file: str):
        with open(self.last_filter_file_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(filter_file)

    def append_output(self, text: str):
        self.output_text.appendPlainText(text)

    def set_progress(self, current: int, total: int):
        percent = 0
        if total > 0:
            if total == 100:
                percent = max(0, min(100, int(current)))
            else:
                percent = max(0, min(100, int((current / total) * 100)))
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}%")

    def start_processing(self):
        if not self.dataset_files:
            QMessageBox.critical(self, "Input Error", "Please select at least one dataset file.")
            return

        try:
            self.batch_size = int(self.batch_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Batch size must be an integer.")
            return

        if self.selected_filter_method == 1:
            try:
                _ = float(self.threshold_entry.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Threshold must be a number.")
                return
            if not self.filter_files:
                QMessageBox.critical(self, "Input Error", "Please select at least one filter file.")
                return
            if not load_filter_criteria(self.filter_files):
                QMessageBox.critical(self, "Input Error", "Filter criteria are empty. Check your filter files.")
                return

        self.process_button.setEnabled(False)
        self.current_file_index = 0
        self.output_text.clear()
        self.progress_bar.setValue(0)
        self.process_next_file()

    def process_next_file(self):
        if self.current_file_index >= len(self.dataset_files):
            self.append_output("\n✅ All datasets processed.")
            self.progress_bar.setValue(100)
            self.process_button.setEnabled(True)
            return

        dataset_file = self.dataset_files[self.current_file_index]
        import os
        self.append_output(f"\n📂 Processing: {os.path.basename(dataset_file)}")

        if self.selected_filter_method == 1:
            try:
                threshold_multiplier = float(self.threshold_entry.text().strip())
            except ValueError:
                threshold_multiplier = 0.69

            self.string_worker = StringMatchWorker(dataset_file, self.filter_files, threshold_multiplier)
            self.string_worker.status_update.connect(self.append_output)
            self.string_worker.progress_update.connect(self.set_progress)
            self.string_worker.finished.connect(self.on_string_filter_finished)
            self.string_worker.start()

        elif self.selected_filter_method == 2:
            try:
                self.slop_filter = CharacterSlopFilter(batch_size=self.batch_size, confidence_margin=0.1)
            except Exception as e:
                QMessageBox.critical(self, "Initialization Error", f"Failed to initialize classifier: {e}")
                self.current_file_index += 1
                self.process_next_file()
                return

            script_dir = Path(__file__).parent.absolute()
            repo_root = script_dir.parent.absolute()
            output_dir = repo_root / "outputs" / "deslopped"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_jsonl_filepath = output_dir / (Path(dataset_file).stem + "_deslopped.jsonl")

            self.slop_worker = SlopFilterWorker(self.slop_filter, dataset_file, output_jsonl_filepath)
            self.slop_worker.status_update.connect(self.append_output)
            self.slop_worker.progress_update.connect(self.set_progress)
            self.slop_worker.finished.connect(self.on_slop_filter_finished)
            self.slop_worker.start()
        else:
            QMessageBox.critical(self, "Error", "Invalid filter method selected.")

    def on_slop_filter_finished(self, message: str):
        self.progress_bar.setValue(100)
        self.append_output(message)
        self.current_file_index += 1
        self.process_next_file()

    def on_string_filter_finished(self, message: str):
        self.progress_bar.setValue(100)
        self.append_output(message)
        self.current_file_index += 1
        self.process_next_file()


def run_app():
    app = QApplication(sys.argv)
    window = DeslopToolApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
