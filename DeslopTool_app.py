# DeslopTool_app.py
import sys
import json
from pathlib import Path

import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QRadioButton, QCheckBox,
    QGroupBox, QGridLayout, QMainWindow, QSizePolicy, QSpacerItem,
    QTextEdit, QProgressBar, QListWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon

from DeslopTool import filter_dataset, load_filter_criteria
from DeslopTool_classifier import CharacterSlopFilter


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


class SlopFilterWorker(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(str)

    def __init__(self, slop_filter, dataset_file: str, output_path: Path):
        super().__init__()
        self.slop_filter = slop_filter
        self.dataset_file = dataset_file
        self.output_path = output_path

    def run(self):
        try:
            self.status_update.emit(f"Starting classifier filtering for {self.dataset_file}...")
            def progress_callback(current, total):
                # Forward through as-is; UI expects (current, total)
                self.progress_update.emit(current, total)

            filtered_count, total = self.slop_filter.filter_conversations(
                self.dataset_file,
                self.output_path,
                progress_callback=progress_callback
            )
            self.status_update.emit(f"Classifier filtering complete for {self.dataset_file}.")
            self.finished.emit(
                f"Filtered conversations saved to {self.output_path}\n"
                f"Kept {filtered_count} of {total} conversations."
            )
        except Exception as e:
            self.finished.emit(f"Classifier filtering failed for {self.dataset_file}: {e}")


class StringMatchWorker(QThread):
    """
    Runs the string-matching pipeline in a background thread to keep the UI responsive.
    """
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # (percent, 100)
    finished = pyqtSignal(str)

    def __init__(self, dataset_file: str, filter_files: list[str], threshold_multiplier: float | None):
        super().__init__()
        self.dataset_file = dataset_file
        self.filter_files = filter_files
        self.threshold_multiplier = threshold_multiplier

    def run(self):
        try:
            self.status_update.emit(f"Starting string matching filtering for {self.dataset_file}...")
            def progress_callback(current_percent: int, total: int):
                self.progress_update.emit(current_percent, total)

            # NOTE: pass the multiplier directly (no double-multiplying)
            output_message = filter_dataset(
                self.dataset_file,
                self.filter_files,
                self.threshold_multiplier,
                progress_callback=progress_callback
            )
            self.status_update.emit(f"String matching filtering complete for {self.dataset_file}.")
            self.finished.emit(output_message)
        except Exception as e:
            self.finished.emit(f"String matching filtering failed for {self.dataset_file}: {e}")


class DeslopToolApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeslopMancer ⚒️")
        self.setMinimumSize(650, 600)

        icon_path = Path(__file__).parent / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.filter_files: list[str] = []
        self.dataset_files: list[str] = []  # Multiple dataset files
        self.last_filter_file_path = Path('last_filter_file.txt')

        self.selected_filter_method = 1
        self.batch_size = 256
        self.force_gpu = True

        self.slop_filter = None
        self.slop_worker: SlopFilterWorker | None = None
        self.string_worker: StringMatchWorker | None = None

        self.current_file_index = 0  # Track which file is processing

        self.init_ui()
        self.load_last_filter_file()
        self.update_device_status()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        font_label = QFont("Helvetica", 12)
        font_button = QFont("Helvetica", 12, QFont.Bold)
        header_font = QFont("Helvetica", 14, QFont.Bold)

        groupbox_style = """
        QGroupBox {
            font-size: 14pt;
            font-weight: bold;
            color: #333366;
            border: none;
            margin-top: 20px;
            background-color: transparent;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 0 5px 0;
            border: none;
            color: #333366;
        }
        """

        # Dataset & Filters Section
        file_group = QGroupBox("Dataset & Filters 📂")
        file_group.setFont(header_font)
        file_group.setStyleSheet(groupbox_style)
        file_layout = QGridLayout()
        file_layout.setSpacing(10)

        lbl_dataset = QLabel("Dataset Files:")
        lbl_dataset.setFont(font_label)
        file_layout.addWidget(lbl_dataset, 0, 0)

        self.dataset_list = QListWidget()
        self.dataset_list.setFont(font_label)
        self.dataset_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dataset_list.setMinimumHeight(100)
        file_layout.addWidget(self.dataset_list, 0, 1, 2, 1)

        browse_button = QPushButton("📂 Add Datasets")
        browse_button.setFont(font_button)
        browse_button.setStyleSheet("QPushButton { padding: 8px; border-radius: 8px; background-color: #1e90ff; color: white; }")
        browse_button.clicked.connect(self.select_files)
        browse_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        file_layout.addWidget(browse_button, 0, 2)

        clear_button = QPushButton("Clear Datasets")
        clear_button.setFont(font_button)
        clear_button.setStyleSheet("QPushButton { padding: 8px; border-radius: 8px; background-color: #ff4444; color: white; }")
        clear_button.clicked.connect(self.clear_dataset_list)
        clear_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        file_layout.addWidget(clear_button, 1, 2)

        self.filter_button = QPushButton("Select Filter Files…")
        self.filter_button.setFont(font_button)
        self.filter_button.setStyleSheet("QPushButton { padding: 8px; border-radius: 8px; background-color: #1e90ff; color: white; }")
        self.filter_button.clicked.connect(self.select_filter_files)
        self.filter_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_layout.addWidget(self.filter_button, 2, 0, 1, 3)

        self.last_filter_label = QLabel("")
        self.last_filter_label.setFont(font_label)
        self.last_filter_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        file_layout.addWidget(self.last_filter_label, 3, 0, 1, 3)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group, stretch=0)

        # Processing Parameters Section
        params_group = QGroupBox("Processing Parameters ⚙️")
        params_group.setFont(header_font)
        params_group.setStyleSheet(groupbox_style)
        params_layout = QGridLayout()
        params_layout.setSpacing(10)

        lbl_threshold = QLabel("Threshold (× average):")
        lbl_threshold.setFont(font_label)
        params_layout.addWidget(lbl_threshold, 0, 0)

        self.threshold_entry = QLineEdit("0.69")
        self.threshold_entry.setFont(font_label)
        self.threshold_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        params_layout.addWidget(self.threshold_entry, 0, 1)

        lbl_batch = QLabel("Batch Size:")
        lbl_batch.setFont(font_label)
        params_layout.addWidget(lbl_batch, 1, 0)

        self.batch_entry = QLineEdit(str(self.batch_size))
        self.batch_entry.setFont(font_label)
        self.batch_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        params_layout.addWidget(self.batch_entry, 1, 1)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group, stretch=0)

        # Filtering Method Section
        method_group = QGroupBox("Filtering Method 🧹")
        method_group.setFont(header_font)
        method_group.setStyleSheet(groupbox_style)
        method_layout = QHBoxLayout()

        self.string_match_button = QRadioButton("String Matching")
        self.string_match_button.setFont(font_label)
        self.string_match_button.setChecked(True)
        self.string_match_button.toggled.connect(lambda: self.set_filter_method(1))
        method_layout.addWidget(self.string_match_button)

        self.classifier_button = QRadioButton("Classifier Removal")
        self.classifier_button.setFont(font_label)
        self.classifier_button.toggled.connect(lambda: self.set_filter_method(2))
        method_layout.addWidget(self.classifier_button)

        method_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        method_group.setLayout(method_layout)
        main_layout.addWidget(method_group, stretch=0)

        self.force_gpu_checkbox = QCheckBox("Force GPU ⚡")
        self.force_gpu_checkbox.setFont(font_label)
        self.force_gpu_checkbox.setChecked(True)
        self.force_gpu_checkbox.stateChanged.connect(self.toggle_force_gpu)
        self.force_gpu_checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(self.force_gpu_checkbox, stretch=0)

        self.device_label = QLabel("")
        self.device_label.setFont(font_label)
        self.device_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(self.device_label, stretch=0)

        process_button = QPushButton("⚙️ Process Datasets")
        process_button.setFont(QFont("Helvetica", 14, QFont.Bold))
        process_button.setStyleSheet("QPushButton { padding: 12px; border-radius: 12px; background-color: #28a745; color: white; }")
        process_button.clicked.connect(self.start_processing)
        process_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(process_button, stretch=0)

        # Output/status/progress area
        output_group = QGroupBox("Status Window")
        output_group.setFont(header_font)
        output_group.setStyleSheet(groupbox_style)
        output_layout = QVBoxLayout()
        output_layout.setSpacing(5)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(font_label)
        self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        output_layout.addWidget(self.output_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        output_layout.addWidget(self.progress_bar)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group, stretch=1)

        central_widget.setLayout(main_layout)

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
        if self.force_gpu or has_cuda:
            device_info = "GPU (Forced)" if self.force_gpu else "GPU"
        else:
            device_info = "CPU"
        self.device_label.setText(f"Running on: {device_info}")

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Dataset Files", "", "JSON/JSON Lines (*.json *.jsonl)")
        if file_paths:
            self.dataset_files.extend(file_paths)
            # Deduplicate while keeping order
            seen = set()
            self.dataset_files = [f for f in self.dataset_files if not (f in seen or seen.add(f))]
            self.update_dataset_list()

    def clear_dataset_list(self):
        self.dataset_files.clear()
        self.update_dataset_list()

    def update_dataset_list(self):
        self.dataset_list.clear()
        for file_path in self.dataset_files:
            self.dataset_list.addItem(file_path)

    def select_filter_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Filter Files", "", "Text files (*.txt)")
        if file_paths:
            self.filter_files = list(file_paths)
            last_filter_file = self.filter_files[-1]
            self.save_last_filter_file(last_filter_file)
            self.last_filter_label.setText(f"Last Selected Filter File: {last_filter_file}")

    def load_last_filter_file(self):
        if self.last_filter_file_path.exists():
            with open(self.last_filter_file_path, 'r', encoding='utf-8', errors='replace') as f:
                last_filter_file = f.read().strip()
                if last_filter_file:
                    # Only add if still exists
                    if Path(last_filter_file).exists():
                        self.filter_files.append(last_filter_file)
                        self.last_filter_label.setText(f"Last Selected Filter File: {last_filter_file}")

    def save_last_filter_file(self, filter_file: str):
        with open(self.last_filter_file_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(filter_file)

    def append_output(self, text: str):
        self.output_text.append(text)

    def set_progress(self, current: int, total: int):
        """
        We expect (current, total) where current is already a percent for string matching,
        and could be raw progress for classifier. Normalize to 0..100 when possible.
        """
        percent = 0
        if total > 0:
            # If caller already sends percentage with total=100, this keeps it 0..100.
            # If caller sends raw counts, scale them.
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

        # Validate threshold early for string-matching path
        if self.selected_filter_method == 1:
            try:
                _ = float(self.threshold_entry.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Input Error", "Threshold must be a number (multiplier of average).")
                return
            if not self.filter_files:
                QMessageBox.critical(self, "Input Error", "Please select at least one filter file.")
                return
            # Also validate criteria isn't empty
            if not load_filter_criteria(self.filter_files):
                QMessageBox.critical(self, "Input Error", "Filter criteria are empty. Please check your filter files.")
                return

        self.current_file_index = 0
        self.output_text.clear()
        self.progress_bar.setValue(0)
        self.process_next_file()

    def process_next_file(self):
        if self.current_file_index >= len(self.dataset_files):
            self.append_output("All datasets processed.")
            self.progress_bar.setValue(100)
            return

        dataset_file = self.dataset_files[self.current_file_index]
        self.append_output(f"Processing dataset: {dataset_file}")

        # Device info display only; selection happens inside classifier implementation
        device = 0 if (self.force_gpu or torch.cuda.is_available()) else -1
        _ = device  # currently not used directly here

        if self.selected_filter_method == 1:
            # STRING MATCH PATH — run in background thread
            try:
                threshold_multiplier = float(self.threshold_entry.text().strip())
            except ValueError:
                threshold_multiplier = None

            self.string_worker = StringMatchWorker(dataset_file, self.filter_files, threshold_multiplier)
            self.string_worker.status_update.connect(self.append_output)
            self.string_worker.progress_update.connect(self.set_progress)
            self.string_worker.finished.connect(self.on_string_filter_finished)
            self.string_worker.start()

        elif self.selected_filter_method == 2:
            # CLASSIFIER PATH — already threaded via SlopFilterWorker
            try:
                self.slop_filter = CharacterSlopFilter(batch_size=self.batch_size, confidence_margin=0.1)
            except Exception as e:
                QMessageBox.critical(self, "Initialization Error", f"Failed to initialize classifier: {e}")
                # Move on to next file to avoid getting stuck
                self.current_file_index += 1
                self.process_next_file()
                return

            output_dir = Path(dataset_file).parent / 'deslopped'
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
