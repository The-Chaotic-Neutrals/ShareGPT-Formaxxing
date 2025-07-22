import os
import time
import logging
from deduplication import Deduplication

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QButtonGroup, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon


class DedupWorker(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)

    def __init__(self, deduplication, input_file, output_file, use_min_hash):
        super().__init__()
        self.deduplication = deduplication
        self.input_file = input_file
        self.output_file = output_file
        self.use_min_hash = use_min_hash

    def run(self):
        self.status_update.emit("🛠 Deduplication started...")
        method = "Min-Hash 🔍" if self.use_min_hash else "String-Match 🔗"
        self.status_update.emit(f"Method: {method} - Processing...")

        self.deduplication.duplicate_count = 0

        if self.use_min_hash:
            self.deduplication.perform_min_hash_deduplication(
                self.input_file, self.output_file,
                self.status_update.emit, self.progress_update.emit)
        else:
            self.deduplication.perform_sha256_deduplication(
                self.input_file, self.output_file,
                self.status_update.emit, self.progress_update.emit)

        self.progress_update.emit(1, 1)  # Ensure progress hits 100%
        self.status_update.emit("✅ Deduplication completed.")


class DeduplicationApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.deduplication = Deduplication()
        self.start_time = None

        self.setWindowTitle("DedupMancer ⚒️")
        self.setStyleSheet(f"""
            background-color: {self.theme.get('bg', '#fff')};
            color: {self.theme.get('fg', '#000')};
        """)

        self.setMinimumWidth(600)
        self.setup_ui()
        self.worker = None

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        font_label = QFont("Arial", 14)
        font_button = QFont("Arial", 13, QFont.Bold)
        font_status = QFont("Arial", 12)

        # Input file selection
        file_layout = QHBoxLayout()
        file_layout.setSpacing(10)

        lbl_input = QLabel("📁 Input File:")
        lbl_input.setFont(font_label)
        file_layout.addWidget(lbl_input)

        self.input_file_entry = QLineEdit()
        self.input_file_entry.setPlaceholderText("Select input file (.jsonl)")
        self.input_file_entry.setFont(font_label)
        file_layout.addWidget(self.input_file_entry)

        browse_button = QPushButton("Browse 🔎")
        browse_button.setFont(font_button)
        browse_button.setCursor(Qt.PointingHandCursor)
        browse_button.setStyleSheet(self._button_style())
        browse_button.setFixedWidth(120)
        browse_button.clicked.connect(self.browse_input_file)
        file_layout.addWidget(browse_button)

        main_layout.addLayout(file_layout)

        # Radio buttons for dedup method
        method_layout = QHBoxLayout()
        method_layout.setSpacing(40)

        self.min_hash_radio = QRadioButton("Min-Hash / Semantic 🔍")
        self.min_hash_radio.setFont(font_label)
        self.sha256_radio = QRadioButton("String-Match 🔗")
        self.sha256_radio.setFont(font_label)

        self.min_hash_radio.setChecked(True)

        method_layout.addStretch()
        method_layout.addWidget(self.min_hash_radio)
        method_layout.addWidget(self.sha256_radio)
        method_layout.addStretch()

        main_layout.addLayout(method_layout)

        # Dedup button
        self.dedup_button = QPushButton("🗑️ Remove Duplicates")
        self.dedup_button.setFont(font_button)
        self.dedup_button.setCursor(Qt.PointingHandCursor)
        self.dedup_button.setStyleSheet(self._button_style())
        self.dedup_button.setFixedHeight(40)
        self.dedup_button.clicked.connect(self.start_deduplication)
        main_layout.addWidget(self.dedup_button)

        # Status bar + speed
        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(font_status)
        self.progress_percent_label = QLabel("0%")
        self.progress_percent_label.setFont(font_status)
        self.speed_label = QLabel("Speed: 0 it/s")
        self.speed_label.setFont(font_status)

        status_layout.addWidget(self.status_label, stretch=3)
        status_layout.addWidget(self.progress_percent_label, stretch=1, alignment=Qt.AlignCenter)
        status_layout.addWidget(self.speed_label, stretch=2, alignment=Qt.AlignRight)

        main_layout.addLayout(status_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 10px;
                background-color: #eee;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                border-radius: 10px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

    def _button_style(self):
        return """
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3aa0ff;
            }
            QPushButton:pressed {
                background-color: #1573cc;
            }
            QPushButton:disabled {
                background-color: #999999;
                color: #666666;
            }
        """

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "",
                                                   "JSONL files (*.jsonl)")
        if file_path:
            self.input_file_entry.setText(file_path)
            self.progress_bar.setValue(0)
            self.progress_percent_label.setText("0%")
            self.speed_label.setText("Speed: 0 it/s")
            self.update_status("Ready")

    def start_deduplication(self):
        input_file = self.input_file_entry.text()
        if not input_file.endswith('.jsonl'):
            QMessageBox.critical(self, "Error", "❌ Invalid file type. Please select a .jsonl file.")
            return

        output_dir = "deduplicated"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-deduplicated.jsonl")

        self.dedup_button.setEnabled(False)
        self.start_time = time.time()

        use_min_hash = self.min_hash_radio.isChecked()

        # Start the worker thread
        self.worker = DedupWorker(self.deduplication, input_file, output_file, use_min_hash)
        self.worker.status_update.connect(self.update_status)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.dedup_finished)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_progress(self, current, total):
        if total == 0:
            percent = 0
        elif current >= total:
            percent = 100
        else:
            percent = (current / total) * 100

        self.progress_bar.setValue(int(percent))
        self.progress_percent_label.setText(f"{percent:.2f}%")

        elapsed = time.time() - self.start_time if self.start_time else 1
        speed = current / elapsed if elapsed > 0 else 0
        self.speed_label.setText(f"Speed: {speed:.2f} it/s")

    def dedup_finished(self):
        self.dedup_button.setEnabled(True)
        self.update_status("✅ Deduplication finished.")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    window = DeduplicationApp(theme)
    window.show()
    sys.exit(app.exec_())
