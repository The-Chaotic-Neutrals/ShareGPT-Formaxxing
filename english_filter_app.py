import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QMessageBox, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from english_filter import filter_english_jsonl
from theme import Theme  # Your DARK theme import


class FilterWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, input_path, threshold):
        super().__init__()
        self.input_path = input_path
        self.threshold = threshold

    def run(self):
        try:
            stats = filter_english_jsonl(
                input_path=self.input_path,
                output_path=None,       # Auto paths inside filter_english_jsonl
                rejected_path=None,     # Auto paths inside filter_english_jsonl
                threshold=self.threshold,
                batch_size=256,
                workers=None
            )
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class EnglishFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EnglishFilter — FastText JSONL Cleaner")
        self.setFixedSize(700, 400)
        self.theme = Theme.DARK

        if os.path.exists("icon.ico"):
            try:
                self.setWindowIcon(QIcon("icon.ico"))
            except Exception as e:
                print(f"Icon error: {e}")

        self.input_path = ""

        self.worker = None

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        central.setStyleSheet(f"background-color: {self.theme['bg']};")

        font_title = QFont("Segoe UI", 28, QFont.Bold)
        font_label = QFont("Segoe UI", 11)
        font_status = QFont("Segoe UI", 14)

        self.title_label = QLabel("🧹 EnglishFilter")
        self.title_label.setFont(font_title)
        self.title_label.setStyleSheet(f"color: {self.theme['fg']};")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        layout.addSpacing(10)

        self.browse_btn = QPushButton("📂 Select Input JSONL")
        self.browse_btn.setFont(font_label)
        self.browse_btn.setStyleSheet(
            f"background-color: {self.theme['button_bg']}; color: {self.theme['button_fg']}; font-weight: bold;"
        )
        self.browse_btn.clicked.connect(self.select_input)
        layout.addWidget(self.browse_btn)

        threshold_layout = QHBoxLayout()
        self.threshold_label = QLabel("Threshold (default 0.69):")
        self.threshold_label.setFont(font_label)
        self.threshold_label.setStyleSheet(f"color: {self.theme['fg']};")
        threshold_layout.addWidget(self.threshold_label)

        self.threshold_entry = QLineEdit()
        self.threshold_entry.setFixedWidth(100)
        self.threshold_entry.setFont(font_label)
        self.threshold_entry.setStyleSheet(
            f"background-color: {self.theme['entry_bg']}; color: {self.theme['entry_fg']};"
        )
        self.threshold_entry.setText("0.69")
        threshold_layout.addWidget(self.threshold_entry)
        threshold_layout.addStretch()
        layout.addLayout(threshold_layout)

        self.start_btn = QPushButton("🚀 Start Filtering")
        self.start_btn.setFont(font_label)
        self.start_btn.setStyleSheet(
            f"background-color: {self.theme['button_bg']}; color: {self.theme['button_fg']}; font-weight: bold;"
        )
        self.start_btn.clicked.connect(self.start_filter)
        layout.addWidget(self.start_btn)
        layout.addSpacing(10)

        self.status_label = QLabel("💤 Waiting...")
        self.status_label.setFont(font_status)
        self.status_label.setStyleSheet(f"color: {self.theme['fg']};")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def select_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input JSONL", "", "JSONL Files (*.jsonl)")
        if path:
            self.input_path = path
            self.status_label.setText(f"📥 Input: {os.path.basename(path)}")

    def start_filter(self):
        if not self.input_path:
            QMessageBox.critical(self, "Error", "Please select an input file first.")
            return
        try:
            threshold = float(self.threshold_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Threshold must be a number.")
            return

        self.start_btn.setEnabled(False)
        self.status_label.setText("⏳ Filtering in progress...")

        self.worker = FilterWorker(
            self.input_path,
            threshold
        )
        self.worker.finished.connect(self.on_filter_finished)
        self.worker.error.connect(self.on_filter_error)
        self.worker.start()

    def on_filter_finished(self, stats):
        self.start_btn.setEnabled(True)
        QMessageBox.information(
            self, "Done",
            f"✅ English Filter Complete!\n"
            f"Total: {stats['total_lines']}\n"
            f"Kept: {stats['english_total']}\n"
            f"Removed: {stats['non_english_total']}\n"
            f"Errors: {stats['json_error_total']}"
        )
        self.status_label.setText("🎉 Done!")

    def on_filter_error(self, message):
        self.start_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Filtering failed:\n{message}")
        self.status_label.setText("❌ Failed!")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = EnglishFilterApp()
    window.show()
    sys.exit(app.exec_())
