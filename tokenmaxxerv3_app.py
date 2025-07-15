import sys
import threading
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QTextEdit, QFileDialog, QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextCursor, QFont
from tokenmaxxerv3 import TokenMaxxerCore


class TokenMaxxerV3App(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧠 TokenMaxxerV3")
        self.setGeometry(100, 100, 960, 800)
        self.setStyleSheet("background-color: black; color: white;")

        # Core
        self.core = TokenMaxxerCore()

        # Queue for thread-safe logs
        self.queue = []

        # Timer to update log
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_queue)
        self.timer.start(100)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        font = QFont("Segoe UI", 11)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("📁 No file selected.")
        self.file_label.setFont(font)
        browse_button = QPushButton("💂 Browse")
        browse_button.setFont(font)
        browse_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        browse_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_button)

        # Model load
        model_layout = QHBoxLayout()
        model_label = QLabel("🤖 HF Model Repo:")
        model_label.setFont(font)
        self.model_combo = QComboBox()
        self.model_combo.setFont(font)
        self.model_combo.setEditable(True)
        load_button = QPushButton("🔄 Load Tokenizer")
        load_button.setFont(font)
        load_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        load_button.clicked.connect(self.load_tokenizer)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(load_button)

        # Max token entry
        token_layout = QHBoxLayout()
        token_label = QLabel("🔢 Max token length:")
        token_label.setFont(font)
        self.max_token_entry = QLineEdit("8192")
        self.max_token_entry.setFont(font)
        token_layout.addWidget(token_label)
        token_layout.addWidget(self.max_token_entry)

        # Buttons
        analyze_button = QPushButton("📊 Analyze Token Lengths")
        analyze_button.setFont(font)
        analyze_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        analyze_button.clicked.connect(self.run_analysis_thread)

        clean_button = QPushButton("🪼 Clean File")
        clean_button.setFont(font)
        clean_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        clean_button.clicked.connect(self.run_clean_thread)

        tokenize_button = QPushButton("🧵 Tokenize Only")
        tokenize_button.setFont(font)
        tokenize_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        tokenize_button.clicked.connect(self.run_tokenize_thread)

        # Output
        self.output_text = QTextEdit()
        self.output_text.setFont(QFont("Consolas", 11))
        self.output_text.setStyleSheet("background-color: black; color: white;")
        self.output_text.setReadOnly(True)

        # Add layouts
        layout.addLayout(file_layout)
        layout.addLayout(model_layout)
        layout.addLayout(token_layout)
        layout.addWidget(analyze_button)
        layout.addWidget(clean_button)
        layout.addWidget(tokenize_button)
        layout.addWidget(self.output_text)
        central_widget.setLayout(layout)

    # ---------------- UI events ----------------
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSONL File", "", "JSONL Files (*.jsonl)")
        if file_path:
            self.core.file_path = file_path
            self.file_label.setText(f"📁 {os.path.basename(file_path)}")

    def load_tokenizer(self):
        model_repo = self.model_combo.currentText()
        self.log(f"📦 Loading tokenizer from: {model_repo}...")
        try:
            self.core.load_tokenizer(model_repo)
            self.log("✅ Tokenizer loaded successfully.")
        except Exception as e:
            self.log(f"❌ Failed to load tokenizer: {e}")

    def run_analysis_thread(self):
        threading.Thread(target=self.analyze_file, daemon=True).start()

    def run_clean_thread(self):
        threading.Thread(target=self.clean_file, daemon=True).start()

    def run_tokenize_thread(self):
        threading.Thread(target=self.tokenize_only, daemon=True).start()

    def analyze_file(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            return
        self.log("🔍 Analyzing token lengths...")
        result = self.core.analyze_file(self.core.file_path)
        self.log(result)

    def clean_file(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            return
        try:
            max_tokens = int(self.max_token_entry.text())
        except ValueError:
            self.log("❌ Max token length must be an integer.")
            return
        result = self.core.clean_file(self.core.file_path, max_tokens)
        self.log(result)

    def tokenize_only(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            return
        result = self.core.tokenize_only(self.core.file_path)
        self.log(result)

    # ---------------- Logging ----------------
    def log(self, message):
        self.queue.append(message + "\n")

    def check_queue(self):
        while self.queue:
            msg = self.queue.pop(0)
            self.output_text.moveCursor(QTextCursor.End)
            self.output_text.insertPlainText(msg)
            self.output_text.ensureCursorVisible()

    # ---------------- Config ----------------
    def load_config(self):
        last, recents = self.core.load_config()
        self.model_combo.addItems(recents if recents else [])
        self.model_combo.setCurrentText(last)


# Standalone launcher
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TokenMaxxerV3App()
    win.show()
    sys.exit(app.exec_())
