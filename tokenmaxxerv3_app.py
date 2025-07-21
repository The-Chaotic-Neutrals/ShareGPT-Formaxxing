import sys
import threading
import os
import threading
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

        # Flag to track running tasks
        self.is_running = False

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

        # Status label
        self.status_label = QLabel("🛌 Idle")
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("color: #1e90ff; font-weight: bold;")
        layout.addWidget(self.status_label)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("📁 No file selected.")
        self.file_label.setFont(font)
        self.browse_button = QPushButton("💂 Browse")
        self.browse_button.setFont(font)
        self.browse_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        self.browse_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.browse_button)

        # Model load
        model_layout = QHBoxLayout()
        model_label = QLabel("🤖 HF Model Repo:")
        model_label.setFont(font)
        self.model_combo = QComboBox()
        self.model_combo.setFont(font)
        self.model_combo.setEditable(True)
        self.load_button = QPushButton("🔄 Load Tokenizer")
        self.load_button.setFont(font)
        self.load_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        self.load_button.clicked.connect(self.load_tokenizer)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.load_button)

        # Max token entry
        token_layout = QHBoxLayout()
        token_label = QLabel("🔢 Max token length:")
        token_label.setFont(font)
        self.max_token_entry = QLineEdit("8192")
        self.max_token_entry.setFont(font)
        token_layout.addWidget(token_label)
        token_layout.addWidget(self.max_token_entry)

        # Buttons
        self.analyze_button = QPushButton("📊 Analyze Token Lengths")
        self.analyze_button.setFont(font)
        self.analyze_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        self.analyze_button.clicked.connect(self.run_analysis_thread)

        self.clean_button = QPushButton("🪼 Clean File")
        self.clean_button.setFont(font)
        self.clean_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        self.clean_button.clicked.connect(self.run_clean_thread)

        self.tokenize_button = QPushButton("🧵 Tokenize Only")
        self.tokenize_button.setFont(font)
        self.tokenize_button.setStyleSheet("background-color: #1e90ff; color: white; font-weight: bold;")
        self.tokenize_button.clicked.connect(self.run_tokenize_thread)

        # Output
        self.output_text = QTextEdit()
        self.output_text.setFont(QFont("Consolas", 11))
        self.output_text.setStyleSheet("background-color: black; color: white;")
        self.output_text.setReadOnly(True)

        # Add layouts
        layout.addLayout(file_layout)
        layout.addLayout(model_layout)
        layout.addLayout(token_layout)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.clean_button)
        layout.addWidget(self.tokenize_button)
        layout.addWidget(self.output_text)
        central_widget.setLayout(layout)

    def update_button_states(self):
        """Enable or disable buttons based on is_running flag."""
        state = not self.is_running
        self.browse_button.setEnabled(state)
        self.load_button.setEnabled(state)
        self.analyze_button.setEnabled(state)
        self.clean_button.setEnabled(state)
        self.tokenize_button.setEnabled(state)

    # ---------------- UI events ----------------
    def select_file(self):
        if self.is_running:
            self.log("❌ Cannot select file while a task is running.")
            return
        self.is_running = True
        self.status_label.setText("📁 Selecting file...")
        self.update_button_states()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSONL File", "", "JSONL Files (*.jsonl)")
        if file_path:
            self.core.file_path = file_path
            self.file_label.setText(f"📁 {os.path.basename(file_path)}")
        self.is_running = False
        self.status_label.setText("🛌 Idle")
        self.update_button_states()

    def load_tokenizer(self):
        if self.is_running:
            self.log("❌ Cannot load tokenizer while a task is running.")
            return
        self.is_running = True
        self.status_label.setText("📦 Loading tokenizer...")
        self.update_button_states()
        model_repo = self.model_combo.currentText()
        self.log(f"📦 Loading tokenizer from: {model_repo}...")
        try:
            self.core.load_tokenizer(model_repo)
            self.log("✅ Tokenizer loaded successfully.")
        except Exception as e:
            self.log(f"❌ Failed to load tokenizer: {e}")
        self.is_running = False
        self.status_label.setText("🛌 Idle")
        self.update_button_states()

    def run_analysis_thread(self):
        if self.is_running:
            self.log("❌ Another task is already running.")
            return
        self.is_running = True
        self.status_label.setText("🔍 Analyzing...")
        self.update_button_states()
        threading.Thread(target=self.analyze_file, daemon=True).start()

    def run_clean_thread(self):
        if self.is_running:
            self.log("❌ Another task is already running.")
            return
        self.is_running = True
        self.status_label.setText("🪼 Cleaning...")
        self.update_button_states()
        threading.Thread(target=self.clean_file, daemon=True).start()

    def run_tokenize_thread(self):
        if self.is_running:
            self.log("❌ Another task is already running.")
            return
        self.is_running = True
        self.status_label.setText("🧵 Tokenizing...")
        self.update_button_states()
        threading.Thread(target=self.tokenize_only, daemon=True).start()

    def analyze_file(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        self.log("🔍 Analyzing token lengths...")
        result = self.core.analyze_file(self.core.file_path)
        self.log(result)
        self.is_running = False
        self.status_label.setText("🛌 Idle")
        self.update_button_states()

    def clean_file(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        try:
            max_tokens = int(self.max_token_entry.text())
        except ValueError:
            self.log("❌ Max token length must be an integer.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        result = self.core.clean_file(self.core.file_path, max_tokens)
        self.log(result)
        self.is_running = False
        self.status_label.setText("🛌 Idle")
        self.update_button_states()

    def tokenize_only(self):
        if not self.core.file_path:
            self.log("❌ Please select a .jsonl file first.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        if not self.core.tokenizer:
            self.log("❌ Tokenizer not loaded.")
            self.is_running = False
            self.status_label.setText("🛌 Idle")
            self.update_button_states()
            return
        result = self.core.tokenize_only(self.core.file_path)
        self.log(result)
        self.is_running = False
        self.status_label.setText("🛌 Idle")
        self.update_button_states()

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
    if threading.current_thread() is not threading.main_thread():
        print("⚠️ Warning: Application not running in main thread!")
        sys.exit(1)
    app = QApplication(sys.argv)
    win = TokenMaxxerV3App()
    win.show()
    sys.exit(app.exec_())