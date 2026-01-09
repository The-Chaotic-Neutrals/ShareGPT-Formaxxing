import os
import shutil
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QGroupBox, QMessageBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from App.SafetensorMaxxer.safetensormaxxer import SafetensorMaxxer
from App.Other.BG import GalaxyBackgroundWidget
import sys
from App.Other.Theme import Theme


class WorkerSignals(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    info = pyqtSignal(str)


class SafetensorMaxxerApp(QWidget):
    def __init__(self, theme=None):
        super().__init__()

        self.theme = theme or Theme.DARK
        self.safetensor_maxxer = SafetensorMaxxer()
        self.safetensor_maxxer.output_folder = os.path.join(os.getcwd(), "safetensorfied")

        self.executor = None
        self.future = None

        self.signals = WorkerSignals()
        self.signals.log.connect(self.log_message)
        self.signals.finished.connect(self.conversion_finished)
        self.signals.error.connect(self.show_error)
        self.signals.warning.connect(self.show_warning)
        self.signals.info.connect(self.log_message)

        self.model_path = None

        # Background
        self.background = GalaxyBackgroundWidget(self)
        self.background.lower()

        self._setup_style()
        self._build_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'background'):
            self.background.resize(self.size())

    def _setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #e6e6fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            QLabel {
                background-color: transparent;
                color: #e6e6fa;
            }
            QLabel#titleLabel {
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 26px;
                font-weight: bold;
                color: #e6e6fa;
                padding: 0px;
                margin: 0px;
            }
            QLabel#subtitleLabel {
                font-size: 12px;
                color: #a0a0c0;
                padding: 0px;
                margin: 0px;
            }
            QGroupBox {
                background-color: rgba(10, 10, 30, 0.7);
                border: 1px solid rgba(100, 100, 180, 0.3);
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
                padding-top: 28px;
                font-weight: bold;
                font-size: 12pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 12px;
                background-color: rgba(30, 60, 120, 0.6);
                border-radius: 8px;
                color: #c0c0ff;
                left: 12px;
            }
            QPushButton {
                background-color: rgba(30, 80, 160, 0.8);
                color: #ffffff;
                border: 1px solid rgba(100, 150, 255, 0.4);
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: rgba(50, 100, 200, 0.9);
                border: 1px solid rgba(150, 180, 255, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(20, 60, 140, 0.9);
            }
            QPushButton:disabled {
                background-color: rgba(60, 60, 80, 0.5);
                color: #707090;
                border: 1px solid rgba(80, 80, 100, 0.3);
            }
            QTextEdit {
                background-color: rgba(5, 5, 20, 0.8);
                color: #c8c8e8;
                border: 1px solid rgba(100, 100, 180, 0.3);
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
                selection-background-color: rgba(70, 100, 180, 0.6);
            }
            QLabel#statusBar {
                background-color: rgba(15, 15, 40, 0.8);
                color: #a0c0ff;
                padding: 10px 14px;
                border: 1px solid rgba(100, 100, 180, 0.3);
                border-radius: 8px;
                font-size: 10pt;
            }
        """)

    def _primary_button_style(self):
        return """
            QPushButton {
                background-color: rgba(30, 80, 160, 0.8);
                color: #ffffff;
                border: 1px solid rgba(100, 150, 255, 0.4);
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: rgba(50, 100, 200, 0.9);
                border: 1px solid rgba(150, 180, 255, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(20, 60, 140, 0.9);
            }
            QPushButton:disabled {
                background-color: rgba(60, 60, 80, 0.5);
                color: #707090;
                border: 1px solid rgba(80, 80, 100, 0.3);
            }
        """

    def _green_button_style(self):
        return """
            QPushButton {
                background-color: rgba(30, 140, 80, 0.8);
                color: #ffffff;
                border: 1px solid rgba(80, 200, 120, 0.4);
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: rgba(40, 160, 100, 0.9);
                border: 1px solid rgba(100, 220, 140, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(20, 120, 60, 0.9);
            }
            QPushButton:disabled {
                background-color: rgba(60, 60, 80, 0.5);
                color: #707090;
                border: 1px solid rgba(80, 80, 100, 0.3);
            }
        """

    def _purple_button_style(self):
        return """
            QPushButton {
                background-color: rgba(100, 60, 160, 0.8);
                color: #ffffff;
                border: 1px solid rgba(150, 100, 220, 0.4);
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: rgba(120, 80, 180, 0.9);
                border: 1px solid rgba(170, 120, 240, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(80, 40, 140, 0.9);
            }
            QPushButton:disabled {
                background-color: rgba(60, 60, 80, 0.5);
                color: #707090;
                border: 1px solid rgba(80, 80, 100, 0.3);
            }
        """

    def _build_ui(self):
        self.setWindowTitle("SafetensorMaxxer")
        self.setMinimumSize(800, 650)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        header_layout.setContentsMargins(0, 0, 0, 8)

        title_label = QLabel("🔒 SafetensorMaxxer")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("Convert PyTorch models to Safetensor format & verify index files")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: rgba(100, 100, 180, 0.3); max-height: 1px;")
        main_layout.addWidget(separator)

        # Model Tools group box
        tools_group = QGroupBox("🛠️ Model Tools")
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.setSpacing(16)
        tools_layout.setContentsMargins(16, 16, 16, 16)

        self.select_button = QPushButton("📁 Select Model Folder")
        self.select_button.setStyleSheet(self._primary_button_style())
        self.select_button.clicked.connect(self.select_model_path)
        self.select_button.setCursor(Qt.PointingHandCursor)

        self.convert_button = QPushButton("⚙️ Start Conversion")
        self.convert_button.setStyleSheet(self._green_button_style())
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_button.setCursor(Qt.PointingHandCursor)

        self.verify_button = QPushButton("🔍 Verify Folder")
        self.verify_button.setStyleSheet(self._purple_button_style())
        self.verify_button.clicked.connect(self.select_verify_folder)
        self.verify_button.setCursor(Qt.PointingHandCursor)

        tools_layout.addStretch()
        tools_layout.addWidget(self.select_button)
        tools_layout.addWidget(self.convert_button)
        tools_layout.addWidget(self.verify_button)
        tools_layout.addStretch()

        main_layout.addWidget(tools_group)

        # Log Output group box
        log_group = QGroupBox("📋 Output Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(12, 12, 12, 12)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_output.setPlaceholderText("Select a model folder to get started...")
        log_layout.addWidget(self.log_output)

        main_layout.addWidget(log_group, stretch=1)

        # Status bar
        self.status_bar = QLabel("✨ Ready - Select a model folder to begin")
        self.status_bar.setObjectName("statusBar")
        self.status_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.status_bar)

    def log_message(self, message):
        self.status_bar.setText(message)
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()

    def select_model_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if not folder:
            return
        self.model_path = folder
        self.safetensor_maxxer.model_path = folder
        self.log_message(f"📁 Selected: {folder}")

    def start_conversion(self):
        if not getattr(self, "model_path", None):
            QMessageBox.warning(self, "Warning", "Please select a model folder first.")
            return

        self.convert_button.setEnabled(False)
        self.log_message("🚀 Conversion started...")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self.run_conversion)
        self.future.add_done_callback(self.conversion_done)

    def conversion_done(self, future):
        try:
            future.result()
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

    def run_conversion(self):
        path = self.model_path
        if not path:
            self.signals.log.emit("⚠️ No model folder selected.")
            return

        os.makedirs(self.safetensor_maxxer.output_folder, exist_ok=True)

        index_filename = os.path.join(path, "pytorch_model.bin.index.json")

        if os.path.exists(index_filename):
            self.signals.log.emit("📦 Converting sharded model...")
            self.safetensor_maxxer.index_filename = index_filename
            operations, errors = self.safetensor_maxxer.convert_multi_local(
                index_filename, path, self.safetensor_maxxer.output_folder, self.safetensor_maxxer.discard_names
            )
        else:
            self.signals.log.emit("📦 Converting single model...")
            pt_file = os.path.join(path, "pytorch_model.bin")
            sf_file = os.path.join(self.safetensor_maxxer.output_folder, "model.safetensors")
            operations, errors = self.safetensor_maxxer.convert_single_local(
                pt_file, sf_file, self.safetensor_maxxer.discard_names
            )

        for op in operations:
            self.signals.log.emit(f"✅ Converted: {op}")

        if errors:
            error_msg = "\n".join([f"{f}: {e}" for f, e in errors])
            self.signals.log.emit("⚠️ Conversion completed with errors.")
            self.signals.error.emit(error_msg)
        else:
            self.signals.log.emit("✅ Conversion successful.")

        self.copy_json_files()

        if hasattr(self.safetensor_maxxer, "verify_and_fix_index"):
            self.signals.log.emit("🔍 Verifying index...")
            issues = self.safetensor_maxxer.verify_and_fix_index()
            if issues:
                for issue in issues:
                    self.signals.log.emit(f"❗ {issue}")
                self.signals.warning.emit("\n".join(issues))
            else:
                self.signals.log.emit("✅ Index verified clean.")

        if hasattr(self.safetensor_maxxer, "show_token_info"):
            self.signals.log.emit("📨 Token Info:")
            self.safetensor_maxxer.show_token_info()
        if hasattr(self.safetensor_maxxer, "show_chat_preview"):
            self.signals.log.emit("💬 Chat Template:")
            self.safetensor_maxxer.show_chat_preview()

    def copy_json_files(self):
        path = self.model_path
        if not path:
            return
        for filename in os.listdir(path):
            if filename.endswith(".json") and filename != "pytorch_model.bin.index.json":
                src = os.path.join(path, filename)
                dst = os.path.join(self.safetensor_maxxer.output_folder, filename)
                try:
                    shutil.copy(src, dst)
                    self.signals.log.emit(f"📄 Copied JSON: {filename}")
                except Exception as e:
                    self.signals.log.emit(f"❌ Error copying {filename}: {e}")

    def conversion_finished(self):
        self.convert_button.setEnabled(True)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_warning(self, message):
        QMessageBox.warning(self, "Warning", message)

    def select_verify_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Safetensor Output Folder")
        if not folder:
            return

        self.log_message(f"📁 Verifying folder: {folder}")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self.verify_only, folder)
        self.future.add_done_callback(self.verify_done)

    def verify_done(self, future):
        try:
            future.result()
        except Exception as e:
            self.signals.error.emit(str(e))

    def verify_only(self, folder):
        self.safetensor_maxxer.output_folder = folder
        self.signals.log.emit("🔎 Verifying safetensors...")
        if hasattr(self.safetensor_maxxer, "verify_and_fix_index"):
            issues = self.safetensor_maxxer.verify_and_fix_index()
            if issues:
                for issue in issues:
                    self.signals.log.emit(f"❗ {issue}")
                self.signals.warning.emit("\n".join(issues))
            else:
                self.signals.log.emit("✅ All files verified successfully.")
        else:
            self.signals.log.emit("🚫 Core is missing verify_and_fix_index().")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SafetensorMaxxerApp(theme=Theme.DARK)
    window.show()
    sys.exit(app.exec_())
