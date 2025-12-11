import os
import shutil
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QTextEdit, QGroupBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from App.SafetensorMaxxer.safetensormaxxer import SafetensorMaxxer
import sys
from App.Other.Theme import Theme


class WorkerSignals(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    warning = pyqtSignal(str)
    info = pyqtSignal(str)


class SafetensorMaxxerApp(QWidget):
    def __init__(self, theme):
        super().__init__()

        self.theme = theme
        self.safetensor_maxxer = SafetensorMaxxer()
        # Don't create folder until conversion actually starts
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

        self.init_ui()
        self.apply_theme()

    def apply_theme(self):
        # Apply theme colors to the entire window and widgets
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme.get('bg', '#222222')};
                color: {self.theme.get('fg', '#ffffff')};
                font-family: Segoe UI, Arial;
                font-size: 12pt;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {self.theme.get('button_bg', '#444444')};
                border-radius: 6px;
                margin-top: 8px;
                padding: 8px;
            }}
            QGroupBox:title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 4px;
                color: {self.theme.get('fg', '#ffffff')};
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#555555')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border: none;
                padding: 10px 16px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('button_hover_bg', '#666666')};
            }}
            QPushButton:disabled {{
                background-color: #777777;
                color: #aaaaaa;
            }}
            QTextEdit {{
                background-color: {self.theme.get('log_bg', '#333333')};
                color: {self.theme.get('log_fg', '#ffffff')};
                border: 1px solid {self.theme.get('button_bg', '#555555')};
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11pt;
                padding: 6px;
            }}
            QLabel#statusBar {{
                background-color: {self.theme.get('status_bar_bg', '#444444')};
                color: {self.theme.get('status_bar_fg', '#ffffff')};
                padding: 6px;
                border-top: 1px solid {self.theme.get('button_bg', '#555555')};
            }}
        """)

    def init_ui(self):
        self.setWindowTitle("Safetensor Maxxer")
        self.setMinimumSize(750, 600)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Buttons group box
        button_group = QGroupBox("Model Tools")
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_group.setLayout(button_layout)

        self.select_button = QPushButton("📁 Select Model Folder")
        self.select_button.clicked.connect(self.select_model_path)

        self.convert_button = QPushButton("⚙️ Start Conversion")
        self.convert_button.clicked.connect(self.start_conversion)

        self.verify_button = QPushButton("🔍 Verify Folder")
        self.verify_button.clicked.connect(self.select_verify_folder)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.verify_button)

        main_layout.addWidget(button_group)

        # Log output (read-only QTextEdit)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_output, stretch=1)

        # Status bar label
        self.status_bar = QLabel("Ready")
        self.status_bar.setObjectName("statusBar")
        self.status_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(self.status_bar)

        self.show()

    def log_message(self, message):
        self.status_bar.setText(message)
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()

    def select_model_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if not folder:
            QMessageBox.warning(self, "Input Folder", "No folder selected")
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

        # Create output folder only when conversion actually starts
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
            QMessageBox.warning(self, "Folder", "No folder selected.")
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
