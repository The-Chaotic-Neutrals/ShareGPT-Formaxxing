import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QTextEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QSizePolicy
)
from PyQt5.QtGui import QFont, QIcon, QColor, QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from multiprocessing import Process, Manager
from App.ParquetMaxxer.parquetmaxxer import jsonl_to_parquet_worker, parquet_to_jsonl_worker
from App.Other.Theme import Theme


class WorkerThread(QThread):
    update_preview = pyqtSignal(str)
    update_status = pyqtSignal(str, QColor)
    finished_signal = pyqtSignal(str)  # Pass output path on finish

    def __init__(self, files, worker_func):
        super().__init__()
        self.files = files
        self.worker_func = worker_func

    def run(self):
        manager = Manager()
        queue = manager.Queue()
        processes = [Process(target=self.worker_func, args=(path, queue)) for path in self.files]
        for p in processes:
            p.start()

        results = []
        while True:
            msg = queue.get()
            if isinstance(msg, tuple):
                results.append(msg)
                if len(results) == len(self.files):
                    break

        preview_text = ""
        output_paths = []
        # Unpack results now expecting 4 elements
        for file_path, out_path, preview, error in results:
            filename = os.path.basename(file_path)
            if error:
                preview_text += f"❌ {filename} failed: {error}\n\n"
            else:
                preview_text += f"✅ {filename}\n{preview}\n\n"
                output_paths.append(out_path)

        self.update_preview.emit(preview_text)
        self.update_status.emit(preview_text, QColor("#FFFFFF"))

        # Show the first output path in status, or fallback
        out_path = output_paths[0] if output_paths else "No output path available"
        self.finished_signal.emit(out_path)


class ParquetMaxxer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParquetMaxxer — JSONL ⇄ Parquet Converter")
        self.resize(960, 680)
        self.theme = Theme.DARK
        self.setStyleSheet(f"background-color: {self.theme['bg']};")
        self.init_ui()

        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            try:
                self.setWindowIcon(QIcon(str(icon_path)))
            except Exception as e:
                print(f"Could not set icon: {e}")

    def init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {self.theme['bg']};")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        central_widget.setLayout(main_layout)

        self.title_label = QLabel("🧬 ParquetMaxxer")
        self.title_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {self.theme['fg']};")
        self.title_label.setAlignment(Qt.AlignCenter)  # type: ignore
        main_layout.addWidget(self.title_label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        main_layout.addLayout(button_layout)

        self.convert_btn = QPushButton("📥 Convert JSONL ➜ Parquet")
        self._style_button(self.convert_btn)
        self.convert_btn.clicked.connect(self.convert_files)
        button_layout.addWidget(self.convert_btn)

        self.reverse_btn = QPushButton("📤 Convert Parquet ➜ JSONL")
        self._style_button(self.reverse_btn)
        self.reverse_btn.clicked.connect(self.revert_files)
        button_layout.addWidget(self.reverse_btn)

        self.preview_label = QLabel("📝 Preview Panel")
        self.preview_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.preview_label.setStyleSheet(f"color: {self.theme['fg']};")
        self.preview_label.setAlignment(Qt.AlignLeft)  # type: ignore
        main_layout.addWidget(self.preview_label)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setFont(QFont("Consolas", 11))
        self.preview_box.setLineWrapMode(QTextEdit.NoWrap)
        self.preview_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_box.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.theme['text_bg']};
                color: {self.theme['text_fg']};
                border: 1px solid #666666;
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        main_layout.addWidget(self.preview_box)

        self.status_label = QLabel("💤 Waiting for action...")
        self.status_label.setFont(QFont("Consolas", 12))
        self.status_label.setStyleSheet(f"color: {self.theme['text_fg']}; background-color: {self.theme['bg']};")
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # type: ignore
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(100)
        main_layout.addWidget(self.status_label)

    def _style_button(self, button):
        button.setFixedSize(280, 45)
        button.setFont(QFont("Arial", 14))
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme['button_bg']};
                color: {self.theme['button_fg']};
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #333;
            }}
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 160))
        button.setGraphicsEffect(shadow)

    def run_with_multiprocessing(self, files, worker_func):
        self.status_label.setText("⏳ Processing...")
        self.status_label.setStyleSheet(f"color: {self.theme['fg']}; background-color: {self.theme['bg']};")

        self.worker_thread = WorkerThread(files, worker_func)
        self.worker_thread.update_preview.connect(self.update_preview)
        self.worker_thread.update_status.connect(self.update_status)
        self.worker_thread.finished_signal.connect(self.processing_finished)
        self.worker_thread.start()

    def update_preview(self, text):
        self.preview_box.clear()
        self.preview_box.setPlainText(text)
        self.preview_box.moveCursor(QTextCursor.Start)

    def update_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color.name()}; background-color: {self.theme['bg']};")

    def processing_finished(self, output_path):
        self.status_label.setText(f"🎉 Success! Output: {output_path}")
        self.status_label.setStyleSheet(f"color: #00FF00; background-color: {self.theme['bg']};")

    def convert_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select JSONL files", "", "JSONL files (*.jsonl)"
        )
        if files:
            self.run_with_multiprocessing(files, jsonl_to_parquet_worker)

    def revert_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Parquet files", "", "Parquet files (*.parquet)"
        )
        if files:
            self.run_with_multiprocessing(files, parquet_to_jsonl_worker)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ParquetMaxxer()
    window.show()
    sys.exit(app.exec_())
