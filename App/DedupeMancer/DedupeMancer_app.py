import os
import time
import logging
from App.DedupeMancer.DedupeMancer import Deduplication
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QProgressBar, QMessageBox, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


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
        self.status_update.emit(f"🛠 Deduplication started for {self.input_file}...")
        method = "Min-Hash 🔍" if self.use_min_hash else "String-Match 🔗"
        self.status_update.emit(f"Method: {method} - Processing...")
        self.deduplication.duplicate_count = 0
        if self.use_min_hash:
            self.deduplication.perform_min_hash_deduplication(
                self.input_file,
                self.output_file,
                self.status_update.emit,
                self.progress_update.emit
            )
        else:
            self.deduplication.perform_sha256_deduplication(
                self.input_file,
                self.output_file,
                self.status_update.emit,
                self.progress_update.emit
            )
        self.progress_update.emit(1, 1)
        self.status_update.emit(f"✅ Deduplication completed for {self.input_file}.")


class FileListWidget(QListWidget):
    """
    QListWidget that accepts drag & drop of .jsonl files and notifies the parent.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # IMPORTANT: allow dropping from outside, but no internal drag-reorder
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

    def dropEvent(self, event):
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
            # Call a method on the parent widget to actually add the files
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_files"):
                parent = parent.parent()
            if parent is not None:
                parent._add_files(paths, source="drag-and-drop")

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


class DeduplicationApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.deduplication = Deduplication()
        self.start_time = None
        self.input_files = []  # List to store multiple input file paths
        self.current_file_index = 0  # Track current file being processed
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

        # ----------------- Input file selection -----------------
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)

        lbl_input = QLabel("📁 Input Files (drag .jsonl here):")
        lbl_input.setFont(font_label)
        file_layout.addWidget(lbl_input)

        list_buttons_layout = QHBoxLayout()
        list_buttons_layout.setSpacing(10)

        # List widget (compact, cleaner, drag-and-drop)
        self.input_file_list = FileListWidget(self)
        file_list_font = QFont("Arial", 11)
        self.input_file_list.setFont(file_list_font)
        self.input_file_list.setMinimumHeight(80)
        self.input_file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.input_file_list.setAlternatingRowColors(True)
        self.input_file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #555;
                border-radius: 6px;
            }
            QListWidget::item {
                padding: 2px 6px;
            }
            QListWidget::item:selected {
                background-color: #1e90ff;
                color: white;
            }
        """)
        list_buttons_layout.addWidget(self.input_file_list, stretch=1)

        # Side buttons (Add / Clear)
        side_button_layout = QVBoxLayout()
        side_button_layout.setSpacing(8)

        browse_button = QPushButton("Add Files 🔎")
        browse_button.setFont(font_button)
        browse_button.setCursor(Qt.PointingHandCursor)
        browse_button.setStyleSheet(self._button_style())
        browse_button.setFixedWidth(150)
        browse_button.clicked.connect(self.browse_input_files)
        side_button_layout.addWidget(browse_button)

        clear_button = QPushButton("Clear Files")
        clear_button.setFont(font_button)
        clear_button.setCursor(Qt.PointingHandCursor)
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc3333;
            }
            QPushButton:disabled {
                background-color: #999999;
                color: #666666;
            }
        """)
        clear_button.setFixedWidth(150)
        clear_button.clicked.connect(self.clear_input_files)
        side_button_layout.addWidget(clear_button)

        side_button_layout.addStretch()
        list_buttons_layout.addLayout(side_button_layout)

        file_layout.addLayout(list_buttons_layout)
        main_layout.addLayout(file_layout)

        # ----------------- Method selection -----------------
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

        # ----------------- Settings row (thresholds, prefix bits) -----------------
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(20)

        # MinHash Jaccard threshold
        lbl_jaccard = QLabel("MinHash Jaccard:")
        lbl_jaccard.setFont(font_status)
        self.jaccard_input = QLineEdit(str(self.deduplication.threshold))
        self.jaccard_input.setFixedWidth(80)
        settings_layout.addWidget(lbl_jaccard)
        settings_layout.addWidget(self.jaccard_input)

        # Semantic cosine threshold
        lbl_sem = QLabel("Semantic cosine:")
        lbl_sem.setFont(font_status)
        self.semantic_input = QLineEdit(str(self.deduplication.semantic_threshold))
        self.semantic_input.setFixedWidth(80)
        settings_layout.addWidget(lbl_sem)
        settings_layout.addWidget(self.semantic_input)

        # SimHash prefix bits
        lbl_prefix = QLabel("SimHash prefix bits:")
        lbl_prefix.setFont(font_status)
        prefix_default = getattr(self.deduplication, "prefix_bits", 16)
        self.prefix_bits_input = QLineEdit(str(prefix_default))
        self.prefix_bits_input.setFixedWidth(80)
        settings_layout.addWidget(lbl_prefix)
        settings_layout.addWidget(self.prefix_bits_input)

        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)

        # ----------------- Dedup button -----------------
        self.dedup_button = QPushButton("🗑️ Remove Duplicates")
        self.dedup_button.setFont(font_button)
        self.dedup_button.setCursor(Qt.PointingHandCursor)
        self.dedup_button.setStyleSheet(self._button_style())
        self.dedup_button.setFixedHeight(40)
        self.dedup_button.clicked.connect(self.start_deduplication)
        main_layout.addWidget(self.dedup_button)

        # ----------------- Status bar + speed -----------------
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

        # ----------------- Progress bar -----------------
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

    # ===================== File Handling =====================

    def _add_files(self, file_paths, source="manual"):
        added = False
        for file_path in file_paths:
            if file_path.endswith('.jsonl') and file_path not in self.input_files:
                self.input_files.append(file_path)
                added = True

        if added:
            self.update_input_file_list()
            self.progress_bar.setValue(0)
            self.progress_percent_label.setText("0%")
            self.speed_label.setText("Speed: 0 it/s")
            if source == "drag-and-drop":
                self.update_status("Ready (files added via drag-and-drop)")
            else:
                self.update_status("Ready")

    def browse_input_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Input Files",
            "",
            "JSONL files (*.jsonl)"
        )
        if file_paths:
            self._add_files(file_paths, source="manual")

    def clear_input_files(self):
        self.input_files.clear()
        self.update_input_file_list()
        self.progress_bar.setValue(0)
        self.progress_percent_label.setText("0%")
        self.speed_label.setText("Speed: 0 it/s")
        self.update_status("Ready")

    def update_input_file_list(self):
        self.input_file_list.clear()
        for file_path in self.input_files:
            folder = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)

            # Show compact "folder/filename.jsonl" or just filename if no folder
            display_text = f"{folder}/{file_name}" if folder else file_name

            item = QListWidgetItem(display_text)
            item.setToolTip(file_path)  # full path as tooltip
            self.input_file_list.addItem(item)

    # ===================== Deduplication Flow =====================

    def start_deduplication(self):
        if not self.input_files:
            QMessageBox.critical(self, "Error", "❌ Please select at least one .jsonl file.")
            return

        # Pull settings from UI → Deduplication instance
        if self.min_hash_radio.isChecked():
            try:
                jacc = float(self.jaccard_input.text())
                sem = float(self.semantic_input.text())
                prefix_bits = int(self.prefix_bits_input.text())

                if not (0.0 <= jacc <= 1.0):
                    raise ValueError("MinHash Jaccard must be between 0 and 1.")
                if not (0.0 <= sem <= 1.0):
                    raise ValueError("Semantic cosine must be between 0 and 1.")
                if prefix_bits <= 0 or prefix_bits > 32:
                    raise ValueError("Prefix bits should be between 1 and 32.")

                self.deduplication.threshold = jacc
                self.deduplication.semantic_threshold = sem
                # only set if the deduplication object supports it
                if hasattr(self.deduplication, "prefix_bits"):
                    self.deduplication.prefix_bits = prefix_bits

            except ValueError as e:
                QMessageBox.critical(self, "Invalid Settings", f"❌ {e}")
                return

        self.current_file_index = 0
        self.dedup_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_percent_label.setText("0%")
        self.speed_label.setText("Speed: 0 it/s")
        self.update_status("Starting deduplication...")
        self.process_next_file()

    def process_next_file(self):
        if self.current_file_index >= len(self.input_files):
            self.dedup_button.setEnabled(True)
            self.update_status("✅ All files processed.")
            self.progress_bar.setValue(100)
            self.progress_percent_label.setText("100%")
            return

        input_file = self.input_files[self.current_file_index]
        if not input_file.endswith('.jsonl'):
            self.update_status(f"❌ Skipping {input_file}: Invalid file type. Must be .jsonl.")
            self.current_file_index += 1
            self.process_next_file()
            return

        # Default to outputs folder in repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        output_dir = os.path.join(repo_root, "outputs", "deduplicated")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-deduplicated.jsonl")

        self.start_time = time.time()
        use_min_hash = self.min_hash_radio.isChecked()
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
        self.current_file_index += 1
        self.process_next_file()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    window = DeduplicationApp(theme)
    window.show()
    sys.exit(app.exec_())
