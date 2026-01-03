from __future__ import annotations
import os
import time
from pathlib import Path
from typing import cast

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QProgressBar, QMessageBox, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDropEvent

from App.ImageDedupMancer.ImageDedupMancer import ImageDeduplication


class ImageDedupWorker(QThread):
    status_update = pyqtSignal(str)
    # Use 'object' type to avoid 32-bit int overflow with large byte counts
    progress_update = pyqtSignal(object, object)
    # Signal to notify phase changes (resets speed counter in UI)
    phase_changed = pyqtSignal()

    def __init__(
        self,
        dedup: ImageDeduplication,
        inputs,
        report_path,
        use_embeddings: bool,
        move_duplicates: bool,
        duplicates_dir: str,
        copy_unique: bool,
        unique_output_dir: str,
        top_k: int,
    ):
        super().__init__()
        self.dedup = dedup
        self.inputs = inputs
        self.report_path = report_path
        self.use_embeddings = use_embeddings
        self.move_duplicates = move_duplicates
        self.duplicates_dir = duplicates_dir
        self.copy_unique = copy_unique
        self.unique_output_dir = unique_output_dir
        self.top_k = top_k

    def run(self):
        try:
            self.status_update.emit("🛠 Image deduplication started...")
            method = "Embeddings 🧠" if self.use_embeddings else "Perceptual Hash 🖼️"
            self.status_update.emit(f"Method: {method} - Processing...")

            # Get all image paths first (needed for copy_unique)
            all_paths = self.dedup.iter_image_paths(self.inputs)
            all_groups = []

            if self.use_embeddings:
                groups = self.dedup.perform_embedding_dedup(
                    self.inputs,
                    self.report_path,
                    self.status_update.emit,
                    self.progress_update.emit,
                    top_k=self.top_k,
                    move_duplicates=self.move_duplicates,
                    duplicates_dir=self.duplicates_dir if self.move_duplicates else None,
                    precomputed_paths=all_paths,
                )
                all_groups.extend(groups)
            else:
                # Run SHA-256 exact first (fast win), then dHash near-dup
                base_dir = os.path.dirname(self.report_path)
                sha_report = os.path.join(base_dir, "report_sha256.jsonl")
                dh_report = self.report_path

                self.status_update.emit("Phase 1/2: SHA-256 exact...")
                self.phase_changed.emit()  # Reset speed timer for new phase
                sha_groups = self.dedup.perform_sha256_image_dedup(
                    self.inputs,
                    sha_report,
                    self.status_update.emit,
                    self.progress_update.emit,
                    move_duplicates=False,
                    duplicates_dir=None,
                    precomputed_paths=all_paths,
                )
                all_groups.extend(sha_groups)
                self.status_update.emit(f"SHA-256 found {len(sha_groups)} duplicate groups")

                self.status_update.emit("Phase 2/2: dHash near-duplicates...")
                self.phase_changed.emit()  # Reset speed timer for new phase
                dhash_groups = self.dedup.perform_dhash_dedup(
                    self.inputs,
                    dh_report,
                    self.status_update.emit,
                    self.progress_update.emit,
                    move_duplicates=self.move_duplicates,
                    duplicates_dir=self.duplicates_dir if self.move_duplicates else None,
                    precomputed_paths=all_paths,
                )
                all_groups.extend(dhash_groups)
                self.status_update.emit(f"Total: {len(all_groups)} duplicate groups from both methods")

            # Copy unique images to output directory if enabled
            if self.copy_unique and self.unique_output_dir:
                self.phase_changed.emit()
                self.status_update.emit("Copying unique images to output...")
                self.dedup.copy_unique_to_output(
                    all_paths,
                    all_groups,
                    self.unique_output_dir,
                    self.status_update.emit,
                    self.progress_update.emit,
                )

            self.progress_update.emit(1, 1)
            self.status_update.emit("✅ Image deduplication completed.")
        except Exception as e:
            self.status_update.emit(f"❌ Failed: {type(e).__name__}")


class PathListWidget(QListWidget):
    """
    Drag & drop list that accepts image files and folders.
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

    def __init__(self, parent=None):
        super().__init__(parent)
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
        mime_data = event.mimeData()
        if mime_data is None or not mime_data.hasUrls():
            event.ignore()
            return

        paths = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            p = Path(local_path)
            if p.is_dir():
                paths.append(str(p))
            elif p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                paths.append(str(p))

        if paths:
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_paths"):
                parent = parent.parent()
            if parent is not None:
                cast(ImageDeduplicationApp, parent)._add_paths(paths, source="drag-and-drop")
            event.acceptProposedAction()
        else:
            event.ignore()

    def _has_valid_urls(self, event) -> bool:
        mime_data = event.mimeData()
        if mime_data is None or not mime_data.hasUrls():
            return False
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            p = Path(url.toLocalFile())
            if p.is_dir():
                return True
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                return True
        return False


class ImageDeduplicationApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.dedup = ImageDeduplication()

        # Use monotonic timer for speed calcs
        self.start_time = None  # perf_counter()
        self.inputs = []
        self.worker = None
        self._worker_running = False  # used to avoid "100% while still running"

        self.setWindowTitle("ImageDedupMancer 🖼️⚒️")
        self.setStyleSheet(f"""
            background-color: {self.theme.get('bg', '#fff')};
            color: {self.theme.get('fg', '#000')};
        """)
        self.setMinimumWidth(700)
        self.setup_ui()

    def setup_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(18)

        font_label = QFont("Segoe UI", 14)
        font_button = QFont("Segoe UI", 14, QFont.Bold)
        font_settings = QFont("Segoe UI", 14)
        font_status = QFont("Segoe UI", 14)

        # Common input field style
        input_style = """
            QLineEdit {
                background-color: #2a2a2a;
                color: #fff;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #1e90ff;
            }
        """

        # -------- Input selection --------
        input_layout = QVBoxLayout()
        input_layout.setSpacing(12)

        lbl = QLabel("📁 Inputs (drag folders or images here):")
        lbl.setFont(font_label)
        input_layout.addWidget(lbl)

        list_buttons = QHBoxLayout()
        list_buttons.setSpacing(12)

        self.input_list = PathListWidget(self)
        self.input_list.setFont(QFont("Segoe UI", 14))
        self.input_list.setMinimumHeight(120)
        self.input_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_list.setAlternatingRowColors(True)
        self.input_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #555;
                border-radius: 8px;
                background-color: #1e1e1e;
            }
            QListWidget::item {
                padding: 6px 10px;
            }
            QListWidget::item:selected {
                background-color: #1e90ff;
                color: white;
            }
        """)
        list_buttons.addWidget(self.input_list, stretch=1)

        side = QVBoxLayout()
        side.setSpacing(10)

        add_files_btn = QPushButton("Add Images 🔎")
        add_files_btn.setFont(font_button)
        add_files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_files_btn.setStyleSheet(self._button_style())
        add_files_btn.setFixedWidth(180)
        add_files_btn.setFixedHeight(42)
        add_files_btn.clicked.connect(self.browse_images)
        side.addWidget(add_files_btn)

        add_folder_btn = QPushButton("Add Folder 📂")
        add_folder_btn.setFont(font_button)
        add_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_folder_btn.setStyleSheet(self._button_style())
        add_folder_btn.setFixedWidth(180)
        add_folder_btn.setFixedHeight(42)
        add_folder_btn.clicked.connect(self.browse_folder)
        side.addWidget(add_folder_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFont(font_button)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #e05050;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                border: none;
            }
            QPushButton:hover { background-color: #f06060; }
            QPushButton:pressed { background-color: #c04040; }
            QPushButton:disabled { background-color: #666666; color: #999999; }
        """)
        clear_btn.setFixedWidth(180)
        clear_btn.setFixedHeight(42)
        clear_btn.clicked.connect(self.clear_inputs)
        side.addWidget(clear_btn)

        side.addStretch()
        list_buttons.addLayout(side)

        input_layout.addLayout(list_buttons)
        main.addLayout(input_layout)

        # -------- Method selection --------
        method_layout = QHBoxLayout()
        method_layout.setSpacing(50)

        self.fast_radio = QRadioButton("Perceptual Hash (Fast) 🖼️")
        self.fast_radio.setFont(font_label)
        self.deep_radio = QRadioButton("Embeddings (Deep) 🧠")
        self.deep_radio.setFont(font_label)
        self.fast_radio.setChecked(True)

        method_layout.addStretch()
        method_layout.addWidget(self.fast_radio)
        method_layout.addWidget(self.deep_radio)
        method_layout.addStretch()
        main.addLayout(method_layout)

        # -------- Settings section --------
        settings_container = QVBoxLayout()
        settings_container.setSpacing(14)

        # Row 1: Hash settings
        hash_row = QHBoxLayout()
        hash_row.setSpacing(16)

        lbl_h = QLabel("dHash Hamming:")
        lbl_h.setFont(font_settings)
        self.hamming_input = QLineEdit(str(self.dedup.dhash_hamming_threshold))
        self.hamming_input.setFixedWidth(70)
        self.hamming_input.setFixedHeight(32)
        self.hamming_input.setStyleSheet(input_style)
        hash_row.addWidget(lbl_h)
        hash_row.addWidget(self.hamming_input)

        hash_row.addSpacing(20)

        lbl_p = QLabel("Prefix bits:")
        lbl_p.setFont(font_settings)
        self.prefix_input = QLineEdit(str(self.dedup.prefix_bits))
        self.prefix_input.setFixedWidth(70)
        self.prefix_input.setFixedHeight(32)
        self.prefix_input.setStyleSheet(input_style)
        hash_row.addWidget(lbl_p)
        hash_row.addWidget(self.prefix_input)

        hash_row.addSpacing(20)

        lbl_c = QLabel("Cosine threshold:")
        lbl_c.setFont(font_settings)
        self.cosine_input = QLineEdit(str(self.dedup.embedding_threshold))
        self.cosine_input.setFixedWidth(70)
        self.cosine_input.setFixedHeight(32)
        self.cosine_input.setStyleSheet(input_style)
        hash_row.addWidget(lbl_c)
        hash_row.addWidget(self.cosine_input)

        hash_row.addSpacing(20)

        lbl_k = QLabel("Top-K:")
        lbl_k.setFont(font_settings)
        self.topk_input = QLineEdit("30")
        self.topk_input.setFixedWidth(70)
        self.topk_input.setFixedHeight(32)
        self.topk_input.setStyleSheet(input_style)
        hash_row.addWidget(lbl_k)
        hash_row.addWidget(self.topk_input)

        hash_row.addStretch()
        settings_container.addLayout(hash_row)

        checkbox_style = """
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """

        # Row 2: Copy unique images option (main dedup output)
        self.copy_unique_checkbox = QCheckBox("Copy unique images to Outputs/image_dedup/unique")
        self.copy_unique_checkbox.setChecked(True)
        self.copy_unique_checkbox.setFont(font_settings)
        self.copy_unique_checkbox.setStyleSheet(checkbox_style)
        settings_container.addWidget(self.copy_unique_checkbox)

        # Row 3: Move duplicates option
        self.move_checkbox = QCheckBox("Also move duplicate files to Outputs/image_dedup/duplicates")
        self.move_checkbox.setChecked(False)
        self.move_checkbox.setFont(font_settings)
        self.move_checkbox.setStyleSheet(checkbox_style)
        settings_container.addWidget(self.move_checkbox)

        main.addLayout(settings_container)

        # -------- Run button --------
        self.run_btn = QPushButton("🧹 Deduplicate Images")
        self.run_btn.setFont(font_button)
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setStyleSheet(self._button_style())
        self.run_btn.setFixedHeight(46)
        self.run_btn.clicked.connect(self.start_dedup)
        main.addWidget(self.run_btn)

        # -------- Status + speed --------
        status_layout = QHBoxLayout()
        status_layout.setSpacing(20)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(font_status)
        self.progress_pct_label = QLabel("0%")
        self.progress_pct_label.setFont(font_status)
        self.speed_label = QLabel("Speed: 0 it/s")
        self.speed_label.setFont(font_status)

        status_layout.addWidget(self.status_label, stretch=3)
        status_layout.addWidget(self.progress_pct_label, stretch=1, alignment=Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.speed_label, stretch=2, alignment=Qt.AlignmentFlag.AlignRight)
        main.addLayout(status_layout)

        # -------- Progress bar --------
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFixedHeight(22)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                border-radius: 7px;
            }
        """)
        main.addWidget(self.progress)

    def _button_style(self):
        return """
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border-radius: 8px;
                padding: 10px 18px;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3aa0ff; }
            QPushButton:pressed { background-color: #1573cc; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """

    # -------- Input handling --------
    def _add_paths(self, paths, source="manual"):
        added = False
        for p in paths:
            if p not in self.inputs:
                self.inputs.append(p)
                added = True

        if added:
            self._refresh_list()
            self._reset_progress()
            self.update_status("Ready (added via drag-and-drop)" if source == "drag-and-drop" else "Ready")

    def browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)"
        )
        if files:
            self._add_paths(files, source="manual")

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._add_paths([folder], source="manual")

    def clear_inputs(self):
        self.inputs.clear()
        self._refresh_list()
        self._reset_progress()
        self.update_status("Ready")

    def _refresh_list(self):
        self.input_list.clear()
        for p in self.inputs:
            pth = Path(p)
            display = pth.name if pth.name else str(pth)
            prefix = "📂 " if pth.is_dir() else "🖼️ "
            item = QListWidgetItem(prefix + display)
            item.setToolTip(str(pth))
            self.input_list.addItem(item)

    # -------- Run flow --------
    def start_dedup(self):
        if not self.inputs:
            QMessageBox.critical(self, "Error", "❌ Please add at least one folder or image file.")
            return

        try:
            hamming = int(self.hamming_input.text())
            prefix_bits = int(self.prefix_input.text())
            cosine = float(self.cosine_input.text())
            top_k = int(self.topk_input.text())

            if hamming < 0 or hamming > 64:
                raise ValueError("dHash Hamming must be between 0 and 64.")
            if prefix_bits < 1 or prefix_bits > 32:
                raise ValueError("Prefix bits must be between 1 and 32.")
            if not (0.0 <= cosine <= 1.0):
                raise ValueError("Cosine threshold must be between 0 and 1.")
            if top_k < 1 or top_k > 200:
                raise ValueError("Top-K should be between 1 and 200.")

            self.dedup.dhash_hamming_threshold = hamming
            self.dedup.prefix_bits = prefix_bits
            self.dedup.embedding_threshold = cosine
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Settings", f"❌ {e}")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))  # Go up 2 levels: ImageDedupMancer -> App -> repo root
        out_dir = os.path.join(repo_root, "Outputs", "image_dedup")
        os.makedirs(out_dir, exist_ok=True)

        report_path = os.path.join(out_dir, "report.jsonl")
        duplicates_dir = os.path.join(out_dir, "duplicates")
        unique_output_dir = os.path.join(out_dir, "unique")

        self.run_btn.setEnabled(False)
        self._reset_progress()
        self.update_status("Starting image deduplication...")

        # ✅ monotonic clock
        self.start_time = time.perf_counter()
        self._worker_running = True

        use_embeddings = self.deep_radio.isChecked()
        move_dups = self.move_checkbox.isChecked()
        copy_unique = self.copy_unique_checkbox.isChecked()

        self.worker = ImageDedupWorker(
            dedup=self.dedup,
            inputs=self.inputs,
            report_path=report_path,
            use_embeddings=use_embeddings,
            move_duplicates=move_dups,
            duplicates_dir=duplicates_dir,
            copy_unique=copy_unique,
            unique_output_dir=unique_output_dir,
            top_k=top_k,
        )
        self.worker.status_update.connect(self.update_status)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.phase_changed.connect(self._on_phase_changed)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _reset_progress(self):
        self.progress.setValue(0)
        self.progress_pct_label.setText("0%")
        self.speed_label.setText("Speed: 0 it/s")

    def _on_phase_changed(self):
        """Reset timer when a new phase starts to get accurate per-phase speed."""
        self.start_time = time.perf_counter()
        self.progress.setValue(0)
        self.progress_pct_label.setText("0%")
        self.speed_label.setText("Speed: 0 it/s")

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_progress(self, current, total):
        # Ensure we're working with Python floats to avoid any overflow issues
        current = float(current) if current is not None else 0.0
        total = float(total) if total is not None else 1.0

        # percent computation
        if total <= 0:
            percent = 0.0
        else:
            percent = (current / total) * 100.0
            if percent >= 100.0 and self._worker_running:
                percent = 99.9  # don't show 100% until finished
        percent = max(0.0, min(percent, 100.0))  # clamp to valid range

        self.progress.setValue(int(percent))
        self.progress_pct_label.setText(f"{percent:.2f}%")

        # speed computation (monotonic, never negative)
        if self.start_time is None:
            elapsed = 1e-9
        else:
            elapsed = time.perf_counter() - self.start_time
        elapsed = max(elapsed, 1e-9)

        # Protect against negative/invalid current values
        current = max(0.0, current)
        speed = current / elapsed

        # Format speed nicely based on magnitude
        if speed >= 1_000_000:
            self.speed_label.setText(f"Speed: {speed/1_000_000:.2f} M/s")
        elif speed >= 1_000:
            self.speed_label.setText(f"Speed: {speed/1_000:.2f} K/s")
        else:
            self.speed_label.setText(f"Speed: {speed:.2f} it/s")

    def _finished(self):
        self._worker_running = False
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)
        self.progress_pct_label.setText("100%")
        if self.copy_unique_checkbox.isChecked():
            self.update_status("✅ Done! Unique images in Outputs/image_dedup/unique/")
        else:
            self.update_status("✅ Done. Check Outputs/image_dedup/report.jsonl")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    w = ImageDeduplicationApp(theme)
    w.show()
    sys.exit(app.exec_())
