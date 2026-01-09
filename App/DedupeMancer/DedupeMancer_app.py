"""
DedupeMancer App - Unified deduplication tool for datasets and images.

This combines:
- Dataset deduplication (JSONL files) using SHA-256 and MinHash/Semantic methods
- Image deduplication using SHA-256, dHash, and CLIP embeddings
"""

from __future__ import annotations
import os
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QProgressBar, QMessageBox, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox, QTabWidget,
    QGroupBox, QFormLayout, QScrollArea, QFrame, QMainWindow
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QDropEvent

from App.DedupeMancer.DedupeMancer import Deduplication
from App.DedupeMancer.ImageDedup import ImageDeduplication
from App.Other.BG import GalaxyBackgroundWidget


# =============================================================================
# Dataset Deduplication Workers and Widgets
# =============================================================================

class DatasetDedupWorker(QThread):
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
    """QListWidget that accepts drag & drop of .jsonl files."""

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
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_dataset_files"):
                parent = parent.parent()
            if parent is not None:
                parent._add_dataset_files(paths, source="drag-and-drop")
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


# =============================================================================
# Image Deduplication Workers and Widgets
# =============================================================================

class ImageDedupWorker(QThread):
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(object, object)
    phase_changed = pyqtSignal()

    def __init__(
        self,
        dedup: ImageDeduplication,
        inputs,
        report_path,
        method: str,  # "perceptual", "embeddings", or "text"
        move_duplicates: bool,
        duplicates_dir: str,
        copy_unique: bool,
        unique_output_dir: str,
        top_k: int,
        # Text dedup specific
        text_field: str = "text",
        filename_field: str = "file_name",
        metadata_file: str = "metadata.jsonl",
        images_subdir: str = "images",
    ):
        super().__init__()
        self.dedup = dedup
        self.inputs = inputs
        self.report_path = report_path
        self.method = method
        self.move_duplicates = move_duplicates
        self.duplicates_dir = duplicates_dir
        self.copy_unique = copy_unique
        self.unique_output_dir = unique_output_dir
        self.top_k = top_k
        self.text_field = text_field
        self.filename_field = filename_field
        self.metadata_file = metadata_file
        self.images_subdir = images_subdir

    def run(self):
        try:
            self.status_update.emit("🛠 Image deduplication started...")
            
            if self.method == "text":
                self.status_update.emit("Method: Text/Caption Similarity 📝 - Processing...")
                
                if not self.inputs:
                    self.status_update.emit("❌ No input folder selected.")
                    return
                
                input_dir = self.inputs[0]
                self.dedup.perform_text_metadata_dedup(
                    input_dir=input_dir,
                    output_dir=self.unique_output_dir,
                    report_path=self.report_path,
                    update_status=self.status_update.emit,
                    update_progress=self.progress_update.emit,
                    metadata_file=self.metadata_file,
                    images_subdir=self.images_subdir,
                    text_field=self.text_field,
                    filename_field=self.filename_field,
                )
                
            elif self.method == "embeddings":
                self.status_update.emit("Method: Image Embeddings 🧠 - Processing...")
                all_paths = self.dedup.iter_image_paths(self.inputs)
                all_groups = []
                
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
            else:
                self.status_update.emit("Method: Perceptual Hash 🖼️ - Processing...")
                all_paths = self.dedup.iter_image_paths(self.inputs)
                all_groups = []
                
                base_dir = os.path.dirname(self.report_path)
                sha_report = os.path.join(base_dir, "report_sha256.jsonl")
                dh_report = self.report_path

                self.status_update.emit("Phase 1/2: SHA-256 exact...")
                self.phase_changed.emit()
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
                self.phase_changed.emit()
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
            self.status_update.emit(f"❌ Failed: {type(e).__name__}: {e}")


class PathListWidget(QListWidget):
    """Drag & drop list that accepts image files and folders."""
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
            while parent is not None and not hasattr(parent, "_add_image_paths"):
                parent = parent.parent()
            if parent is not None:
                parent._add_image_paths(paths, source="drag-and-drop")
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


# =============================================================================
# Main Unified App
# =============================================================================

APP_TITLE = "DedupeMancer"

class DeduplicationApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        
        # Dataset deduplication state
        self.dataset_dedup = Deduplication()
        self.dataset_files = []
        self.current_file_index = 0
        self.dataset_worker = None
        self.dataset_start_time = None
        
        # Image deduplication state
        self.image_dedup = ImageDeduplication()
        self.image_inputs = []
        self.image_worker = None
        self.image_start_time = None
        self._image_worker_running = False

        self.setWindowTitle(f"{APP_TITLE} ⚒️")
        self.setMinimumSize(900, 700)
        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #F9FAFB;
                font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
                font-size: 11pt;
            }
            QLabel {
                color: #E5E7EB;
                background-color: transparent;
            }
            QGroupBox {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                margin-top: 18px;
                padding: 10px;
                background-color: rgba(5, 5, 15, 180);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #9CA3AF;
                font-weight: 600;
                font-size: 13pt;
            }
            QLineEdit {
                background-color: rgba(5, 5, 15, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 4px;
                padding: 5px 8px;
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QLineEdit:focus {
                border: 1px solid #2563EB;
            }
            QLineEdit::placeholder {
                color: #6B7280;
            }
            QListWidget {
                background-color: rgba(2, 2, 10, 220);
                color: #D1D5DB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #2563EB;
                color: #F9FAFB;
            }
            QListWidget::item:hover {
                background-color: rgba(37, 99, 235, 0.3);
            }
            QPushButton {
                background-color: rgba(2, 6, 23, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(17, 24, 39, 220);
                border-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: rgba(3, 7, 18, 240);
            }
            QPushButton:disabled {
                color: #6B7280;
                border-color: rgba(17, 24, 39, 200);
                background-color: rgba(2, 2, 2, 200);
            }
            QRadioButton {
                spacing: 8px;
                color: #E5E7EB;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #4B5563;
                background-color: transparent;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #2563EB;
                background-color: #2563EB;
                border-radius: 8px;
            }
            QCheckBox {
                spacing: 8px;
                color: #E5E7EB;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #4B5563;
                background-color: #000000;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #2563EB;
                background-color: #2563EB;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                background-color: rgba(2, 2, 10, 220);
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2563EB;
                border-radius: 7px;
            }
            QTabWidget::pane {
                border: 1px solid rgba(31, 41, 55, 200);
                background-color: rgba(5, 5, 15, 180);
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: rgba(5, 5, 15, 200);
                color: #9CA3AF;
                border: 1px solid rgba(31, 41, 55, 200);
                border-bottom: none;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: rgba(5, 5, 15, 180);
                color: #F9FAFB;
                border-color: #2563EB;
            }
            QTabBar::tab:hover {
                background-color: rgba(17, 24, 39, 200);
                color: #E5E7EB;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

    def resizeEvent(self, event):
        """Handle window resize to update background widget"""
        super().resizeEvent(event)
        if hasattr(self, 'galaxy_bg'):
            self.galaxy_bg.resize(self.size())

    def _build_ui(self):
        # Create galaxy background widget
        self.galaxy_bg = GalaxyBackgroundWidget(self)
        self.galaxy_bg.lower()
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Initial resize
        QTimer.singleShot(100, lambda: self.galaxy_bg.resize(self.size()) if hasattr(self, 'galaxy_bg') else None)

        # Header
        header_row = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #F9FAFB;")

        subtitle_label = QLabel("Dataset & Image Deduplication Tool")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 11pt;")

        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_row.addLayout(title_container)
        header_row.addStretch()
        main_layout.addLayout(header_row)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.dataset_tab = QWidget()
        self.image_tab = QWidget()
        
        self._build_dataset_tab()
        self._build_image_tab()

        self.tab_widget.addTab(self.dataset_tab, "📄 Dataset Dedup")
        self.tab_widget.addTab(self.image_tab, "🖼️ Image Dedup")

        main_layout.addWidget(self.tab_widget, stretch=1)

    # =========================================================================
    # Dataset Deduplication Tab
    # =========================================================================
    
    def _build_dataset_tab(self):
        layout = QVBoxLayout(self.dataset_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Input files group
        input_group = QGroupBox("📁 Input Files")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        input_group.setLayout(input_layout)

        input_label = QLabel("Drag .jsonl files here or use Add Files button:")
        input_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        input_layout.addWidget(input_label)

        list_row = QHBoxLayout()
        list_row.setSpacing(10)

        self.dataset_file_list = FileListWidget(self)
        self.dataset_file_list.setMinimumHeight(100)
        self.dataset_file_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_row.addWidget(self.dataset_file_list, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        add_btn = QPushButton("Add Files")
        add_btn.setFixedWidth(120)
        add_btn.clicked.connect(self.browse_dataset_files)
        btn_col.addWidget(add_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(120)
        clear_btn.setStyleSheet(self._red_button_style())
        clear_btn.clicked.connect(self.clear_dataset_files)
        btn_col.addWidget(clear_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        input_layout.addLayout(list_row)

        layout.addWidget(input_group)

        # Settings group
        settings_group = QGroupBox("⚙️ Deduplication Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(12)
        settings_group.setLayout(settings_layout)

        # Method selection
        method_row = QHBoxLayout()
        method_row.setSpacing(30)
        method_label = QLabel("Method:")
        method_label.setStyleSheet("font-weight: 500;")
        self.dataset_minhash_radio = QRadioButton("Min-Hash / Semantic")
        self.dataset_sha256_radio = QRadioButton("String-Match (SHA-256)")
        self.dataset_minhash_radio.setChecked(True)
        method_row.addWidget(method_label)
        method_row.addWidget(self.dataset_minhash_radio)
        method_row.addWidget(self.dataset_sha256_radio)
        method_row.addStretch()
        settings_layout.addLayout(method_row)

        # Thresholds
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(20)

        thresh_row.addWidget(QLabel("Jaccard:"))
        self.dataset_jaccard_input = QLineEdit(str(self.dataset_dedup.threshold))
        self.dataset_jaccard_input.setFixedWidth(70)
        thresh_row.addWidget(self.dataset_jaccard_input)

        thresh_row.addWidget(QLabel("Semantic:"))
        self.dataset_semantic_input = QLineEdit(str(self.dataset_dedup.semantic_threshold))
        self.dataset_semantic_input.setFixedWidth(70)
        thresh_row.addWidget(self.dataset_semantic_input)

        thresh_row.addWidget(QLabel("SimHash Prefix:"))
        self.dataset_prefix_input = QLineEdit(str(getattr(self.dataset_dedup, "prefix_bits", 16)))
        self.dataset_prefix_input.setFixedWidth(70)
        thresh_row.addWidget(self.dataset_prefix_input)

        thresh_row.addStretch()
        settings_layout.addLayout(thresh_row)

        layout.addWidget(settings_group)

        # Run button
        self.dataset_run_btn = QPushButton("🗑️ Remove Duplicates")
        self.dataset_run_btn.setFixedHeight(42)
        self.dataset_run_btn.setStyleSheet(self._primary_button_style())
        self.dataset_run_btn.clicked.connect(self.start_dataset_dedup)
        layout.addWidget(self.dataset_run_btn)

        # Progress group
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(8)
        progress_group.setLayout(progress_layout)

        status_row = QHBoxLayout()
        self.dataset_status_label = QLabel("Status: Ready")
        self.dataset_status_label.setStyleSheet("color: #9CA3AF;")
        self.dataset_progress_label = QLabel("0%")
        self.dataset_progress_label.setStyleSheet("color: #2563EB; font-weight: bold;")
        self.dataset_speed_label = QLabel("Speed: —")
        self.dataset_speed_label.setStyleSheet("color: #6B7280;")
        status_row.addWidget(self.dataset_status_label, stretch=3)
        status_row.addWidget(self.dataset_progress_label, stretch=1, alignment=Qt.AlignCenter)
        status_row.addWidget(self.dataset_speed_label, stretch=2, alignment=Qt.AlignRight)
        progress_layout.addLayout(status_row)

        self.dataset_progress_bar = QProgressBar()
        self.dataset_progress_bar.setValue(0)
        self.dataset_progress_bar.setFixedHeight(20)
        self.dataset_progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.dataset_progress_bar)

        layout.addWidget(progress_group)
        layout.addStretch()

    # =========================================================================
    # Image Deduplication Tab
    # =========================================================================

    def _build_image_tab(self):
        layout = QVBoxLayout(self.image_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Input group
        input_group = QGroupBox("📁 Input Images/Folders")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)
        input_group.setLayout(input_layout)

        input_label = QLabel("Drag images or folders here, or use the buttons:")
        input_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        input_layout.addWidget(input_label)

        list_row = QHBoxLayout()
        list_row.setSpacing(10)

        self.image_input_list = PathListWidget(self)
        self.image_input_list.setMinimumHeight(100)
        self.image_input_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_row.addWidget(self.image_input_list, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        add_img_btn = QPushButton("Add Images")
        add_img_btn.setFixedWidth(120)
        add_img_btn.clicked.connect(self.browse_images)
        btn_col.addWidget(add_img_btn)

        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.setFixedWidth(120)
        add_folder_btn.clicked.connect(self.browse_image_folder)
        btn_col.addWidget(add_folder_btn)

        clear_img_btn = QPushButton("Clear")
        clear_img_btn.setFixedWidth(120)
        clear_img_btn.setStyleSheet(self._red_button_style())
        clear_img_btn.clicked.connect(self.clear_image_inputs)
        btn_col.addWidget(clear_img_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        input_layout.addLayout(list_row)

        layout.addWidget(input_group)

        # Settings group
        settings_group = QGroupBox("⚙️ Deduplication Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(12)
        settings_group.setLayout(settings_layout)

        # Method selection
        method_row = QHBoxLayout()
        method_row.setSpacing(20)
        method_label = QLabel("Method:")
        method_label.setStyleSheet("font-weight: 500;")
        self.image_fast_radio = QRadioButton("Perceptual Hash")
        self.image_deep_radio = QRadioButton("Image Embeddings")
        self.image_text_radio = QRadioButton("Text/Caption")
        self.image_fast_radio.setChecked(True)
        
        self.image_fast_radio.toggled.connect(self._on_image_method_changed)
        self.image_deep_radio.toggled.connect(self._on_image_method_changed)
        self.image_text_radio.toggled.connect(self._on_image_method_changed)
        
        method_row.addWidget(method_label)
        method_row.addWidget(self.image_fast_radio)
        method_row.addWidget(self.image_deep_radio)
        method_row.addWidget(self.image_text_radio)
        method_row.addStretch()
        settings_layout.addLayout(method_row)

        # Image-based settings
        self.image_settings_widget = QWidget()
        img_settings_layout = QHBoxLayout(self.image_settings_widget)
        img_settings_layout.setContentsMargins(0, 0, 0, 0)
        img_settings_layout.setSpacing(20)

        img_settings_layout.addWidget(QLabel("Hamming:"))
        self.image_hamming_input = QLineEdit(str(self.image_dedup.dhash_hamming_threshold))
        self.image_hamming_input.setFixedWidth(60)
        img_settings_layout.addWidget(self.image_hamming_input)

        img_settings_layout.addWidget(QLabel("Prefix:"))
        self.image_prefix_input = QLineEdit(str(self.image_dedup.prefix_bits))
        self.image_prefix_input.setFixedWidth(60)
        img_settings_layout.addWidget(self.image_prefix_input)

        img_settings_layout.addWidget(QLabel("Cosine:"))
        self.image_cosine_input = QLineEdit(str(self.image_dedup.embedding_threshold))
        self.image_cosine_input.setFixedWidth(60)
        img_settings_layout.addWidget(self.image_cosine_input)

        img_settings_layout.addWidget(QLabel("Top-K:"))
        self.image_topk_input = QLineEdit("30")
        self.image_topk_input.setFixedWidth(60)
        img_settings_layout.addWidget(self.image_topk_input)

        img_settings_layout.addStretch()
        settings_layout.addWidget(self.image_settings_widget)

        # Text-based settings
        self.text_settings_widget = QWidget()
        text_settings_layout = QVBoxLayout(self.text_settings_widget)
        text_settings_layout.setContentsMargins(0, 0, 0, 0)
        text_settings_layout.setSpacing(8)

        text_row1 = QHBoxLayout()
        text_row1.setSpacing(15)
        
        text_row1.addWidget(QLabel("Similarity:"))
        self.text_sim_input = QLineEdit(str(self.image_dedup.text_sim_threshold))
        self.text_sim_input.setFixedWidth(60)
        text_row1.addWidget(self.text_sim_input)

        text_row1.addWidget(QLabel("Text field:"))
        self.text_field_input = QLineEdit("text")
        self.text_field_input.setFixedWidth(100)
        text_row1.addWidget(self.text_field_input)

        text_row1.addWidget(QLabel("Filename field:"))
        self.filename_field_input = QLineEdit("file_name")
        self.filename_field_input.setFixedWidth(100)
        text_row1.addWidget(self.filename_field_input)

        text_row1.addStretch()
        text_settings_layout.addLayout(text_row1)

        text_row2 = QHBoxLayout()
        text_row2.setSpacing(15)

        text_row2.addWidget(QLabel("Metadata file:"))
        self.metadata_file_input = QLineEdit("metadata.jsonl")
        self.metadata_file_input.setFixedWidth(120)
        text_row2.addWidget(self.metadata_file_input)

        text_row2.addWidget(QLabel("Images subdir:"))
        self.images_subdir_input = QLineEdit("images")
        self.images_subdir_input.setFixedWidth(100)
        text_row2.addWidget(self.images_subdir_input)

        text_row2.addStretch()
        text_settings_layout.addLayout(text_row2)

        text_note = QLabel("📌 For HuggingFace datasets: folder with metadata.jsonl + images/ subfolder")
        text_note.setStyleSheet("color: #6B7280; font-size: 10pt;")
        text_settings_layout.addWidget(text_note)

        settings_layout.addWidget(self.text_settings_widget)
        self.text_settings_widget.setVisible(False)

        # Options
        options_row = QHBoxLayout()
        options_row.setSpacing(20)
        self.image_copy_unique_cb = QCheckBox("Copy unique images to output")
        self.image_copy_unique_cb.setChecked(True)
        self.image_move_dups_cb = QCheckBox("Move duplicates to separate folder")
        self.image_move_dups_cb.setChecked(False)
        options_row.addWidget(self.image_copy_unique_cb)
        options_row.addWidget(self.image_move_dups_cb)
        options_row.addStretch()
        settings_layout.addLayout(options_row)

        layout.addWidget(settings_group)

        # Run button
        self.image_run_btn = QPushButton("🧹 Deduplicate Images")
        self.image_run_btn.setFixedHeight(42)
        self.image_run_btn.setStyleSheet(self._primary_button_style())
        self.image_run_btn.clicked.connect(self.start_image_dedup)
        layout.addWidget(self.image_run_btn)

        # Progress group
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(8)
        progress_group.setLayout(progress_layout)

        status_row = QHBoxLayout()
        self.image_status_label = QLabel("Status: Ready")
        self.image_status_label.setStyleSheet("color: #9CA3AF;")
        self.image_progress_label = QLabel("0%")
        self.image_progress_label.setStyleSheet("color: #2563EB; font-weight: bold;")
        self.image_speed_label = QLabel("Speed: —")
        self.image_speed_label.setStyleSheet("color: #6B7280;")
        status_row.addWidget(self.image_status_label, stretch=3)
        status_row.addWidget(self.image_progress_label, stretch=1, alignment=Qt.AlignCenter)
        status_row.addWidget(self.image_speed_label, stretch=2, alignment=Qt.AlignRight)
        progress_layout.addLayout(status_row)

        self.image_progress_bar = QProgressBar()
        self.image_progress_bar.setValue(0)
        self.image_progress_bar.setFixedHeight(20)
        self.image_progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.image_progress_bar)

        layout.addWidget(progress_group)
        layout.addStretch()

    # =========================================================================
    # Styles
    # =========================================================================

    def _primary_button_style(self):
        return """
            QPushButton {
                background-color: #2563EB;
                color: #F9FAFB;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: rgba(37, 99, 235, 0.3);
                color: #6B7280;
            }
        """

    def _red_button_style(self):
        return """
            QPushButton {
                background-color: rgba(220, 38, 38, 0.8);
                color: #F9FAFB;
                border: 1px solid rgba(220, 38, 38, 0.5);
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(185, 28, 28, 0.9);
            }
        """

    # =========================================================================
    # Dataset Dedup Methods
    # =========================================================================

    def _add_dataset_files(self, file_paths, source="manual"):
        added = False
        for fp in file_paths:
            if fp.endswith('.jsonl') and fp not in self.dataset_files:
                self.dataset_files.append(fp)
                added = True
        if added:
            self._refresh_dataset_list()
            self._reset_dataset_progress()
            self.dataset_status_label.setText("Status: Ready" + (" (drag-and-drop)" if source == "drag-and-drop" else ""))

    def browse_dataset_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select JSONL Files", "", "JSONL files (*.jsonl)")
        if files:
            self._add_dataset_files(files)

    def clear_dataset_files(self):
        self.dataset_files.clear()
        self._refresh_dataset_list()
        self._reset_dataset_progress()
        self.dataset_status_label.setText("Status: Ready")

    def _refresh_dataset_list(self):
        self.dataset_file_list.clear()
        for fp in self.dataset_files:
            folder = os.path.basename(os.path.dirname(fp))
            fname = os.path.basename(fp)
            display = f"{folder}/{fname}" if folder else fname
            item = QListWidgetItem(f"📄 {display}")
            item.setToolTip(fp)
            self.dataset_file_list.addItem(item)

    def _reset_dataset_progress(self):
        self.dataset_progress_bar.setValue(0)
        self.dataset_progress_label.setText("0%")
        self.dataset_speed_label.setText("Speed: —")

    def start_dataset_dedup(self):
        if not self.dataset_files:
            QMessageBox.critical(self, "Error", "❌ Please select at least one .jsonl file.")
            return

        if self.dataset_minhash_radio.isChecked():
            try:
                jacc = float(self.dataset_jaccard_input.text())
                sem = float(self.dataset_semantic_input.text())
                prefix = int(self.dataset_prefix_input.text())
                if not (0.0 <= jacc <= 1.0):
                    raise ValueError("Jaccard must be 0-1")
                if not (0.0 <= sem <= 1.0):
                    raise ValueError("Semantic must be 0-1")
                if not (1 <= prefix <= 32):
                    raise ValueError("Prefix must be 1-32")
                self.dataset_dedup.threshold = jacc
                self.dataset_dedup.semantic_threshold = sem
                if hasattr(self.dataset_dedup, "prefix_bits"):
                    self.dataset_dedup.prefix_bits = prefix
            except ValueError as e:
                QMessageBox.critical(self, "Invalid Settings", f"❌ {e}")
                return

        self.current_file_index = 0
        self.dataset_run_btn.setEnabled(False)
        self._reset_dataset_progress()
        self.dataset_status_label.setText("Status: Starting...")
        self._process_next_dataset()

    def _process_next_dataset(self):
        if self.current_file_index >= len(self.dataset_files):
            self.dataset_run_btn.setEnabled(True)
            self.dataset_status_label.setText("Status: ✅ All files processed.")
            self.dataset_progress_bar.setValue(100)
            self.dataset_progress_label.setText("100%")
            return

        input_file = self.dataset_files[self.current_file_index]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        output_dir = os.path.join(repo_root, "outputs", "dedupemancer", "datasets")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base}-deduplicated.jsonl")

        self.dataset_start_time = time.time()
        use_minhash = self.dataset_minhash_radio.isChecked()
        self.dataset_worker = DatasetDedupWorker(self.dataset_dedup, input_file, output_file, use_minhash)
        self.dataset_worker.status_update.connect(lambda m: self.dataset_status_label.setText(f"Status: {m}"))
        self.dataset_worker.progress_update.connect(self._update_dataset_progress)
        self.dataset_worker.finished.connect(self._dataset_finished)
        self.dataset_worker.start()

    def _update_dataset_progress(self, current, total):
        if total == 0:
            pct = 0
        elif current >= total:
            pct = 100
        else:
            pct = (current / total) * 100
        self.dataset_progress_bar.setValue(int(pct))
        self.dataset_progress_label.setText(f"{pct:.1f}%")

        elapsed = time.time() - self.dataset_start_time if self.dataset_start_time else 1
        speed = current / elapsed if elapsed > 0 else 0
        self.dataset_speed_label.setText(f"Speed: {speed:.1f} it/s")

    def _dataset_finished(self):
        self.current_file_index += 1
        self._process_next_dataset()

    # =========================================================================
    # Image Dedup Methods
    # =========================================================================

    def _add_image_paths(self, paths, source="manual"):
        added = False
        for p in paths:
            if p not in self.image_inputs:
                self.image_inputs.append(p)
                added = True
        if added:
            self._refresh_image_list()
            self._reset_image_progress()
            self.image_status_label.setText("Status: Ready" + (" (drag-and-drop)" if source == "drag-and-drop" else ""))

    def browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)"
        )
        if files:
            self._add_image_paths(files)

    def browse_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._add_image_paths([folder])

    def clear_image_inputs(self):
        self.image_inputs.clear()
        self._refresh_image_list()
        self._reset_image_progress()
        self.image_status_label.setText("Status: Ready")

    def _refresh_image_list(self):
        self.image_input_list.clear()
        for p in self.image_inputs:
            pth = Path(p)
            display = pth.name if pth.name else str(pth)
            prefix = "📂 " if pth.is_dir() else "🖼️ "
            item = QListWidgetItem(prefix + display)
            item.setToolTip(str(pth))
            self.image_input_list.addItem(item)

    def _reset_image_progress(self):
        self.image_progress_bar.setValue(0)
        self.image_progress_label.setText("0%")
        self.image_speed_label.setText("Speed: —")

    def _on_image_method_changed(self):
        """Toggle settings visibility based on selected method."""
        is_text_mode = self.image_text_radio.isChecked()
        self.image_settings_widget.setVisible(not is_text_mode)
        self.text_settings_widget.setVisible(is_text_mode)

    def start_image_dedup(self):
        if not self.image_inputs:
            QMessageBox.critical(self, "Error", "❌ Please add at least one folder or image file.")
            return

        # Determine method
        if self.image_text_radio.isChecked():
            method = "text"
        elif self.image_deep_radio.isChecked():
            method = "embeddings"
        else:
            method = "perceptual"

        # Validate settings based on method
        try:
            if method == "text":
                text_sim = float(self.text_sim_input.text())
                if not (0.0 <= text_sim <= 1.0):
                    raise ValueError("Text similarity must be 0-1")
                self.image_dedup.text_sim_threshold = text_sim
                
                if len(self.image_inputs) != 1 or not Path(self.image_inputs[0]).is_dir():
                    QMessageBox.warning(self, "Warning", 
                        "Text/Caption mode works best with a single folder containing metadata.jsonl and images/")
            else:
                hamming = int(self.image_hamming_input.text())
                prefix = int(self.image_prefix_input.text())
                cosine = float(self.image_cosine_input.text())
                top_k = int(self.image_topk_input.text())

                if not (0 <= hamming <= 64):
                    raise ValueError("Hamming must be 0-64")
                if not (1 <= prefix <= 32):
                    raise ValueError("Prefix must be 1-32")
                if not (0.0 <= cosine <= 1.0):
                    raise ValueError("Cosine must be 0-1")
                if not (1 <= top_k <= 200):
                    raise ValueError("Top-K must be 1-200")

                self.image_dedup.dhash_hamming_threshold = hamming
                self.image_dedup.prefix_bits = prefix
                self.image_dedup.embedding_threshold = cosine
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Settings", f"❌ {e}")
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        
        if method == "text":
            out_dir = os.path.join(repo_root, "outputs", "dedupemancer", "text_dedup")
        else:
            out_dir = os.path.join(repo_root, "outputs", "dedupemancer", "images")
        os.makedirs(out_dir, exist_ok=True)

        report_path = os.path.join(out_dir, "report.jsonl")
        duplicates_dir = os.path.join(out_dir, "duplicates")
        unique_dir = os.path.join(out_dir, "unique")

        self.image_run_btn.setEnabled(False)
        self._reset_image_progress()
        self.image_status_label.setText("Status: Starting...")

        self.image_start_time = time.perf_counter()
        self._image_worker_running = True

        text_field = self.text_field_input.text() or "text"
        filename_field = self.filename_field_input.text() or "file_name"
        metadata_file = self.metadata_file_input.text() or "metadata.jsonl"
        images_subdir = self.images_subdir_input.text() or "images"
        top_k = int(self.image_topk_input.text()) if method != "text" else 30

        self.image_worker = ImageDedupWorker(
            dedup=self.image_dedup,
            inputs=self.image_inputs,
            report_path=report_path,
            method=method,
            move_duplicates=self.image_move_dups_cb.isChecked(),
            duplicates_dir=duplicates_dir,
            copy_unique=self.image_copy_unique_cb.isChecked(),
            unique_output_dir=unique_dir,
            top_k=top_k,
            text_field=text_field,
            filename_field=filename_field,
            metadata_file=metadata_file,
            images_subdir=images_subdir,
        )
        self.image_worker.status_update.connect(lambda m: self.image_status_label.setText(f"Status: {m}"))
        self.image_worker.progress_update.connect(self._update_image_progress)
        self.image_worker.phase_changed.connect(self._on_image_phase_changed)
        self.image_worker.finished.connect(self._image_finished)
        self.image_worker.start()

    def _on_image_phase_changed(self):
        self.image_start_time = time.perf_counter()
        self._reset_image_progress()

    def _update_image_progress(self, current, total):
        current = float(current) if current is not None else 0.0
        total = float(total) if total is not None else 1.0

        if total <= 0:
            pct = 0.0
        else:
            pct = (current / total) * 100.0
            if pct >= 100.0 and self._image_worker_running:
                pct = 99.9
        pct = max(0.0, min(pct, 100.0))

        self.image_progress_bar.setValue(int(pct))
        self.image_progress_label.setText(f"{pct:.1f}%")

        elapsed = time.perf_counter() - self.image_start_time if self.image_start_time else 1e-9
        elapsed = max(elapsed, 1e-9)
        current = max(0.0, current)
        speed = current / elapsed

        if speed >= 1_000_000:
            self.image_speed_label.setText(f"Speed: {speed/1_000_000:.1f} M/s")
        elif speed >= 1_000:
            self.image_speed_label.setText(f"Speed: {speed/1_000:.1f} K/s")
        else:
            self.image_speed_label.setText(f"Speed: {speed:.1f} it/s")

    def _image_finished(self):
        self._image_worker_running = False
        self.image_run_btn.setEnabled(True)
        self.image_progress_bar.setValue(100)
        self.image_progress_label.setText("100%")
        if self.image_copy_unique_cb.isChecked():
            self.image_status_label.setText("Status: ✅ Done! Check outputs/dedupemancer/")
        else:
            self.image_status_label.setText("Status: ✅ Done. Check outputs/dedupemancer/report.jsonl")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    window = DeduplicationApp(theme)
    window.show()
    sys.exit(app.exec_())
