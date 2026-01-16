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

    def __init__(self, deduplication, input_file, output_file, use_min_hash, randomize=False):
        super().__init__()
        self.deduplication = deduplication
        self.input_file = input_file
        self.output_file = output_file
        self.use_min_hash = use_min_hash
        self.randomize = randomize

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
                self.progress_update.emit,
                randomize=self.randomize
            )
        else:
            self.deduplication.perform_sha256_deduplication(
                self.input_file,
                self.output_file,
                self.status_update.emit,
                self.progress_update.emit,
                randomize=self.randomize
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
        # Parquet support
        parquet_files: list = None,
        export_parquet: bool = False,
        parquet_output_path: str = None,
        # HuggingFace dataset support
        hf_dataset: str = None,
        hf_token: str = None,
        hf_image_column: str = "image",
        hf_text_column: str = "text",
        hf_split: str = None,
        hf_max_samples: int = None,
        # Output options
        randomize_output: bool = False,
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
        # Parquet support
        self.parquet_files = parquet_files or []
        self.export_parquet = export_parquet
        self.parquet_output_path = parquet_output_path
        self.parquet_metadata_map = {}
        # HuggingFace dataset support
        self.hf_dataset = hf_dataset
        self.hf_token = hf_token
        self.hf_image_column = hf_image_column
        self.hf_text_column = hf_text_column
        self.hf_split = hf_split
        self.hf_max_samples = hf_max_samples
        self.hf_metadata_map = {}
        # Output options
        self.randomize_output = randomize_output

    def run(self):
        import tempfile
        import shutil
        
        try:
            self.status_update.emit("🛠 Image deduplication started...")
            
            # Handle parquet file extraction if any
            parquet_extracted_dir = None
            parquet_paths = []
            
            if self.parquet_files:
                self.status_update.emit(f"📦 Loading {len(self.parquet_files)} parquet file(s)...")
                self.phase_changed.emit()
                
                # Create temp dir for extracted images
                parquet_extracted_dir = tempfile.mkdtemp(prefix="dedup_parquet_")
                
                parquet_paths, self.parquet_metadata_map = self.dedup.load_parquet_images(
                    self.parquet_files,
                    parquet_extracted_dir,
                    self.status_update.emit,
                    self.progress_update.emit,
                )
                
                if not parquet_paths:
                    self.status_update.emit("⚠️ No images extracted from parquet files.")
                else:
                    self.status_update.emit(f"✅ Extracted {len(parquet_paths)} images from parquet")
            
            # Handle HuggingFace dataset extraction if specified
            hf_extracted_dir = None
            hf_paths = []
            
            if self.hf_dataset:
                self.status_update.emit(f"🤗 Loading HuggingFace dataset: {self.hf_dataset}...")
                self.phase_changed.emit()
                
                # Create temp dir for extracted images
                hf_extracted_dir = tempfile.mkdtemp(prefix="dedup_hf_")
                
                hf_paths, self.hf_metadata_map = self.dedup.load_hf_dataset_images(
                    self.hf_dataset,
                    hf_extracted_dir,
                    self.status_update.emit,
                    self.progress_update.emit,
                    image_column=self.hf_image_column,
                    text_column=self.hf_text_column,
                    split=self.hf_split,
                    token=self.hf_token,
                    max_samples=self.hf_max_samples,
                )
                
                if not hf_paths:
                    self.status_update.emit("⚠️ No images extracted from HuggingFace dataset.")
                else:
                    self.status_update.emit(f"✅ Extracted {len(hf_paths)} images from HuggingFace")
                    # Merge HF metadata with parquet metadata for unified handling
                    self.parquet_metadata_map.update(self.hf_metadata_map)
            
            if self.method == "text":
                self.status_update.emit("Method: Text/Caption Similarity 📝 - Processing...")
                
                if not self.inputs and not self.parquet_files and not self.hf_dataset:
                    self.status_update.emit("❌ No input folder, parquet files, or HF dataset selected.")
                    return
                
                # For parquet/HF with text dedup, we need to handle differently
                all_extracted_paths = parquet_paths + hf_paths
                if all_extracted_paths and self.parquet_metadata_map:
                    # Use parquet/HF metadata for text deduplication
                    self._run_parquet_text_dedup(all_extracted_paths, parquet_extracted_dir or hf_extracted_dir)
                elif self.inputs:
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
                
                # Combine regular inputs with parquet-extracted and HF images
                all_paths = self.dedup.iter_image_paths(self.inputs) if self.inputs else []
                all_paths.extend(parquet_paths)
                all_paths.extend(hf_paths)
                
                if not all_paths:
                    self.status_update.emit("❌ No images found.")
                    return
                
                all_groups = []
                
                groups = self.dedup.perform_embedding_dedup(
                    self.inputs if self.inputs else [],
                    self.report_path,
                    self.status_update.emit,
                    self.progress_update.emit,
                    top_k=self.top_k,
                    move_duplicates=self.move_duplicates,
                    duplicates_dir=self.duplicates_dir if self.move_duplicates else None,
                    precomputed_paths=all_paths,
                )
                all_groups.extend(groups)
                
                # Get unique paths after deduplication
                kept_paths = self._get_kept_paths(all_paths, all_groups)
                
                if self.copy_unique and self.unique_output_dir:
                    self.phase_changed.emit()
                    self.status_update.emit("Copying unique images to output...")
                    self.dedup.copy_unique_to_output(
                        all_paths,
                        all_groups,
                        self.unique_output_dir,
                        self.status_update.emit,
                        self.progress_update.emit,
                        randomize=self.randomize_output,
                    )
                
                # Export to parquet if requested
                if self.export_parquet and self.parquet_output_path and self.parquet_metadata_map:
                    self.phase_changed.emit()
                    self._export_to_parquet(kept_paths)
                    
            else:
                self.status_update.emit("Method: Perceptual Hash 🖼️ - Processing...")
                
                # Combine regular inputs with parquet-extracted and HF images
                all_paths = self.dedup.iter_image_paths(self.inputs) if self.inputs else []
                all_paths.extend(parquet_paths)
                all_paths.extend(hf_paths)
                
                if not all_paths:
                    self.status_update.emit("❌ No images found.")
                    return
                
                all_groups = []
                
                base_dir = os.path.dirname(self.report_path)
                sha_report = os.path.join(base_dir, "report_sha256.jsonl")
                dh_report = self.report_path

                self.status_update.emit("Phase 1/2: SHA-256 exact...")
                self.phase_changed.emit()
                sha_groups = self.dedup.perform_sha256_image_dedup(
                    self.inputs if self.inputs else [],
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
                    self.inputs if self.inputs else [],
                    dh_report,
                    self.status_update.emit,
                    self.progress_update.emit,
                    move_duplicates=self.move_duplicates,
                    duplicates_dir=self.duplicates_dir if self.move_duplicates else None,
                    precomputed_paths=all_paths,
                )
                all_groups.extend(dhash_groups)
                self.status_update.emit(f"Total: {len(all_groups)} duplicate groups from both methods")

                # Get unique paths after deduplication
                kept_paths = self._get_kept_paths(all_paths, all_groups)

                if self.copy_unique and self.unique_output_dir:
                    self.phase_changed.emit()
                    self.status_update.emit("Copying unique images to output...")
                    self.dedup.copy_unique_to_output(
                        all_paths,
                        all_groups,
                        self.unique_output_dir,
                        self.status_update.emit,
                        self.progress_update.emit,
                        randomize=self.randomize_output,
                    )
                
                # Export to parquet if requested
                if self.export_parquet and self.parquet_output_path and self.parquet_metadata_map:
                    self.phase_changed.emit()
                    self._export_to_parquet(kept_paths)

            self.progress_update.emit(1, 1)
            self.status_update.emit("✅ Image deduplication completed.")
            
            # Cleanup temp extraction directories
            if parquet_extracted_dir and os.path.exists(parquet_extracted_dir):
                try:
                    shutil.rmtree(parquet_extracted_dir)
                except Exception:
                    pass
            if hf_extracted_dir and os.path.exists(hf_extracted_dir):
                try:
                    shutil.rmtree(hf_extracted_dir)
                except Exception:
                    pass
                    
        except Exception as e:
            self.status_update.emit(f"❌ Failed: {type(e).__name__}: {e}")
    
    def _get_kept_paths(self, all_paths, all_groups):
        """Get list of paths to keep after deduplication."""
        duplicates_set = set()
        for paths in all_groups:
            keep = self.dedup._choose_keep(paths)
            for p in paths:
                if p != keep:
                    duplicates_set.add(p)
        return [p for p in all_paths if p not in duplicates_set]
    
    def _export_to_parquet(self, kept_paths):
        """Export deduplicated images to parquet format."""
        # Filter to only parquet-sourced images that have metadata
        parquet_kept = [p for p in kept_paths if p in self.parquet_metadata_map]
        
        if not parquet_kept:
            self.status_update.emit("⚠️ No parquet-sourced images to export.")
            return
        
        self.status_update.emit(f"📦 Exporting {len(parquet_kept)} images to parquet...")
        self.dedup.write_deduplicated_parquet(
            parquet_kept,
            self.parquet_metadata_map,
            self.parquet_output_path,
            self.status_update.emit,
            self.progress_update.emit,
            randomize=self.randomize_output,
        )
    
    def _run_parquet_text_dedup(self, parquet_paths, extracted_dir):
        """Run text-based deduplication on parquet-sourced images."""
        import numpy as np
        import shutil
        import json
        
        if not parquet_paths:
            self.status_update.emit("❌ No images from parquet to deduplicate.")
            return
        
        texts = []
        valid_paths = []
        for p in parquet_paths:
            meta = self.parquet_metadata_map.get(p, {})
            text = meta.get("text", "")
            if text and text.strip():
                texts.append(text.strip())
                valid_paths.append(p)
        
        if not texts:
            self.status_update.emit("❌ No text found in parquet metadata.")
            return
        
        self.status_update.emit(f"Encoding {len(texts)} texts for similarity...")
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.dedup.text_embed_model)
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=128)
        except ImportError:
            self.status_update.emit("❌ sentence_transformers required for text dedup.")
            return
        
        self.status_update.emit("Finding duplicates by text similarity...")
        kept_indices = []
        kept_embeddings = []
        threshold = float(self.dedup.text_sim_threshold)
        
        for i, emb in enumerate(embeddings):
            if kept_embeddings:
                sims = np.dot(kept_embeddings, emb)
                max_sim = float(np.max(sims))
                if max_sim >= threshold:
                    continue
            kept_indices.append(i)
            kept_embeddings.append(emb)
        
        kept_paths = [valid_paths[i] for i in kept_indices]
        removed = len(valid_paths) - len(kept_paths)
        
        self.status_update.emit(f"Keeping {len(kept_paths)} / {len(valid_paths)} (removed {removed} duplicates)")
        
        # Randomize order if requested
        if self.randomize_output:
            import random
            random.shuffle(kept_paths)
            self.status_update.emit("Shuffled output order")
        
        # Copy unique to output
        if self.copy_unique and self.unique_output_dir:
            os.makedirs(self.unique_output_dir, exist_ok=True)
            images_out = os.path.join(self.unique_output_dir, "images")
            os.makedirs(images_out, exist_ok=True)
            
            # Write metadata.jsonl
            meta_out = os.path.join(self.unique_output_dir, "metadata.jsonl")
            copied = 0
            with open(meta_out, 'w', encoding='utf-8') as f:
                for p in kept_paths:
                    meta = self.parquet_metadata_map.get(p, {})
                    fname = os.path.basename(p)
                    dst = os.path.join(images_out, fname)
                    try:
                        shutil.copy2(p, dst)
                        copied += 1
                        # Write metadata record - include all available metadata
                        record = {
                            "file_name": fname,
                            "text": meta.get("text", ""),
                        }
                        # Add prompt if present
                        if meta.get("prompt"):
                            record["prompt"] = meta.get("prompt")
                        # Add any other extra metadata (from HF datasets)
                        for key, val in meta.items():
                            if key not in ["text", "prompt", "source_parquet", "source_dataset", "row_idx", "file_name"]:
                                if isinstance(val, (str, int, float, bool)):
                                    record[key] = val
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
            
            self.status_update.emit(f"✅ Copied {copied} images to {self.unique_output_dir}")
        
        # Export to parquet if requested
        if self.export_parquet and self.parquet_output_path:
            self._export_to_parquet(kept_paths)


class PathListWidget(QListWidget):
    """Drag & drop list that accepts image files, folders, and parquet files."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
    PARQUET_EXTS = {".parquet"}

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
        parquet_paths = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            p = Path(local_path)
            if p.is_dir():
                paths.append(str(p))
            elif p.is_file():
                if p.suffix.lower() in self.IMAGE_EXTS:
                    paths.append(str(p))
                elif p.suffix.lower() in self.PARQUET_EXTS:
                    parquet_paths.append(str(p))

        if paths or parquet_paths:
            parent = self.parent()
            while parent is not None and not hasattr(parent, "_add_image_paths"):
                parent = parent.parent()
            if parent is not None:
                if paths:
                    parent._add_image_paths(paths, source="drag-and-drop")
                if parquet_paths:
                    parent._add_parquet_files(parquet_paths, source="drag-and-drop")
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
            if p.is_file() and p.suffix.lower() in self.PARQUET_EXTS:
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
        self.parquet_files = []  # List of parquet file paths
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

        # Options row
        dataset_options_row = QHBoxLayout()
        dataset_options_row.setSpacing(20)
        self.dataset_randomize_cb = QCheckBox("Randomize output order")
        self.dataset_randomize_cb.setChecked(False)
        self.dataset_randomize_cb.setToolTip("Shuffle the order of output records")
        dataset_options_row.addWidget(self.dataset_randomize_cb)
        dataset_options_row.addStretch()
        settings_layout.addLayout(dataset_options_row)

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

        input_label = QLabel("Drag images, folders, or parquet files here, or use the buttons:")
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

        add_parquet_btn = QPushButton("Add Parquet")
        add_parquet_btn.setFixedWidth(120)
        add_parquet_btn.setToolTip("Add parquet files with embedded images (HuggingFace format)")
        add_parquet_btn.clicked.connect(self.browse_parquet_files)
        btn_col.addWidget(add_parquet_btn)

        clear_img_btn = QPushButton("Clear")
        clear_img_btn.setFixedWidth(120)
        clear_img_btn.setStyleSheet(self._red_button_style())
        clear_img_btn.clicked.connect(self.clear_image_inputs)
        btn_col.addWidget(clear_img_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        input_layout.addLayout(list_row)

        layout.addWidget(input_group)

        # HuggingFace Dataset Input Group
        hf_group = QGroupBox("🤗 HuggingFace Dataset (Optional)")
        hf_layout = QVBoxLayout()
        hf_layout.setSpacing(8)
        hf_group.setLayout(hf_layout)
        
        # Enable checkbox
        self.image_use_hf_check = QCheckBox("Load from HuggingFace dataset")
        self.image_use_hf_check.setChecked(False)
        self.image_use_hf_check.toggled.connect(self._on_hf_mode_toggled)
        hf_layout.addWidget(self.image_use_hf_check)
        
        # HF settings container
        self.hf_settings_widget = QWidget()
        hf_settings_layout = QFormLayout(self.hf_settings_widget)
        hf_settings_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hf_settings_layout.setHorizontalSpacing(10)
        hf_settings_layout.setVerticalSpacing(4)
        
        self.hf_dataset_input = QLineEdit()
        self.hf_dataset_input.setPlaceholderText("e.g., username/dataset_name")
        self.hf_dataset_input.setToolTip("HuggingFace dataset name (e.g., 'laion/laion400m')")
        hf_settings_layout.addRow(QLabel("Dataset:"), self.hf_dataset_input)
        
        # Token input (optional)
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("(optional, for private datasets)")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        self.hf_token_input.setToolTip("HuggingFace token for private datasets")
        hf_settings_layout.addRow(QLabel("Token:"), self.hf_token_input)
        
        # Column settings row
        col_row = QHBoxLayout()
        col_row.setSpacing(10)
        
        col_row.addWidget(QLabel("Image col:"))
        self.hf_image_col_input = QLineEdit("image")
        self.hf_image_col_input.setFixedWidth(80)
        col_row.addWidget(self.hf_image_col_input)
        
        col_row.addWidget(QLabel("Text col:"))
        self.hf_text_col_input = QLineEdit("text")
        self.hf_text_col_input.setFixedWidth(80)
        col_row.addWidget(self.hf_text_col_input)
        
        col_row.addWidget(QLabel("Split:"))
        self.hf_split_input = QLineEdit()
        self.hf_split_input.setPlaceholderText("auto")
        self.hf_split_input.setFixedWidth(80)
        col_row.addWidget(self.hf_split_input)
        
        col_row.addStretch()
        hf_settings_layout.addRow("", col_row)
        
        # Max samples
        max_row = QHBoxLayout()
        max_row.setSpacing(10)
        self.hf_max_samples_input = QLineEdit()
        self.hf_max_samples_input.setPlaceholderText("All")
        self.hf_max_samples_input.setFixedWidth(100)
        self.hf_max_samples_input.setToolTip("Maximum samples to load (leave empty for all)")
        max_row.addWidget(QLabel("Max samples:"))
        max_row.addWidget(self.hf_max_samples_input)
        max_row.addStretch()
        hf_settings_layout.addRow("", max_row)
        
        hf_layout.addWidget(self.hf_settings_widget)
        self.hf_settings_widget.setVisible(False)
        
        layout.addWidget(hf_group)

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

        text_note = QLabel("📌 Supports: folder with metadata.jsonl + images/, parquet files, or HuggingFace datasets")
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
        self.image_export_parquet_cb = QCheckBox("Export to Parquet")
        self.image_export_parquet_cb.setChecked(False)
        self.image_export_parquet_cb.setToolTip("Export deduplicated images back to parquet format (for parquet inputs)")
        self.image_randomize_cb = QCheckBox("Randomize output order")
        self.image_randomize_cb.setChecked(False)
        self.image_randomize_cb.setToolTip("Shuffle the order of output images/records")
        options_row.addWidget(self.image_copy_unique_cb)
        options_row.addWidget(self.image_move_dups_cb)
        options_row.addWidget(self.image_export_parquet_cb)
        options_row.addWidget(self.image_randomize_cb)
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
        randomize = self.dataset_randomize_cb.isChecked()
        self.dataset_worker = DatasetDedupWorker(self.dataset_dedup, input_file, output_file, use_minhash, randomize=randomize)
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

    def _add_parquet_files(self, paths, source="manual"):
        added = False
        for p in paths:
            if p.lower().endswith('.parquet') and p not in self.parquet_files:
                self.parquet_files.append(p)
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

    def browse_parquet_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Parquet Files", "",
            "Parquet files (*.parquet)"
        )
        if files:
            self._add_parquet_files(files)

    def clear_image_inputs(self):
        self.image_inputs.clear()
        self.parquet_files.clear()
        self._refresh_image_list()
        self._reset_image_progress()
        self.image_status_label.setText("Status: Ready")

    def _refresh_image_list(self):
        self.image_input_list.clear()
        # Show regular image inputs
        for p in self.image_inputs:
            pth = Path(p)
            display = pth.name if pth.name else str(pth)
            prefix = "📂 " if pth.is_dir() else "🖼️ "
            item = QListWidgetItem(prefix + display)
            item.setToolTip(str(pth))
            self.image_input_list.addItem(item)
        # Show parquet files
        for p in self.parquet_files:
            pth = Path(p)
            display = pth.name if pth.name else str(pth)
            item = QListWidgetItem("📦 " + display)
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

    def _on_hf_mode_toggled(self, checked):
        """Toggle HuggingFace settings visibility."""
        self.hf_settings_widget.setVisible(checked)

    def start_image_dedup(self):
        # Get HuggingFace settings
        use_hf = self.image_use_hf_check.isChecked()
        hf_dataset = self.hf_dataset_input.text().strip() if use_hf else None
        hf_token = self.hf_token_input.text().strip() if use_hf else None
        hf_image_col = self.hf_image_col_input.text().strip() or "image"
        hf_text_col = self.hf_text_col_input.text().strip() or "text"
        hf_split = self.hf_split_input.text().strip() or None
        hf_max_samples_text = self.hf_max_samples_input.text().strip()
        hf_max_samples = int(hf_max_samples_text) if hf_max_samples_text.isdigit() else None
        
        if not self.image_inputs and not self.parquet_files and not hf_dataset:
            QMessageBox.critical(self, "Error", "❌ Please add at least one folder, image file, parquet file, or HuggingFace dataset.")
            return
        
        if use_hf and not hf_dataset:
            QMessageBox.critical(self, "Error", "❌ Please enter a HuggingFace dataset name.")
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
                
                # Text mode can work with parquet files or HF datasets
                if not self.parquet_files and not hf_dataset:
                    if len(self.image_inputs) != 1 or not Path(self.image_inputs[0]).is_dir():
                        QMessageBox.warning(self, "Warning", 
                            "Text/Caption mode works best with a single folder containing metadata.jsonl and images/, parquet files, or a HuggingFace dataset.")
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
        
        # Parquet export path
        export_parquet = self.image_export_parquet_cb.isChecked() and self.parquet_files
        parquet_output_path = os.path.join(out_dir, "deduplicated.parquet") if export_parquet else None

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
            # Parquet support
            parquet_files=self.parquet_files,
            export_parquet=export_parquet,
            parquet_output_path=parquet_output_path,
            # HuggingFace support
            hf_dataset=hf_dataset,
            hf_token=hf_token,
            hf_image_column=hf_image_col,
            hf_text_column=hf_text_col,
            hf_split=hf_split,
            hf_max_samples=hf_max_samples,
            # Output options
            randomize_output=self.image_randomize_cb.isChecked(),
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
        
        output_hint = "outputs/dedupemancer/"
        extras = []
        if self.image_export_parquet_cb.isChecked() and self.parquet_files:
            extras.append("parquet")
        if self.image_use_hf_check.isChecked():
            extras.append("HF")
        if extras:
            output_hint += f" (incl. {', '.join(extras)})"
        
        if self.image_copy_unique_cb.isChecked():
            self.image_status_label.setText(f"Status: ✅ Done! Check {output_hint}")
        else:
            self.image_status_label.setText(f"Status: ✅ Done. Check {output_hint}report.jsonl")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    window = DeduplicationApp(theme)
    window.show()
    sys.exit(app.exec_())
