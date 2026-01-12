"""
HuggingFace module - HuggingFace dataset/model downloader and uploader tab functionality
Handles HuggingFace dataset/model downloading and caption dataset uploading UI and logic
Includes parquet packing for efficient dataset uploads
"""
import os
import json
import math
import shutil
import tempfile
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QScrollArea,
    QFrame, QSpinBox, QCheckBox, QListWidget, QComboBox,
    QFileDialog
)
from PyQt5.QtCore import Qt


def build_huggingface_tab(main_window):
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    """Build the HuggingFace Dataset and Model Download tab UI"""
    hf_tab = QWidget()
    hf_layout = QVBoxLayout()
    hf_layout.setContentsMargins(0, 0, 0, 0)
    hf_tab.setLayout(hf_layout)
    
    hf_header = QHBoxLayout()
    main_window.hf_dataset_start_button = QPushButton("Start Dataset Download")
    main_window.hf_dataset_start_button.clicked.connect(lambda: start_hf_download(main_window))
    main_window.hf_dataset_stop_button = QPushButton("Stop Dataset")
    main_window.hf_dataset_stop_button.clicked.connect(lambda: stop_hf_download(main_window))
    main_window.hf_dataset_stop_button.setEnabled(False)
    main_window.hf_model_start_button = QPushButton("Start Model Download")
    main_window.hf_model_start_button.clicked.connect(lambda: start_hf_model_download(main_window))
    main_window.hf_model_stop_button = QPushButton("Stop Model")
    main_window.hf_model_stop_button.clicked.connect(lambda: stop_hf_model_download(main_window))
    main_window.hf_model_stop_button.setEnabled(False)
    hf_header.addStretch()
    hf_header.addWidget(main_window.hf_dataset_start_button)
    hf_header.addWidget(main_window.hf_dataset_stop_button)
    hf_header.addWidget(main_window.hf_model_start_button)
    hf_header.addWidget(main_window.hf_model_stop_button)
    hf_layout.addLayout(hf_header)
    
    hf_split = QHBoxLayout()
    hf_split.setSpacing(14)
    hf_layout.addLayout(hf_split, stretch=1)
    
    hf_left_scroll = QScrollArea()
    hf_left_scroll.setWidgetResizable(True)
    hf_left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    hf_left_scroll.setFrameShape(QFrame.NoFrame)
    
    hf_left_container = QWidget()
    hf_left_panel = QVBoxLayout()
    hf_left_panel.setSpacing(10)
    hf_left_container.setLayout(hf_left_panel)
    hf_left_scroll.setWidget(hf_left_container)
    
    hf_split.addWidget(hf_left_scroll, stretch=3)
    
    hf_right_container = QWidget()
    hf_right_panel = QVBoxLayout()
    hf_right_panel.setSpacing(10)
    hf_right_container.setLayout(hf_right_panel)
    
    hf_split.addWidget(hf_right_container, stretch=4)
    
    # HuggingFace Dataset Input
    hf_dataset_group = QGroupBox("📊 HuggingFace Dataset Downloader")
    hf_dataset_layout = QFormLayout()
    hf_dataset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_dataset_layout.setHorizontalSpacing(10)
    hf_dataset_layout.setVerticalSpacing(6)
    hf_dataset_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    hf_dataset_group.setLayout(hf_dataset_layout)

    main_window.hf_dataset_edit = QLineEdit()
    main_window.hf_dataset_edit.setPlaceholderText("e.g., dataset_name or org/dataset_name")
    hf_dataset_layout.addRow(QLabel("HF Dataset:"), main_window.hf_dataset_edit)

    # Output directory
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outputs_dir = os.path.join(repo_root, "outputs")
    default_output = os.path.join(outputs_dir, "hf_dataset")
    output_row, _ = create_file_browse_row(
        line_edit_name="hf_output_edit",
        placeholder_text="Output directory for downloaded dataset",
        default_text=default_output,
        on_browse_clicked=lambda: browse_hf_output(main_window)
    )
    main_window.hf_output_edit = output_row.itemAt(0).widget()
    hf_dataset_layout.addRow(QLabel("Output Directory:"), _wrap_row(output_row))
    
    hf_left_panel.addWidget(hf_dataset_group)
    
    # HuggingFace Model Downloader
    hf_model_group = QGroupBox("🕹️ HuggingFace Model Downloader")
    hf_model_layout = QFormLayout()
    hf_model_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_model_layout.setHorizontalSpacing(10)
    hf_model_layout.setVerticalSpacing(6)
    hf_model_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    hf_model_group.setLayout(hf_model_layout)

    main_window.hf_model_edit = QLineEdit()
    main_window.hf_model_edit.setPlaceholderText("e.g., model_name or org/model_name")
    hf_model_layout.addRow(QLabel("HF Model:"), main_window.hf_model_edit)

    # Output directory for model
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outputs_dir = os.path.join(repo_root, "outputs")
    default_model_output = os.path.join(outputs_dir, "hf_models")
    model_output_row, _ = create_file_browse_row(
        line_edit_name="hf_model_output_edit",
        placeholder_text="Output directory for downloaded model",
        default_text=default_model_output,
        on_browse_clicked=lambda: browse_hf_model_output(main_window)
    )
    main_window.hf_model_output_edit = model_output_row.itemAt(0).widget()
    hf_model_layout.addRow(QLabel("Output Directory:"), _wrap_row(model_output_row))
    
    hf_left_panel.addWidget(hf_model_group)
    
    # HuggingFace Caption Dataset Upload (Parquet Packing)
    hf_upload_group = QGroupBox("🚀 Caption Dataset Upload (Parquet)")
    hf_upload_main_layout = QVBoxLayout()
    hf_upload_group.setLayout(hf_upload_main_layout)

    # Parent directories list
    parent_dirs_label = QLabel("Source Directories (folders containing images/ + metadata.jsonl):")
    hf_upload_main_layout.addWidget(parent_dirs_label)
    
    main_window.hf_parent_dirs_list = QListWidget()
    main_window.hf_parent_dirs_list.setMinimumHeight(80)
    main_window.hf_parent_dirs_list.setToolTip("Add folders containing captioned datasets. Subfolders will be scanned automatically.")
    hf_upload_main_layout.addWidget(main_window.hf_parent_dirs_list)
    
    # Add/Remove buttons for parent dirs
    parent_dirs_btn_row = QHBoxLayout()
    main_window.hf_add_dir_btn = QPushButton("+ Add Directory")
    main_window.hf_add_dir_btn.clicked.connect(lambda: add_parent_directory(main_window))
    main_window.hf_remove_dir_btn = QPushButton("- Remove Selected")
    main_window.hf_remove_dir_btn.clicked.connect(lambda: remove_parent_directory(main_window))
    parent_dirs_btn_row.addWidget(main_window.hf_add_dir_btn)
    parent_dirs_btn_row.addWidget(main_window.hf_remove_dir_btn)
    parent_dirs_btn_row.addStretch()
    hf_upload_main_layout.addLayout(parent_dirs_btn_row)

    # Upload settings form
    hf_upload_layout = QFormLayout()
    hf_upload_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_upload_layout.setHorizontalSpacing(10)
    hf_upload_layout.setVerticalSpacing(6)
    hf_upload_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    main_window.hf_upload_repo_edit = QLineEdit()
    main_window.hf_upload_repo_edit.setPlaceholderText("e.g., username/dataset_name")
    main_window.hf_upload_repo_edit.setToolTip("HuggingFace dataset repo ID (must be like 'username/dataset_name')")
    hf_upload_layout.addRow(QLabel("HF Repo ID:"), main_window.hf_upload_repo_edit)

    main_window.hf_private_repo_check = QCheckBox("Private repository")
    main_window.hf_private_repo_check.setChecked(True)
    main_window.hf_private_repo_check.setToolTip("Create repository as private")
    hf_upload_layout.addRow("", main_window.hf_private_repo_check)

    # Parquet settings
    main_window.hf_shard_rows_spin = QSpinBox()
    main_window.hf_shard_rows_spin.setRange(500, 50000)
    main_window.hf_shard_rows_spin.setValue(5000)
    main_window.hf_shard_rows_spin.setMaximumWidth(100)
    main_window.hf_shard_rows_spin.setToolTip("Rows per parquet shard (5000 recommended)")
    shard_row = QHBoxLayout()
    shard_row.addWidget(main_window.hf_shard_rows_spin)
    shard_row.addStretch()
    hf_upload_layout.addRow(QLabel("Rows/Shard:"), _wrap_row(shard_row))

    main_window.hf_compression_combo = QComboBox()
    main_window.hf_compression_combo.addItems(["zstd", "snappy", "gzip", "none"])
    main_window.hf_compression_combo.setCurrentText("zstd")
    main_window.hf_compression_combo.setMaximumWidth(100)
    main_window.hf_compression_combo.setToolTip("Parquet compression (zstd recommended for best ratio)")
    compression_row = QHBoxLayout()
    compression_row.addWidget(main_window.hf_compression_combo)
    compression_row.addStretch()
    hf_upload_layout.addRow(QLabel("Compression:"), _wrap_row(compression_row))

    hf_upload_main_layout.addLayout(hf_upload_layout)

    # Upload buttons
    upload_btn_row = QHBoxLayout()
    main_window.hf_pack_button = QPushButton("Pack to Parquet")
    main_window.hf_pack_button.clicked.connect(lambda: start_parquet_pack(main_window))
    main_window.hf_pack_button.setToolTip("Pack datasets into parquet shards (local only)")
    main_window.hf_upload_button = QPushButton("Pack && Upload")
    main_window.hf_upload_button.clicked.connect(lambda: start_hf_upload(main_window))
    main_window.hf_upload_button.setToolTip("Pack datasets and upload to HuggingFace")
    main_window.hf_stop_upload_button = QPushButton("Stop")
    main_window.hf_stop_upload_button.clicked.connect(lambda: stop_hf_upload(main_window))
    main_window.hf_stop_upload_button.setEnabled(False)
    upload_btn_row.addWidget(main_window.hf_pack_button)
    upload_btn_row.addWidget(main_window.hf_upload_button)
    upload_btn_row.addWidget(main_window.hf_stop_upload_button)
    upload_btn_row.addStretch()
    hf_upload_main_layout.addLayout(upload_btn_row)

    hf_left_panel.addWidget(hf_upload_group)
    hf_left_panel.addStretch(1)

    # Right panel - Logs
    hf_progress_group, main_window.hf_log_view = create_log_view()
    hf_right_panel.addWidget(hf_progress_group, stretch=1)
    
    return hf_tab


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtWidgets import QSizePolicy
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def browse_hf_output(main_window):
    """Browse for HuggingFace output directory"""
    from PyQt5.QtWidgets import QFileDialog
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "outputs", "hf_dataset")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    current_path = main_window.hf_output_edit.text().strip()
    if current_path and os.path.exists(current_path):
        start_dir = current_path
    else:
        start_dir = default_path
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select output directory for HuggingFace dataset",
        start_dir,
    )
    if directory:
        main_window.hf_output_edit.setText(directory)


def hf_dataset_worker(dataset_name, output_dir, token, q=None, stop_flag=None):
    """Worker function to download and save HuggingFace dataset"""
    try:
        if q:
            q.put(("log", f"Starting HuggingFace dataset download: {dataset_name}"))
        
        # Check if datasets library is available
        try:
            from datasets import load_dataset, DatasetDict, IterableDataset
        except ImportError:
            if q:
                q.put(("error", "datasets library is required. Install with: pip install datasets"))
            return
        
        if q:
            q.put(("log", f"Output directory: {output_dir}"))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if stop_flag and stop_flag.is_set():
            if q:
                q.put(("stopped", "Download stopped by user"))
            return
        
        # Load dataset
        if q:
            if token:
                q.put(("log", f"⬇️  Downloading dataset: {dataset_name} (using token)"))
            else:
                q.put(("log", f"⬇️  Downloading dataset: {dataset_name}"))
            q.put(("log", "   (Dataset will be cached at ~/.cache/huggingface/datasets)"))
        
        token_param = token if token else None
        
        try:
            dataset_dict = load_dataset(dataset_name, token=token_param)
        except Exception as e:
            if q:
                q.put(("error", f"Failed to load dataset: {str(e)}"))
            return
        
        if stop_flag and stop_flag.is_set():
            if q:
                q.put(("stopped", "Download stopped by user"))
            return
        
        # Handle DatasetDict (multiple splits) or single Dataset
        if isinstance(dataset_dict, DatasetDict):
            if q:
                q.put(("log", f"Dataset has {len(dataset_dict)} splits: {', '.join(dataset_dict.keys())}"))
            
            # Save each split separately
            for split_name, dataset in dataset_dict.items():
                if stop_flag and stop_flag.is_set():
                    if q:
                        q.put(("stopped", "Download stopped by user"))
                    return
                
                output_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_{split_name}.jsonl")
                if q:
                    q.put(("log", f"Saving split '{split_name}' to {os.path.basename(output_file)}..."))
                
                _save_dataset_to_jsonl(dataset, output_file, q, stop_flag)
                
                if stop_flag and stop_flag.is_set():
                    if q:
                        q.put(("stopped", "Download stopped by user"))
                    return
        else:
            # Single dataset
            output_file = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}.jsonl")
            if q:
                q.put(("log", f"Saving dataset to {os.path.basename(output_file)}..."))
            
            _save_dataset_to_jsonl(dataset_dict, output_file, q, stop_flag)
        
        if stop_flag and stop_flag.is_set():
            if q:
                q.put(("stopped", "Download stopped by user"))
            return
        
        if q:
            q.put(("success", f"✅ Successfully downloaded and saved dataset: {dataset_name}"))
    
    except Exception as e:
        if q:
            import traceback
            q.put(("error", f"Error downloading dataset: {str(e)}"))
            q.put(("log", traceback.format_exc()))


def _save_dataset_to_jsonl(dataset, output_file, q=None, stop_flag=None):
    """Save a dataset to JSONL format"""
    try:
        from datasets import IterableDataset
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(dataset, IterableDataset):
                # For iterable datasets, we can't get length
                if q:
                    q.put(("log", "   Processing iterable dataset (streaming)..."))
                
                for item in dataset:
                    if stop_flag and stop_flag.is_set():
                        break
                    
                    # Convert to dict and handle special types
                    item_dict = {}
                    for key, value in item.items():
                        # Handle PIL Images and other non-serializable types
                        if hasattr(value, 'save'):  # PIL Image
                            # Skip images in JSONL - they're too large
                            continue
                        elif hasattr(value, '__dict__'):
                            item_dict[key] = str(value)
                        else:
                            try:
                                json.dumps(value)  # Test if serializable
                                item_dict[key] = value
                            except (TypeError, ValueError):
                                item_dict[key] = str(value)
                    
                    json.dump(item_dict, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
                    
                    if count % 1000 == 0:
                        if q:
                            q.put(("log", f"   Processed {count} examples..."))
            else:
                # Regular dataset with known length
                total = len(dataset)
                if q:
                    q.put(("log", f"   Saving {total} examples..."))
                
                for i, item in enumerate(dataset):
                    if stop_flag and stop_flag.is_set():
                        break
                    
                    # Convert to dict and handle special types
                    item_dict = {}
                    for key, value in item.items():
                        # Handle PIL Images and other non-serializable types
                        if hasattr(value, 'save'):  # PIL Image
                            # Skip images in JSONL - they're too large
                            continue
                        elif hasattr(value, '__dict__'):
                            item_dict[key] = str(value)
                        else:
                            try:
                                json.dumps(value)  # Test if serializable
                                item_dict[key] = value
                            except (TypeError, ValueError):
                                item_dict[key] = str(value)
                    
                    json.dump(item_dict, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
                    
                    if (i + 1) % 1000 == 0 or (i + 1) == total:
                        if q:
                            q.put(("log", f"   Saved {i + 1}/{total} examples..."))
        
        if q:
            q.put(("log", f"✅ Saved {count} examples to {os.path.basename(output_file)}"))
    
    except Exception as e:
        if q:
            q.put(("log", f"Error saving dataset: {str(e)}"))
        raise


def start_hf_download(main_window):
    """Start HuggingFace dataset download"""
    dataset_name = main_window.hf_dataset_edit.text().strip()
    output_dir = main_window.hf_output_edit.text().strip()
    token = main_window.hf_token_edit.text().strip() if hasattr(main_window, 'hf_token_edit') else None
    
    if not dataset_name:
        main_window._show_error("Please enter a HuggingFace dataset name.")
        return
    
    if not output_dir:
        main_window._show_error("Please select an output directory.")
        return
    
    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.hf_log_view.clear()
    main_window._append_hf_log("=== Starting HuggingFace Dataset Download ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Downloading dataset...")
    main_window.hf_dataset_start_button.setEnabled(False)
    main_window.hf_dataset_stop_button.setEnabled(True)
    
    # Create queue and stop flag
    main_window.hf_queue = queue.Queue()
    main_window.hf_stop_flag = threading.Event()
    
    # Start worker thread
    def worker_wrapper():
        try:
            hf_dataset_worker(
                dataset_name,
                output_dir,
                token,
                main_window.hf_queue,
                main_window.hf_stop_flag
            )
            if not main_window.hf_stop_flag.is_set():
                main_window.hf_queue.put(("success", f"Dataset download completed: {dataset_name}"))
        except Exception as e:
            import traceback
            main_window.hf_queue.put(("error", f"Download failed: {str(e)}"))
            main_window.hf_queue.put(("log", traceback.format_exc()))
    
    t = threading.Thread(target=worker_wrapper, daemon=True)
    t.start()
    main_window.timer.start()


def stop_hf_download(main_window):
    """Stop HuggingFace dataset download"""
    if hasattr(main_window, 'hf_stop_flag'):
        main_window.hf_stop_flag.set()
    if hasattr(main_window, '_append_hf_log'):
        main_window._append_hf_log("Stopping HuggingFace dataset download...")
    if hasattr(main_window, 'hf_dataset_stop_button'):
        main_window.hf_dataset_stop_button.setEnabled(False)


def start_hf_model_download(main_window):
    """Start HuggingFace model download"""
    model_name = main_window.hf_model_edit.text().strip()
    output_dir = main_window.hf_model_output_edit.text().strip()
    
    if not model_name:
        main_window._show_error("Please enter a HuggingFace model name.")
        return
    
    if not output_dir:
        main_window._show_error("Please select an output directory.")
        return
    
    # Placeholder - implement actual download logic
    main_window._append_hf_log(f"Starting HuggingFace model download: {model_name}")
    main_window._append_hf_log(f"Output directory: {output_dir}")
    main_window._append_hf_log("HuggingFace model download not yet implemented")


def stop_hf_model_download(main_window):
    """Stop HuggingFace model download"""
    # Placeholder - implement actual stop logic
    main_window._append_hf_log("Stopping HuggingFace model download...")


def browse_hf_model_output(main_window):
    """Browse for HuggingFace model output directory"""
    from PyQt5.QtWidgets import QFileDialog
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "outputs", "hf_models")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    current_path = main_window.hf_model_output_edit.text().strip()
    if current_path and os.path.exists(current_path):
        start_dir = current_path
    else:
        start_dir = default_path
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select output directory for HuggingFace model",
        start_dir,
    )
    if directory:
        main_window.hf_model_output_edit.setText(directory)


def add_parent_directory(main_window):
    """Add a parent directory to the upload list"""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "Outputs")
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select folder containing captioned datasets (images/ + metadata.jsonl)",
        default_path,
    )
    if directory:
        # Check if already in list
        existing = [main_window.hf_parent_dirs_list.item(i).text() 
                   for i in range(main_window.hf_parent_dirs_list.count())]
        if directory not in existing:
            main_window.hf_parent_dirs_list.addItem(directory)


def remove_parent_directory(main_window):
    """Remove selected directory from the upload list"""
    current_row = main_window.hf_parent_dirs_list.currentRow()
    if current_row >= 0:
        main_window.hf_parent_dirs_list.takeItem(current_row)


def browse_hf_upload_source(main_window):
    """Browse for caption dataset source folder to upload (legacy, kept for compatibility)"""
    add_parent_directory(main_window)


def get_parent_dirs_from_ui(main_window) -> List[str]:
    """Get list of parent directories from the UI"""
    return [main_window.hf_parent_dirs_list.item(i).text() 
            for i in range(main_window.hf_parent_dirs_list.count())]


def start_parquet_pack(main_window):
    """Start parquet packing without upload"""
    parent_dirs = get_parent_dirs_from_ui(main_window)
    
    if not parent_dirs:
        main_window._show_error("Please add at least one source directory.")
        return
    
    # Get output directory
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_out = os.path.join(repo_root, "Outputs", "parquet_packed")
    
    out_dir = QFileDialog.getExistingDirectory(
        main_window,
        "Select output directory for parquet files",
        default_out,
    )
    if not out_dir:
        return
    
    shard_rows = main_window.hf_shard_rows_spin.value()
    compression = main_window.hf_compression_combo.currentText()
    
    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window._append_hf_log("")
    main_window._append_hf_log("=== Parquet Packing started ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Packing...")
    main_window.hf_pack_button.setEnabled(False)
    main_window.hf_upload_button.setEnabled(False)
    main_window.hf_stop_upload_button.setEnabled(True)
    main_window.hf_upload_stop_flag = threading.Event()
    
    if not hasattr(main_window, 'hf_queue') or main_window.hf_queue is None:
        main_window.hf_queue = queue.Queue()
    
    main_window.hf_upload_thread = threading.Thread(
        target=parquet_pack_worker,
        args=(
            parent_dirs,
            out_dir,
            shard_rows,
            compression,
            main_window.hf_upload_stop_flag,
            main_window.hf_queue,
        ),
        daemon=True,
    )
    main_window.hf_upload_thread.start()
    main_window.timer.start()


def start_hf_upload(main_window):
    """Start the HuggingFace upload process (pack + upload)"""
    parent_dirs = get_parent_dirs_from_ui(main_window)
    repo_id = main_window.hf_upload_repo_edit.text().strip()
    hf_token = main_window.hf_token_edit.text().strip() if hasattr(main_window, 'hf_token_edit') else None
    is_private = main_window.hf_private_repo_check.isChecked()
    shard_rows = main_window.hf_shard_rows_spin.value()
    compression = main_window.hf_compression_combo.currentText()

    # Validate inputs
    if not parent_dirs:
        main_window._show_error("Please add at least one source directory.")
        return
    
    if not repo_id:
        main_window._show_error("Please enter a HuggingFace repo ID (e.g., 'username/dataset_name').")
        return
    
    if repo_id.count("/") != 1:
        main_window._show_error("Repo ID must be in format 'username/dataset_name'.")
        return

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window._append_hf_log("")
    main_window._append_hf_log("=== HuggingFace Pack & Upload started ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Packing & Uploading...")
    main_window.hf_pack_button.setEnabled(False)
    main_window.hf_upload_button.setEnabled(False)
    main_window.hf_stop_upload_button.setEnabled(True)
    main_window.hf_upload_stop_flag = threading.Event()

    # Ensure queue exists
    if not hasattr(main_window, 'hf_queue') or main_window.hf_queue is None:
        main_window.hf_queue = queue.Queue()

    # Start worker thread
    main_window.hf_upload_thread = threading.Thread(
        target=hf_parquet_upload_worker,
        args=(
            parent_dirs,
            repo_id,
            hf_token,
            is_private,
            shard_rows,
            compression,
            main_window.hf_upload_stop_flag,
            main_window.hf_queue,
        ),
        daemon=True,
    )
    main_window.hf_upload_thread.start()
    main_window.timer.start()


def stop_hf_upload(main_window):
    """Stop the HuggingFace upload process"""
    if hasattr(main_window, 'hf_upload_stop_flag'):
        main_window.hf_upload_stop_flag.set()
    main_window._append_hf_log("Stopping...")
    main_window.hf_stop_upload_button.setEnabled(False)
    main_window.hf_pack_button.setEnabled(True)
    main_window.hf_upload_button.setEnabled(True)


# ==============================
# PARQUET PACKING HELPERS
# ==============================

IGNORE_DIRS = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "node_modules"}
DEFAULT_PROMPT = "Describe the image accurately and in detail."
FRAGMENT_KEYS_HINT = ('"model":', '"api_type":', '"max_tokens":', '"temperature":')
MAX_WARNINGS_PER_FILE = 25


def is_dataset_root(d: str) -> bool:
    """Check if directory is a valid dataset root (has images/ + metadata.jsonl)"""
    return (
        os.path.isdir(d)
        and os.path.isfile(os.path.join(d, "metadata.jsonl"))
        and os.path.isdir(os.path.join(d, "images"))
    )


def find_dataset_roots(parents: List[str], q=None) -> List[str]:
    """Find all dataset roots under the given parent directories"""
    roots = []
    for parent in parents:
        if not os.path.isdir(parent):
            if q:
                q.put(("log", f"⚠️  Parent dir does not exist: {parent}"))
            continue

        if is_dataset_root(parent):
            roots.append(parent)
            continue

        for dirpath, dirnames, filenames in os.walk(parent):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            if "metadata.jsonl" in filenames and "images" in dirnames:
                roots.append(dirpath)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in roots:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def dataset_id_from_root(root: str) -> str:
    """Get dataset identifier from root path"""
    return os.path.basename(os.path.normpath(root))


def looks_like_stray_fragment(line: str) -> bool:
    """Check if line looks like a stray JSON fragment"""
    s = line.strip()
    if not s:
        return True
    if s.startswith("{") or s.startswith("["):
        return False
    hit = sum(1 for k in FRAGMENT_KEYS_HINT if k in s)
    if hit >= 2:
        return True
    return False


def parse_json_line_salvage(line: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse JSON line with salvage attempt for trailing junk"""
    try:
        return json.loads(line), "ok"
    except json.JSONDecodeError:
        pass

    try:
        dec = json.JSONDecoder()
        obj, _end = dec.raw_decode(line)
        return obj, "salvaged"
    except Exception:
        return None, "fail"


def load_meta(meta_path: str, q=None) -> List[Dict[str, Any]]:
    """Load metadata from JSONL file with error handling"""
    meta: List[Dict[str, Any]] = []
    warnings = 0

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            if looks_like_stray_fragment(line):
                warnings += 1
                continue

            if not (line.startswith("{") or line.startswith("[")):
                warnings += 1
                continue

            obj, mode = parse_json_line_salvage(line)

            if obj is None:
                warnings += 1
                continue

            if mode == "salvaged":
                warnings += 1

            if not isinstance(obj, dict):
                warnings += 1
                continue

            if "file_name" not in obj or "text" not in obj:
                warnings += 1
                continue

            meta.append(obj)

    if warnings > 0 and q:
        q.put(("log", f"   ⚠️  {warnings} warnings in {os.path.basename(meta_path)}"))
    return meta


def parquet_pack_worker(parent_dirs, out_dir, shard_rows, compression, stop_flag, q):
    """Worker function to pack datasets into parquet shards"""
    try:
        from datasets import Dataset, Features, Value, Image as HFImage
    except ImportError:
        if q:
            q.put(("error", "datasets library is required. Install with: pip install datasets"))
        return

    try:
        if q:
            q.put(("log", f"📁 Output directory: {out_dir}"))
            q.put(("log", f"⚙️  Shard rows: {shard_rows}, Compression: {compression}"))

        os.makedirs(out_dir, exist_ok=True)

        # Find all dataset roots
        if q:
            q.put(("log", "🔍 Scanning for datasets..."))
        dataset_roots = find_dataset_roots(parent_dirs, q)
        
        if not dataset_roots:
            if q:
                q.put(("error", f"No dataset roots found under: {parent_dirs}"))
            return

        if q:
            q.put(("log", f"📂 Found {len(dataset_roots)} dataset(s):"))
            for r in dataset_roots:
                q.put(("log", f"   - {r}"))

        # Build features: image, text, prompt (always included)
        features = Features({
            "image": HFImage(),
            "text": Value("string"),
            "prompt": Value("string"),
        })

        # Load all metadata
        if q:
            q.put(("log", "📝 Loading metadata..."))
        
        combined_rows: List[Tuple[str, str, str, str, str]] = []
        # tuple = (dataset_root, dataset_id, file_name, text, prompt_text)

        for root in dataset_roots:
            if stop_flag and stop_flag.is_set():
                if q:
                    q.put(("stopped", "Stopped by user"))
                return
                
            ds_id = dataset_id_from_root(root)
            meta_path = os.path.join(root, "metadata.jsonl")
            
            if q:
                q.put(("log", f"   Loading {ds_id}..."))
            
            meta = load_meta(meta_path, q)

            for r in meta:
                rel = r["file_name"]
                txt = r["text"]
                pm = r.get("prompt_metadata", {}) or {}
                prompt_text = pm.get("prompt_text", DEFAULT_PROMPT)

                if not isinstance(rel, str) or not isinstance(txt, str):
                    continue
                if not isinstance(prompt_text, str):
                    prompt_text = DEFAULT_PROMPT

                combined_rows.append((root, ds_id, rel, txt, prompt_text))

        if q:
            q.put(("log", f"📊 Total rows: {len(combined_rows)}"))

        num_shards = math.ceil(len(combined_rows) / shard_rows)
        if q:
            q.put(("log", f"📦 Creating {num_shards} shard(s)..."))

        # Build shards
        written = 0
        skipped_missing = 0

        for shard_idx in range(num_shards):
            if stop_flag and stop_flag.is_set():
                if q:
                    q.put(("stopped", "Stopped by user"))
                return

            start = shard_idx * shard_rows
            end = min((shard_idx + 1) * shard_rows, len(combined_rows))

            image_items = []
            text_list = []
            prompt_list = []

            for j in range(start, end):
                if stop_flag and stop_flag.is_set():
                    break
                    
                root, ds_id, rel, txt, prompt_text = combined_rows[j]
                images_dir = os.path.join(root, "images")
                rel_norm = rel.replace("\\", "/")
                img_path = os.path.join(images_dir, rel)

                if not os.path.exists(img_path):
                    skipped_missing += 1
                    continue

                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                if isinstance(img_bytes, (bytearray, memoryview)):
                    img_bytes = bytes(img_bytes)

                rel_out = f"{ds_id}/{rel_norm}"

                image_items.append({"bytes": img_bytes, "path": rel_out})
                text_list.append(txt)
                prompt_list.append(prompt_text)

            out_path = os.path.join(out_dir, f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet")

            data = {"image": image_items, "text": text_list, "prompt": prompt_list}

            ds = Dataset.from_dict(data)
            ds = ds.cast(features)
            
            compression_arg = None if compression == "none" else compression
            ds.to_parquet(out_path, compression=compression_arg)

            wrote_rows = len(text_list)
            written += wrote_rows
            if q:
                q.put(("log", f"✅ Shard {shard_idx+1}/{num_shards}: {wrote_rows} rows"))

        if skipped_missing > 0 and q:
            q.put(("log", f"⚠️  Skipped {skipped_missing} missing images"))

        success_msg = f"✅ Packing complete! {written} rows in {num_shards} shards → {out_dir}"
        if q:
            q.put(("log", success_msg))
            q.put(("success", success_msg))

    except Exception as e:
        error_msg = f"Packing failed: {str(e)}"
        if q:
            q.put(("error", error_msg))
        import traceback
        print(f"PARQUET_PACK_WORKER_ERROR: {error_msg}")
        print(traceback.format_exc())


def hf_parquet_upload_worker(parent_dirs, repo_id, token, is_private, shard_rows, compression, stop_flag, q):
    """Worker function to pack datasets into parquet and upload to HuggingFace"""
    try:
        from datasets import Dataset, Features, Value, Image as HFImage
        from huggingface_hub import create_repo, upload_folder, HfApi
    except ImportError as e:
        if q:
            q.put(("error", f"Required library missing: {e}. Install with: pip install datasets huggingface_hub"))
        return

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "data")
            os.makedirs(out_dir, exist_ok=True)

            if q:
                q.put(("log", f"📦 Repo: {repo_id} (private={is_private})"))
                q.put(("log", f"⚙️  Shard rows: {shard_rows}, Compression: {compression}"))

            # Find all dataset roots
            if q:
                q.put(("log", "🔍 Scanning for datasets..."))
            dataset_roots = find_dataset_roots(parent_dirs, q)
            
            if not dataset_roots:
                if q:
                    q.put(("error", f"No dataset roots found under: {parent_dirs}"))
                return

            if q:
                q.put(("log", f"📂 Found {len(dataset_roots)} dataset(s):"))
                for r in dataset_roots:
                    q.put(("log", f"   - {r}"))

            # Build features: image, text, prompt (always included)
            features = Features({
                "image": HFImage(),
                "text": Value("string"),
                "prompt": Value("string"),
            })

            # Load all metadata
            if q:
                q.put(("log", "📝 Loading metadata..."))
            
            combined_rows: List[Tuple[str, str, str, str, str]] = []

            for root in dataset_roots:
                if stop_flag and stop_flag.is_set():
                    if q:
                        q.put(("stopped", "Stopped by user"))
                    return
                    
                ds_id = dataset_id_from_root(root)
                meta_path = os.path.join(root, "metadata.jsonl")
                
                if q:
                    q.put(("log", f"   Loading {ds_id}..."))
                
                meta = load_meta(meta_path, q)

                for r in meta:
                    rel = r["file_name"]
                    txt = r["text"]
                    pm = r.get("prompt_metadata", {}) or {}
                    prompt_text = pm.get("prompt_text", DEFAULT_PROMPT)

                    if not isinstance(rel, str) or not isinstance(txt, str):
                        continue
                    if not isinstance(prompt_text, str):
                        prompt_text = DEFAULT_PROMPT

                    combined_rows.append((root, ds_id, rel, txt, prompt_text))

            if q:
                q.put(("log", f"📊 Total rows: {len(combined_rows)}"))

            num_shards = math.ceil(len(combined_rows) / shard_rows)
            if q:
                q.put(("log", f"📦 Creating {num_shards} shard(s)..."))

            # Build shards
            written = 0
            skipped_missing = 0

            for shard_idx in range(num_shards):
                if stop_flag and stop_flag.is_set():
                    if q:
                        q.put(("stopped", "Stopped by user"))
                    return

                start = shard_idx * shard_rows
                end = min((shard_idx + 1) * shard_rows, len(combined_rows))

                image_items = []
                text_list = []
                prompt_list = []

                for j in range(start, end):
                    if stop_flag and stop_flag.is_set():
                        break
                        
                    root, ds_id, rel, txt, prompt_text = combined_rows[j]
                    images_dir = os.path.join(root, "images")
                    rel_norm = rel.replace("\\", "/")
                    img_path = os.path.join(images_dir, rel)

                    if not os.path.exists(img_path):
                        skipped_missing += 1
                        continue

                    with open(img_path, "rb") as f:
                        img_bytes = f.read()

                    if isinstance(img_bytes, (bytearray, memoryview)):
                        img_bytes = bytes(img_bytes)

                    rel_out = f"{ds_id}/{rel_norm}"

                    image_items.append({"bytes": img_bytes, "path": rel_out})
                    text_list.append(txt)
                    prompt_list.append(prompt_text)

                out_path = os.path.join(out_dir, f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet")

                data = {"image": image_items, "text": text_list, "prompt": prompt_list}

                ds = Dataset.from_dict(data)
                ds = ds.cast(features)
                
                compression_arg = None if compression == "none" else compression
                ds.to_parquet(out_path, compression=compression_arg)

                wrote_rows = len(text_list)
                written += wrote_rows
                if q:
                    q.put(("log", f"✅ Shard {shard_idx+1}/{num_shards}: {wrote_rows} rows"))

            if skipped_missing > 0 and q:
                q.put(("log", f"⚠️  Skipped {skipped_missing} missing images"))

            if q:
                q.put(("log", f"📊 Packed {written} rows total"))

            # Create repo
            if q:
                q.put(("log", "🔧 Creating repository..."))
            create_repo(repo_id, repo_type="dataset", private=is_private, exist_ok=True, token=token)

            # Upload
            if q:
                q.put(("log", "📤 Uploading to HuggingFace..."))

            upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=tmp_dir,
                path_in_repo=".",
                token=token,
                commit_message=f"Upload parquet dataset ({written} rows, {num_shards} shards)",
            )

            success_msg = f"✅ Upload complete! https://huggingface.co/datasets/{repo_id}"
            if q:
                q.put(("log", success_msg))
                q.put(("success", success_msg))

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        if q:
            q.put(("error", error_msg))
        import traceback
        print(f"HF_PARQUET_UPLOAD_WORKER_ERROR: {error_msg}")
        print(traceback.format_exc())


# Legacy upload worker (kept for compatibility)
def hf_upload_worker(work_dir, repo_id, token, is_private, shard_size, batch_size, resume, stop_flag, q):
    """Legacy worker function to upload caption dataset to HuggingFace (non-parquet)"""
    try:
        from huggingface_hub import create_repo, upload_file, upload_folder, HfApi
    except ImportError:
        if q:
            q.put(("error", "huggingface_hub is required. Install with: pip install huggingface_hub"))
        return

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

    def shard_subdir_from_index(idx, per_dir):
        """Get shard subdirectory name from image index"""
        return f"{idx // per_dir:05d}"

    try:
        images_dir = Path(work_dir) / "images"
        meta_path = Path(work_dir) / "metadata.jsonl"

        if q:
            q.put(("log", f"📁 Source directory: {work_dir}"))
            q.put(("log", f"📦 Repo: {repo_id} (private={is_private})"))
            q.put(("log", f"⚙️  Shard size: {shard_size}, Batch size: {batch_size}"))

        # Create repo
        if q:
            q.put(("log", "Creating repository..."))
        create_repo(repo_id, repo_type="dataset", private=is_private, exist_ok=True, token=token)

        # Collect local images
        local_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if not local_files:
            if q:
                q.put(("error", "No images found in images directory."))
            return

        if q:
            q.put(("log", f"📷 Found {len(local_files)} local images"))

        # Resume: check already-uploaded files
        already = set()
        if resume:
            if q:
                q.put(("log", "Checking existing files on Hub..."))
            api = HfApi(token=token)
            try:
                repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                for p in repo_files:
                    if p.startswith("images/"):
                        already.add(Path(p).name)
                if q:
                    q.put(("log", f"📂 Found {len(already)} already-uploaded images"))
            except Exception as e:
                if q:
                    q.put(("log", f"⚠️  Could not check existing files: {e}"))

        to_upload = [p for p in local_files if p.name not in already] if already else local_files
        total = len(to_upload)

        if total == 0:
            if q:
                q.put(("log", "✅ Nothing to upload (all images already present)."))
                q.put(("success", "Upload complete - all images already present."))
            return

        # Upload sharded metadata.jsonl
        if q:
            q.put(("log", "📝 Uploading sharded metadata.jsonl..."))

        with tempfile.TemporaryDirectory() as tmp_meta_dir:
            tmp_meta = Path(tmp_meta_dir) / "metadata.jsonl"
            with meta_path.open("r", encoding="utf-8") as fin, tmp_meta.open("w", encoding="utf-8") as fout:
                for line in fin:
                    try:
                        row = json.loads(line)
                        fn = row.get("file_name", "")
                        if not fn:
                            continue
                        # Parse index from filename (e.g., "00000123.png" -> 123)
                        try:
                            idx = int(Path(fn).stem)
                        except ValueError:
                            idx = 0
                        sub = shard_subdir_from_index(idx, shard_size)
                        row["file_name"] = f"{sub}/{fn}"
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError:
                        continue

            upload_file(
                path_or_fileobj=str(tmp_meta),
                path_in_repo="metadata.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"Upload sharded metadata (shard_size={shard_size})",
            )

        # Upload images in batches
        num_batches = math.ceil(total / batch_size)
        if q:
            q.put(("log", f"📤 Uploading {total} images in {num_batches} batches..."))

        for b in range(num_batches):
            if stop_flag and stop_flag.is_set():
                if q:
                    q.put(("log", "Upload stopped by user"))
                    q.put(("stopped", "Stopped by user"))
                return

            start = b * batch_size
            end = min((b + 1) * batch_size, total)
            batch = to_upload[start:end]

            if q:
                q.put(("log", f"📦 Batch {b+1}/{num_batches} ({start+1}-{end} of {total})..."))

            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = Path(tmp)
                tmp_images_root = tmp_root / "images"
                tmp_images_root.mkdir(parents=True, exist_ok=True)

                # Stage files with sharding
                for p in batch:
                    try:
                        idx = int(p.stem)
                    except ValueError:
                        idx = 0
                    sub = shard_subdir_from_index(idx, shard_size)
                    target_dir = tmp_images_root / sub
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target_dir / p.name)

                commit_msg = f"Upload images batch {start:07d}-{end-1:07d}"

                upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(tmp_root),
                    path_in_repo=".",
                    token=token,
                    commit_message=commit_msg,
                )

            if q:
                q.put(("log", f"✅ Batch {b+1}/{num_batches} uploaded"))

        success_msg = f"✅ Upload complete! https://huggingface.co/datasets/{repo_id}"
        if q:
            q.put(("log", success_msg))
            q.put(("success", success_msg))

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        if q:
            q.put(("error", error_msg))
        import traceback
        print(f"HF_UPLOAD_WORKER_ERROR: {error_msg}")
        print(traceback.format_exc())

