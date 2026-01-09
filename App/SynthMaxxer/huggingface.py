"""
HuggingFace module - HuggingFace dataset/model downloader and uploader tab functionality
Handles HuggingFace dataset/model downloading and caption dataset uploading UI and logic
"""
import os
import json
import math
import shutil
import tempfile
import threading
import queue
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QScrollArea,
    QFrame, QSpinBox, QCheckBox
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
    
    # HuggingFace Caption Dataset Upload
    hf_upload_group = QGroupBox("🚀 Caption Dataset Upload")
    hf_upload_layout = QFormLayout()
    hf_upload_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_upload_layout.setHorizontalSpacing(10)
    hf_upload_layout.setVerticalSpacing(6)
    hf_upload_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    hf_upload_group.setLayout(hf_upload_layout)

    # Source folder (captioned output folder)
    source_row, _ = create_file_browse_row(
        line_edit_name="hf_upload_source_edit",
        placeholder_text="Folder containing images/ + metadata.jsonl",
        on_browse_clicked=lambda: browse_hf_upload_source(main_window)
    )
    main_window.hf_upload_source_edit = source_row.itemAt(0).widget()
    hf_upload_layout.addRow(QLabel("Source Folder:"), _wrap_row(source_row))

    main_window.hf_upload_repo_edit = QLineEdit()
    main_window.hf_upload_repo_edit.setPlaceholderText("e.g., username/dataset_name")
    main_window.hf_upload_repo_edit.setToolTip("HuggingFace dataset repo ID (must be like 'username/dataset_name')")
    hf_upload_layout.addRow(QLabel("HF Repo ID:"), main_window.hf_upload_repo_edit)

    main_window.hf_private_repo_check = QCheckBox("Private repository")
    main_window.hf_private_repo_check.setChecked(True)
    main_window.hf_private_repo_check.setToolTip("Create repository as private")
    hf_upload_layout.addRow("", main_window.hf_private_repo_check)

    main_window.hf_shard_size_spin = QSpinBox()
    main_window.hf_shard_size_spin.setRange(100, 9000)
    main_window.hf_shard_size_spin.setValue(1000)
    main_window.hf_shard_size_spin.setMaximumWidth(100)
    main_window.hf_shard_size_spin.setToolTip("Images per subfolder (HF limit: <10000 files per folder)")
    shard_row = QHBoxLayout()
    shard_row.addWidget(main_window.hf_shard_size_spin)
    shard_row.addStretch()
    hf_upload_layout.addRow(QLabel("Shard Size:"), _wrap_row(shard_row))

    main_window.hf_upload_batch_spin = QSpinBox()
    main_window.hf_upload_batch_spin.setRange(50, 20000)
    main_window.hf_upload_batch_spin.setValue(2000)
    main_window.hf_upload_batch_spin.setMaximumWidth(100)
    main_window.hf_upload_batch_spin.setToolTip("Images per commit (larger = fewer commits, but larger individual uploads)")
    batch_row = QHBoxLayout()
    batch_row.addWidget(main_window.hf_upload_batch_spin)
    batch_row.addStretch()
    hf_upload_layout.addRow(QLabel("Batch Size:"), _wrap_row(batch_row))

    main_window.hf_resume_upload_check = QCheckBox("Resume (skip already-uploaded images)")
    main_window.hf_resume_upload_check.setChecked(True)
    main_window.hf_resume_upload_check.setToolTip("Check remote repo and skip images that already exist")
    hf_upload_layout.addRow("", main_window.hf_resume_upload_check)

    # Upload buttons
    upload_btn_row = QHBoxLayout()
    main_window.hf_upload_button = QPushButton("Upload to HuggingFace")
    main_window.hf_upload_button.clicked.connect(lambda: start_hf_upload(main_window))
    main_window.hf_upload_button.setToolTip("Upload the caption dataset to HuggingFace")
    main_window.hf_stop_upload_button = QPushButton("Stop Upload")
    main_window.hf_stop_upload_button.clicked.connect(lambda: stop_hf_upload(main_window))
    main_window.hf_stop_upload_button.setEnabled(False)
    upload_btn_row.addWidget(main_window.hf_upload_button)
    upload_btn_row.addWidget(main_window.hf_stop_upload_button)
    upload_btn_row.addStretch()
    hf_upload_layout.addRow("", _wrap_row(upload_btn_row))

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


def browse_hf_upload_source(main_window):
    """Browse for caption dataset source folder to upload"""
    from PyQt5.QtWidgets import QFileDialog
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "Outputs")
    
    current_path = main_window.hf_upload_source_edit.text().strip()
    if current_path and os.path.exists(current_path):
        start_dir = current_path
    else:
        start_dir = default_path
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select caption dataset folder (contains images/ + metadata.jsonl)",
        start_dir,
    )
    if directory:
        main_window.hf_upload_source_edit.setText(directory)


def start_hf_upload(main_window):
    """Start the HuggingFace upload process"""
    output_folder = main_window.hf_upload_source_edit.text().strip()
    repo_id = main_window.hf_upload_repo_edit.text().strip()
    hf_token = main_window.hf_token_edit.text().strip() if hasattr(main_window, 'hf_token_edit') else None
    is_private = main_window.hf_private_repo_check.isChecked()
    shard_size = main_window.hf_shard_size_spin.value()
    batch_size = main_window.hf_upload_batch_spin.value()
    resume = main_window.hf_resume_upload_check.isChecked()

    # Validate inputs
    if not output_folder:
        main_window._show_error("Please specify a source folder containing captioned images.")
        return
    
    if not repo_id:
        main_window._show_error("Please enter a HuggingFace repo ID (e.g., 'username/dataset_name').")
        return
    
    if repo_id.count("/") != 1:
        main_window._show_error("Repo ID must be in format 'username/dataset_name'.")
        return

    # Source folder should contain images/ + metadata.jsonl
    images_dir = os.path.join(output_folder, "images")
    metadata_path = os.path.join(output_folder, "metadata.jsonl")

    if not os.path.exists(output_folder):
        main_window._show_error(f"Source folder not found: {output_folder}\nRun captioning first to generate images and metadata.")
        return

    if not os.path.exists(images_dir):
        main_window._show_error(f"Images directory not found: {images_dir}\nRun captioning first.")
        return

    if not os.path.exists(metadata_path):
        main_window._show_error(f"metadata.jsonl not found: {metadata_path}\nRun captioning first.")
        return

    # Check for images
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    local_files = [p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not local_files:
        main_window._show_error("No images found in images directory.")
        return

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window._append_hf_log("")
    main_window._append_hf_log("=== HuggingFace Upload started ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Uploading...")
    main_window.hf_upload_button.setEnabled(False)
    main_window.hf_stop_upload_button.setEnabled(True)
    main_window.hf_upload_stop_flag = threading.Event()

    # Ensure queue exists
    if not hasattr(main_window, 'hf_queue') or main_window.hf_queue is None:
        main_window.hf_queue = queue.Queue()

    # Start worker thread
    main_window.hf_upload_thread = threading.Thread(
        target=hf_upload_worker,
        args=(
            output_folder,
            repo_id,
            hf_token,
            is_private,
            shard_size,
            batch_size,
            resume,
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
    main_window._append_hf_log("Stopping upload...")
    main_window.hf_stop_upload_button.setEnabled(False)


def hf_upload_worker(work_dir, repo_id, token, is_private, shard_size, batch_size, resume, stop_flag, q):
    """Worker function to upload caption dataset to HuggingFace"""
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

