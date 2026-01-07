"""
CaptionMaxxer module - Image captioning tab functionality
Handles multimodal image captioning UI and logic
"""
import os
import queue
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QGroupBox, QScrollArea,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt

from App.SynthMaxxer.multimodal_worker import image_captioning_worker


def build_captionmaxxer_tab(main_window):
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    """Build the CaptionMaxxer (Image Captioning) tab UI"""
    multimodal_tab = QWidget()
    multimodal_layout = QVBoxLayout()
    multimodal_layout.setContentsMargins(0, 0, 0, 0)
    multimodal_tab.setLayout(multimodal_layout)
    
    mm_header = QHBoxLayout()
    main_window.mm_start_button = QPushButton("Start Captioning")
    main_window.mm_start_button.clicked.connect(lambda: start_image_captioning(main_window))
    main_window.mm_stop_button = QPushButton("Stop")
    main_window.mm_stop_button.clicked.connect(lambda: stop_image_captioning(main_window))
    main_window.mm_stop_button.setEnabled(False)
    mm_header.addStretch()
    mm_header.addWidget(main_window.mm_start_button)
    mm_header.addWidget(main_window.mm_stop_button)
    multimodal_layout.addLayout(mm_header)
    
    mm_split = QHBoxLayout()
    mm_split.setSpacing(14)
    multimodal_layout.addLayout(mm_split, stretch=1)
    
    mm_left_scroll = QScrollArea()
    mm_left_scroll.setWidgetResizable(True)
    mm_left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    mm_left_scroll.setFrameShape(QFrame.NoFrame)
    
    mm_left_container = QWidget()
    mm_left_panel = QVBoxLayout()
    mm_left_panel.setSpacing(10)
    mm_left_container.setLayout(mm_left_panel)
    mm_left_scroll.setWidget(mm_left_container)
    
    mm_split.addWidget(mm_left_scroll, stretch=3)
    
    mm_right_container = QWidget()
    mm_right_panel = QVBoxLayout()
    mm_right_panel.setSpacing(10)
    mm_right_container.setLayout(mm_right_panel)
    
    mm_split.addWidget(mm_right_container, stretch=4)
    
    # Build multimodal UI
    _build_multimodal_ui(main_window, mm_left_panel, mm_right_panel)
    
    return multimodal_tab


def _build_multimodal_ui(main_window, left_panel, right_panel):
    """Build the multimodal image captioning UI components"""
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    # Files group
    mm_files_group = QGroupBox("📁 Image Input")
    mm_files_layout = QFormLayout()
    mm_files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    mm_files_layout.setHorizontalSpacing(10)
    mm_files_layout.setVerticalSpacing(6)
    mm_files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    mm_files_group.setLayout(mm_files_layout)

    image_dir_row, _ = create_file_browse_row(
        line_edit_name="mm_image_dir_edit",
        placeholder_text="Path to folder containing images",
        on_browse_clicked=lambda: browse_mm_image_dir(main_window)
    )
    main_window.mm_image_dir_edit = image_dir_row.itemAt(0).widget()
    mm_files_layout.addRow(QLabel("Image Folder:"), _wrap_row(image_dir_row))

    # Get repo root (go up from App/SynthMaxxer to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outputs_dir = os.path.join(repo_root, "Outputs")
    default_output = os.path.join(outputs_dir, "captions")
    output_row, _ = create_file_browse_row(
        line_edit_name="mm_output_edit",
        placeholder_text="Output folder (will contain images/ + metadata.jsonl)",
        default_text=default_output,
        on_browse_clicked=lambda: browse_mm_output(main_window)
    )
    main_window.mm_output_edit = output_row.itemAt(0).widget()
    mm_files_layout.addRow(QLabel("Output Folder:"), _wrap_row(output_row))
    left_panel.addWidget(mm_files_group)

    # HuggingFace Dataset Input (for captioning from HF datasets)
    hf_dataset_group = QGroupBox("🤗 HuggingFace Dataset (Optional)")
    hf_dataset_layout = QFormLayout()
    hf_dataset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_dataset_layout.setHorizontalSpacing(10)
    hf_dataset_layout.setVerticalSpacing(6)
    hf_dataset_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    hf_dataset_group.setLayout(hf_dataset_layout)

    main_window.mm_hf_dataset_edit = QLineEdit()
    main_window.mm_hf_dataset_edit.setPlaceholderText("e.g., dataset_name or org/dataset_name (leave empty to use image folder)")
    hf_dataset_layout.addRow(QLabel("HF Dataset:"), main_window.mm_hf_dataset_edit)

    main_window.mm_use_hf_dataset_check = QCheckBox("Use HuggingFace dataset instead of image folder")
    main_window.mm_use_hf_dataset_check.setChecked(False)
    main_window.mm_use_hf_dataset_check.toggled.connect(lambda checked: _toggle_hf_dataset_mode(main_window, checked))
    # Initialize state
    main_window.mm_hf_dataset_edit.setEnabled(False)
    hf_dataset_layout.addRow("", main_window.mm_use_hf_dataset_check)
    left_panel.addWidget(hf_dataset_group)

    # Caption Settings
    mm_caption_group = QGroupBox("📝 Caption Settings")
    mm_caption_layout = QFormLayout()
    mm_caption_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    mm_caption_layout.setHorizontalSpacing(10)
    mm_caption_layout.setVerticalSpacing(6)
    mm_caption_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    mm_caption_group.setLayout(mm_caption_layout)

    main_window.mm_caption_prompt_edit = QPlainTextEdit()
    main_window.mm_caption_prompt_edit.setPlaceholderText("Describe what you see in this image in accurate detail. Include all important elements, objects, people, text, colors, composition, and context. Avoid repetitious phrasing in your analysis when possible. Do not shy away from using vulgar or sexual prose if, and when appropriate.")
    main_window.mm_caption_prompt_edit.setFixedHeight(240)
    mm_caption_layout.addRow(QLabel("Caption Prompt:"), main_window.mm_caption_prompt_edit)

    main_window.mm_max_tokens_spin = QSpinBox()
    main_window.mm_max_tokens_spin.setRange(50, 4000)
    main_window.mm_max_tokens_spin.setValue(500)
    main_window.mm_max_tokens_spin.setMaximumWidth(100)
    main_window.mm_max_tokens_spin.setToolTip("Maximum tokens for caption generation")
    max_tokens_row = QHBoxLayout()
    max_tokens_row.addWidget(main_window.mm_max_tokens_spin)
    max_tokens_row.addStretch()
    mm_caption_layout.addRow(QLabel("Max Tokens:"), _wrap_row(max_tokens_row))

    main_window.mm_temperature_spin = QDoubleSpinBox()
    main_window.mm_temperature_spin.setRange(0.0, 2.0)
    main_window.mm_temperature_spin.setValue(0.7)
    main_window.mm_temperature_spin.setSingleStep(0.1)
    main_window.mm_temperature_spin.setDecimals(1)
    main_window.mm_temperature_spin.setMaximumWidth(100)
    main_window.mm_temperature_spin.setToolTip("Temperature for caption generation")
    temp_row = QHBoxLayout()
    temp_row.addWidget(main_window.mm_temperature_spin)
    temp_row.addStretch()
    mm_caption_layout.addRow(QLabel("Temperature:"), _wrap_row(temp_row))

    main_window.mm_batch_size_spin = QSpinBox()
    main_window.mm_batch_size_spin.setRange(1, 20)
    main_window.mm_batch_size_spin.setValue(1)
    main_window.mm_batch_size_spin.setMaximumWidth(100)
    main_window.mm_batch_size_spin.setToolTip("Number of images to process in parallel (1 recommended for most APIs)")
    batch_row = QHBoxLayout()
    batch_row.addWidget(main_window.mm_batch_size_spin)
    batch_row.addStretch()
    mm_caption_layout.addRow(QLabel("Batch Size:"), _wrap_row(batch_row))

    main_window.mm_max_captions_spin = QSpinBox()
    main_window.mm_max_captions_spin.setRange(0, 100000)
    main_window.mm_max_captions_spin.setValue(0)
    main_window.mm_max_captions_spin.setSpecialValueText("Unlimited")
    main_window.mm_max_captions_spin.setMaximumWidth(100)
    main_window.mm_max_captions_spin.setToolTip("Maximum number of captions to generate (0 = unlimited, processes all images)")
    max_captions_row = QHBoxLayout()
    max_captions_row.addWidget(main_window.mm_max_captions_spin)
    max_captions_row.addStretch()
    mm_caption_layout.addRow(QLabel("Max Captions:"), _wrap_row(max_captions_row))
    left_panel.addWidget(mm_caption_group)

    # HuggingFace Upload Settings (Optional - collapsible)
    mm_upload_group = QGroupBox("🚀 HuggingFace Upload (Optional)")
    mm_upload_group.setCheckable(True)
    mm_upload_group.setChecked(False)  # Collapsed by default
    mm_upload_layout = QFormLayout()
    mm_upload_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    mm_upload_layout.setHorizontalSpacing(10)
    mm_upload_layout.setVerticalSpacing(6)
    mm_upload_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    mm_upload_group.setLayout(mm_upload_layout)

    main_window.mm_hf_repo_edit = QLineEdit()
    main_window.mm_hf_repo_edit.setPlaceholderText("e.g., username/dataset_name")
    main_window.mm_hf_repo_edit.setToolTip("HuggingFace dataset repo ID (must be like 'username/dataset_name')")
    mm_upload_layout.addRow(QLabel("HF Repo ID:"), main_window.mm_hf_repo_edit)

    main_window.mm_private_repo_check = QCheckBox("Private repository")
    main_window.mm_private_repo_check.setChecked(True)
    main_window.mm_private_repo_check.setToolTip("Create repository as private")
    mm_upload_layout.addRow("", main_window.mm_private_repo_check)

    main_window.mm_shard_size_spin = QSpinBox()
    main_window.mm_shard_size_spin.setRange(100, 9000)
    main_window.mm_shard_size_spin.setValue(1000)
    main_window.mm_shard_size_spin.setMaximumWidth(100)
    main_window.mm_shard_size_spin.setToolTip("Images per subfolder (HF limit: <10000 files per folder)")
    shard_row = QHBoxLayout()
    shard_row.addWidget(main_window.mm_shard_size_spin)
    shard_row.addStretch()
    mm_upload_layout.addRow(QLabel("Shard Size:"), _wrap_row(shard_row))

    main_window.mm_upload_batch_spin = QSpinBox()
    main_window.mm_upload_batch_spin.setRange(50, 20000)
    main_window.mm_upload_batch_spin.setValue(2000)
    main_window.mm_upload_batch_spin.setMaximumWidth(100)
    main_window.mm_upload_batch_spin.setToolTip("Images per commit (larger = fewer commits, but larger individual uploads)")
    batch_row = QHBoxLayout()
    batch_row.addWidget(main_window.mm_upload_batch_spin)
    batch_row.addStretch()
    mm_upload_layout.addRow(QLabel("Batch Size:"), _wrap_row(batch_row))

    main_window.mm_resume_upload_check = QCheckBox("Resume (skip already-uploaded images)")
    main_window.mm_resume_upload_check.setChecked(True)
    main_window.mm_resume_upload_check.setToolTip("Check remote repo and skip images that already exist")
    mm_upload_layout.addRow("", main_window.mm_resume_upload_check)

    # Upload button
    upload_btn_row = QHBoxLayout()
    main_window.mm_upload_button = QPushButton("Upload to HuggingFace")
    main_window.mm_upload_button.clicked.connect(lambda: start_hf_upload(main_window))
    main_window.mm_upload_button.setToolTip("Upload the output folder to HuggingFace")
    main_window.mm_stop_upload_button = QPushButton("Stop Upload")
    main_window.mm_stop_upload_button.clicked.connect(lambda: stop_hf_upload(main_window))
    main_window.mm_stop_upload_button.setEnabled(False)
    upload_btn_row.addWidget(main_window.mm_upload_button)
    upload_btn_row.addWidget(main_window.mm_stop_upload_button)
    upload_btn_row.addStretch()
    mm_upload_layout.addRow("", _wrap_row(upload_btn_row))

    left_panel.addWidget(mm_upload_group)
    
    # Store reference for config save/load
    main_window.mm_upload_group = mm_upload_group
    left_panel.addStretch(1)

    # Right panel - Logs
    mm_progress_group, main_window.mm_log_view = create_log_view()
    right_panel.addWidget(mm_progress_group, stretch=1)


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def browse_mm_image_dir(main_window):
    """Browse for image directory"""
    from PyQt5.QtWidgets import QFileDialog
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select image folder",
        "",
    )
    if directory:
        main_window.mm_image_dir_edit.setText(directory)


def browse_mm_output(main_window):
    """Browse for output folder"""
    from PyQt5.QtWidgets import QFileDialog
    # Get repo root (go up from App/SynthMaxxer to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "Outputs")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    current_path = main_window.mm_output_edit.text().strip()
    if current_path and os.path.isdir(current_path):
        start_dir = current_path
    elif current_path and os.path.isdir(os.path.dirname(current_path)):
        start_dir = os.path.dirname(current_path)
    else:
        start_dir = default_path
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select output folder for captions",
        start_dir,
    )
    if directory:
        main_window.mm_output_edit.setText(directory)


def _toggle_hf_dataset_mode(main_window, checked):
    """Enable/disable image folder input based on HF dataset mode"""
    if checked:
        main_window.mm_image_dir_edit.setEnabled(False)
        main_window.mm_hf_dataset_edit.setEnabled(True)
    else:
        main_window.mm_image_dir_edit.setEnabled(True)
        main_window.mm_hf_dataset_edit.setEnabled(False)


def start_image_captioning(main_window):
    """Start the image captioning process"""
    use_hf_dataset = main_window.mm_use_hf_dataset_check.isChecked()
    hf_dataset = main_window.mm_hf_dataset_edit.text().strip() if use_hf_dataset else None
    hf_token = main_window.hf_token_edit.text().strip() if use_hf_dataset else None
    image_dir = main_window.mm_image_dir_edit.text().strip() if not use_hf_dataset else None
    output_folder = main_window.mm_output_edit.text().strip()
    api_key = main_window.mm_api_key_edit.text().strip()
    endpoint = main_window.mm_endpoint_edit.text().strip()
    model = main_window.mm_model_combo.currentText().strip()
    api_type = main_window.mm_api_type_combo.currentText()
    caption_prompt = main_window.mm_caption_prompt_edit.toPlainText().strip()
    max_tokens = main_window.mm_max_tokens_spin.value()
    temperature = main_window.mm_temperature_spin.value()
    batch_size = main_window.mm_batch_size_spin.value()
    max_captions = main_window.mm_max_captions_spin.value()

    if use_hf_dataset:
        if not hf_dataset:
            main_window._show_error("Please enter a HuggingFace dataset name.")
            return
    else:
        if not image_dir:
            main_window._show_error("Please select an image folder.")
            return
        if not os.path.isdir(image_dir):
            main_window._show_error("Image folder path is invalid.")
            return
    
    if not api_key:
        main_window._show_error("Please enter your API key.")
        return
    if not endpoint:
        main_window._show_error("Please enter the API endpoint.")
        return
    if not model:
        main_window._show_error("Please enter the model name.")
        return

    # Validate model is selected
    if model == "(Click Refresh to load models)" or not model or model.startswith("("):
        main_window._show_error("Please select a model. Click 'Refresh' to load available models.")
        return

    # Auto-generate output folder if not provided
    if not output_folder:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(repo_root, "Outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(outputs_dir, f"captions_{timestamp}")
        main_window.mm_output_edit.setText(output_folder)

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.mm_log_view.clear()
    main_window._append_mm_log("=== Image Captioning started ===")
    main_window._append_mm_log(f"Output folder: {output_folder}")
    main_window.setWindowTitle(f"{APP_TITLE} - Captioning...")
    main_window.mm_start_button.setEnabled(False)
    main_window.mm_stop_button.setEnabled(True)
    main_window.mm_stop_flag = threading.Event()

    main_window.mm_queue = queue.Queue()

    # Start worker thread
    main_window.mm_worker_thread = threading.Thread(
        target=image_captioning_worker,
        args=(
            image_dir,
            output_folder,
            api_key,
            endpoint,
            model,
            api_type,
            caption_prompt,
            max_tokens,
            temperature,
            batch_size,
            max_captions,
            main_window.mm_stop_flag,
            hf_dataset,
            hf_token,
            main_window.mm_queue,
        ),
        daemon=True,
    )
    main_window.mm_worker_thread.start()
    main_window.timer.start()


def stop_image_captioning(main_window):
    """Stop the image captioning process"""
    if hasattr(main_window, 'mm_stop_flag'):
        main_window.mm_stop_flag.set()
    main_window._append_mm_log("Stopping image captioning...")
    main_window.mm_stop_button.setEnabled(False)


def start_hf_upload(main_window):
    """Start the HuggingFace upload process"""
    output_folder = main_window.mm_output_edit.text().strip()
    repo_id = main_window.mm_hf_repo_edit.text().strip()
    hf_token = main_window.hf_token_edit.text().strip() if hasattr(main_window, 'hf_token_edit') else None
    is_private = main_window.mm_private_repo_check.isChecked()
    shard_size = main_window.mm_shard_size_spin.value()
    batch_size = main_window.mm_upload_batch_spin.value()
    resume = main_window.mm_resume_upload_check.isChecked()

    # Validate inputs
    if not output_folder:
        main_window._show_error("Please specify an output folder first.")
        return
    
    if not repo_id:
        main_window._show_error("Please enter a HuggingFace repo ID (e.g., 'username/dataset_name').")
        return
    
    if repo_id.count("/") != 1:
        main_window._show_error("Repo ID must be in format 'username/dataset_name'.")
        return

    # Output folder IS the dataset folder (contains images/ + metadata.jsonl)
    images_dir = os.path.join(output_folder, "images")
    metadata_path = os.path.join(output_folder, "metadata.jsonl")

    if not os.path.exists(output_folder):
        main_window._show_error(f"Output folder not found: {output_folder}\nRun captioning first to generate images and metadata.")
        return

    if not os.path.exists(images_dir):
        main_window._show_error(f"Images directory not found: {images_dir}\nRun captioning first.")
        return

    if not os.path.exists(metadata_path):
        main_window._show_error(f"metadata.jsonl not found: {metadata_path}\nRun captioning first.")
        return

    # Check for images
    from pathlib import Path
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    local_files = [p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not local_files:
        main_window._show_error("No images found in images directory.")
        return

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window._append_mm_log("")
    main_window._append_mm_log("=== HuggingFace Upload started ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Uploading...")
    main_window.mm_upload_button.setEnabled(False)
    main_window.mm_stop_upload_button.setEnabled(True)
    main_window.mm_upload_stop_flag = threading.Event()

    # Start worker thread
    main_window.mm_upload_thread = threading.Thread(
        target=hf_upload_worker,
        args=(
            output_folder,
            repo_id,
            hf_token,
            is_private,
            shard_size,
            batch_size,
            resume,
            main_window.mm_upload_stop_flag,
            main_window.mm_queue,
        ),
        daemon=True,
    )
    main_window.mm_upload_thread.start()
    main_window.timer.start()


def stop_hf_upload(main_window):
    """Stop the HuggingFace upload process"""
    if hasattr(main_window, 'mm_upload_stop_flag'):
        main_window.mm_upload_stop_flag.set()
    main_window._append_mm_log("Stopping upload...")
    main_window.mm_stop_upload_button.setEnabled(False)


def hf_upload_worker(work_dir, repo_id, token, is_private, shard_size, batch_size, resume, stop_flag, q):
    """Worker function to upload dataset to HuggingFace"""
    import math
    import json
    import shutil
    import tempfile
    from pathlib import Path
    
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
            q.put(("log", f"📁 Working directory: {work_dir}"))
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

