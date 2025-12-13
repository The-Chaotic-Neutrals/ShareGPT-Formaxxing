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
    outputs_dir = os.path.join(repo_root, "outputs")
    default_output = os.path.join(outputs_dir, "image_captions.parquet")
    output_row, _ = create_file_browse_row(
        line_edit_name="mm_output_edit",
        placeholder_text="Leave empty to auto-generate in outputs folder (will be .parquet)",
        default_text=default_output,
        on_browse_clicked=lambda: browse_mm_output(main_window)
    )
    main_window.mm_output_edit = output_row.itemAt(0).widget()
    mm_files_layout.addRow(QLabel("Output JSONL:"), _wrap_row(output_row))
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
    main_window.mm_caption_prompt_edit.setPlaceholderText("Describe what you see in this image in detail. Include all important elements, objects, people, text, colors, composition, and context.")
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
    """Browse for output file"""
    from PyQt5.QtWidgets import QFileDialog
    # Get repo root (go up from App/SynthMaxxer to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "outputs")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    current_path = main_window.mm_output_edit.text().strip()
    if current_path and os.path.dirname(current_path):
        start_dir = os.path.dirname(current_path)
    else:
        start_dir = default_path
    
    filename, _ = QFileDialog.getSaveFileName(
        main_window,
        "Select output Parquet file",
        os.path.join(start_dir, "image_captions.parquet"),
        "Parquet files (*.parquet);;All files (*.*)",
    )
    if filename:
        main_window.mm_output_edit.setText(filename)


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
    output_file = main_window.mm_output_edit.text().strip()
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

    # Normalize output file path (fix any App\outputs paths)
    if output_file and ("App\\outputs" in output_file or "App/outputs" in output_file):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(repo_root, "outputs")
        filename = os.path.basename(output_file)
        output_file = os.path.join(outputs_dir, filename)
        main_window.mm_output_edit.setText(output_file)
    
    # Auto-generate output filename if not provided
    if not output_file:
        # Get repo root (go up from App/SynthMaxxer to repo root)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(repo_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(outputs_dir, f"image_captions_{timestamp}.parquet")
        main_window.mm_output_edit.setText(output_file)

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.mm_log_view.clear()
    main_window._append_mm_log("=== Image Captioning started ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Captioning...")
    main_window.mm_start_button.setEnabled(False)
    main_window.mm_stop_button.setEnabled(True)
    main_window.mm_stop_flag = threading.Event()

    main_window.mm_queue = queue.Queue()
    
    # Log path correction if it happened
    if output_file and ("App\\outputs" not in output_file and "App/outputs" not in output_file):
        # Path is correct, log it
        main_window._append_mm_log(f"Output file: {output_file}")

    # Start worker thread
    main_window.mm_worker_thread = threading.Thread(
        target=image_captioning_worker,
        args=(
            image_dir,
            output_file,
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

