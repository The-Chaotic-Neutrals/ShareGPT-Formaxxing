"""
HuggingFace module - HuggingFace dataset and model downloader tab functionality
Handles HuggingFace dataset/model downloading UI and logic
"""
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QScrollArea,
    QFrame
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


def start_hf_download(main_window):
    """Start HuggingFace dataset download"""
    # Placeholder - implement actual download logic
    main_window._append_hf_log("HuggingFace dataset download not yet implemented")


def stop_hf_download(main_window):
    """Stop HuggingFace dataset download"""
    # Placeholder - implement actual stop logic
    main_window._append_hf_log("Stopping HuggingFace dataset download...")


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

