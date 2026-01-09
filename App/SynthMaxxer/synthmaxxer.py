"""
SynthMaxxer module - Unified generation and processing tab functionality
Handles conversation generation, JSONL processing, improvement, and extension
"""
import os
import queue
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QGroupBox, QScrollArea,
    QFrame, QSizePolicy, QFileDialog, QMessageBox,
    QComboBox
)
from PyQt5.QtCore import Qt

from App.SynthMaxxer.worker import worker
from App.SynthMaxxer.processing_worker import processing_worker


def build_synthmaxxer_tab(main_window):
    """Build the unified SynthMaxxer tab UI with generation and processing capabilities"""
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    
    tab = QWidget()
    tab_layout = QVBoxLayout()
    tab_layout.setContentsMargins(0, 0, 0, 0)
    tab.setLayout(tab_layout)
    
    # Header with action buttons
    header = QHBoxLayout()
    
    # Mode selector
    mode_label = QLabel("Mode:")
    mode_label.setStyleSheet("font-weight: bold;")
    main_window.synth_mode_combo = QComboBox()
    main_window.synth_mode_combo.addItems(["Generate New", "Process Existing"])
    main_window.synth_mode_combo.setToolTip("Generate new conversations or process existing JSONL files")
    main_window.synth_mode_combo.currentTextChanged.connect(lambda text: _update_mode_ui(main_window, text))
    main_window.synth_mode_combo.setFixedWidth(160)
    
    header.addWidget(mode_label)
    header.addWidget(main_window.synth_mode_combo)
    header.addStretch()
    
    # Action buttons
    main_window.start_button = QPushButton("Start")
    main_window.start_button.clicked.connect(lambda: _start_action(main_window))
    main_window.stop_button = QPushButton("Stop")
    main_window.stop_button.clicked.connect(lambda: _stop_action(main_window))
    main_window.stop_button.setEnabled(False)
    
    header.addWidget(main_window.start_button)
    header.addWidget(main_window.stop_button)
    tab_layout.addLayout(header)

    # Main split layout
    main_split = QHBoxLayout()
    main_split.setSpacing(14)
    tab_layout.addLayout(main_split, stretch=1)

    # Left panel (scrollable)
    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    left_scroll.setFrameShape(QFrame.NoFrame)

    left_container = QWidget()
    left_panel = QVBoxLayout()
    left_panel.setSpacing(10)
    left_container.setLayout(left_panel)
    left_scroll.setWidget(left_container)

    main_split.addWidget(left_scroll, stretch=5)

    # Right panel
    right_container = QWidget()
    right_panel = QVBoxLayout()
    right_panel.setSpacing(10)
    right_container.setLayout(right_panel)

    main_split.addWidget(right_container, stretch=2)

    # Build UI sections
    _build_files_section(main_window, left_panel)
    _build_generation_section(main_window, left_panel)
    _build_processing_section(main_window, left_panel)
    _build_conversation_section(main_window, left_panel)
    _build_filter_section(main_window, left_panel)
    
    left_panel.addStretch(1)

    # Right panel - Logs
    progress_group, main_window.log_view = create_log_view()
    # Also alias for processing log
    main_window.proc_log_view = main_window.log_view
    right_panel.addWidget(progress_group, stretch=1)
    
    # Set initial mode
    _update_mode_ui(main_window, "Generate New")
    
    return tab


def _build_files_section(main_window, parent_layout):
    """Build the file input/output configuration section"""
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row
    
    files_group = QGroupBox("📁 Files")
    files_layout = QFormLayout()
    files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    files_layout.setHorizontalSpacing(10)
    files_layout.setVerticalSpacing(6)
    files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    files_group.setLayout(files_layout)

    # Input file (for processing mode)
    input_row, _ = create_file_browse_row(
        line_edit_name="proc_input_edit",
        placeholder_text="Path to input JSONL (for processing mode)",
        on_browse_clicked=lambda: _browse_input_file(main_window)
    )
    main_window.proc_input_edit = input_row.itemAt(0).widget()
    files_layout.addRow(QLabel("Input JSONL:"), _wrap_row(input_row))

    # Output directory/file
    repo_root = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(repo_root, "outputs")
    
    output_row, _ = create_file_browse_row(
        line_edit_name="output_dir_edit",
        placeholder_text="outputs",
        default_text=outputs_dir,
        on_browse_clicked=lambda: _browse_output(main_window)
    )
    main_window.output_dir_edit = output_row.itemAt(0).widget()
    # Alias for processing
    main_window.proc_output_edit = main_window.output_dir_edit
    files_layout.addRow(QLabel("Output:"), _wrap_row(output_row))

    parent_layout.addWidget(files_group)


def _build_generation_section(main_window, parent_layout):
    """Build the generation settings section (for Generate New mode)"""
    main_window.generation_group = QGroupBox("⚙️ Generation Settings")
    generation_layout = QFormLayout()
    generation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    generation_layout.setHorizontalSpacing(10)
    generation_layout.setVerticalSpacing(6)
    generation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    main_window.generation_group.setLayout(generation_layout)

    # Delay settings
    delay_row = QHBoxLayout()
    main_window.min_delay_spin = QDoubleSpinBox()
    main_window.min_delay_spin.setRange(0.0, 10.0)
    main_window.min_delay_spin.setValue(0.1)
    main_window.min_delay_spin.setSingleStep(0.1)
    main_window.min_delay_spin.setDecimals(2)
    main_window.min_delay_spin.setMaximumWidth(80)
    main_window.max_delay_spin = QDoubleSpinBox()
    main_window.max_delay_spin.setRange(0.0, 10.0)
    main_window.max_delay_spin.setValue(0.5)
    main_window.max_delay_spin.setSingleStep(0.1)
    main_window.max_delay_spin.setDecimals(2)
    main_window.max_delay_spin.setMaximumWidth(80)
    delay_row.addWidget(QLabel("Min:"))
    delay_row.addWidget(main_window.min_delay_spin)
    delay_row.addSpacing(8)
    delay_row.addWidget(QLabel("Max:"))
    delay_row.addWidget(main_window.max_delay_spin)
    delay_row.addStretch()
    generation_layout.addRow(QLabel("Delay (seconds):"), _wrap_row(delay_row))

    # Stop probability
    main_window.stop_percentage_spin = QDoubleSpinBox()
    main_window.stop_percentage_spin.setRange(0.0, 1.0)
    main_window.stop_percentage_spin.setValue(0.05)
    main_window.stop_percentage_spin.setSingleStep(0.01)
    main_window.stop_percentage_spin.setDecimals(2)
    main_window.stop_percentage_spin.setMaximumWidth(80)
    main_window.stop_percentage_spin.setToolTip("Probability of stopping after minimum turns")
    stop_row = QHBoxLayout()
    stop_row.addWidget(main_window.stop_percentage_spin)
    stop_row.addStretch()
    generation_layout.addRow(QLabel("Stop Probability:"), _wrap_row(stop_row))

    # Min turns
    main_window.min_turns_spin = QSpinBox()
    main_window.min_turns_spin.setRange(0, 100)
    main_window.min_turns_spin.setValue(1)
    main_window.min_turns_spin.setMaximumWidth(80)
    main_window.min_turns_spin.setToolTip("Minimum number of assistant turns before stopping")
    min_turns_row = QHBoxLayout()
    min_turns_row.addWidget(main_window.min_turns_spin)
    min_turns_row.addStretch()
    generation_layout.addRow(QLabel("Min Turns:"), _wrap_row(min_turns_row))

    parent_layout.addWidget(main_window.generation_group)


def _build_processing_section(main_window, parent_layout):
    """Build the processing settings section (for Process Existing mode)"""
    main_window.processing_group = QGroupBox("⚙️ Processing Settings")
    processing_layout = QFormLayout()
    processing_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    processing_layout.setHorizontalSpacing(10)
    processing_layout.setVerticalSpacing(6)
    processing_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    main_window.processing_group.setLayout(processing_layout)

    # Line range
    range_row = QHBoxLayout()
    main_window.proc_start_line_edit = QLineEdit()
    main_window.proc_start_line_edit.setPlaceholderText("from")
    main_window.proc_start_line_edit.setMaximumWidth(80)
    main_window.proc_end_line_edit = QLineEdit()
    main_window.proc_end_line_edit.setPlaceholderText("to")
    main_window.proc_end_line_edit.setMaximumWidth(80)
    range_row.addWidget(QLabel("from"))
    range_row.addWidget(main_window.proc_start_line_edit)
    range_row.addSpacing(8)
    range_row.addWidget(QLabel("to"))
    range_row.addWidget(main_window.proc_end_line_edit)
    range_row.addStretch()
    processing_layout.addRow(QLabel("Process lines (1-based):"), _wrap_row(range_row))

    # Rewrite checkbox
    main_window.proc_rewrite_check = QCheckBox("Rewrite existing entries in range")
    main_window.proc_rewrite_check.setChecked(True)
    processing_layout.addRow("", main_window.proc_rewrite_check)

    # Extra pairs
    extra_row = QHBoxLayout()
    main_window.proc_extra_pairs_spin = QSpinBox()
    main_window.proc_extra_pairs_spin.setRange(0, 100)
    main_window.proc_extra_pairs_spin.setValue(0)
    main_window.proc_extra_pairs_spin.setMaximumWidth(80)
    extra_row.addWidget(main_window.proc_extra_pairs_spin)
    extra_row.addStretch()
    processing_layout.addRow(QLabel("Extra pairs per entry:"), _wrap_row(extra_row))

    # Generate new entries
    new_row = QHBoxLayout()
    main_window.proc_num_new_edit = QLineEdit()
    main_window.proc_num_new_edit.setPlaceholderText("0")
    main_window.proc_num_new_edit.setMaximumWidth(80)
    new_row.addWidget(main_window.proc_num_new_edit)
    new_row.addStretch()
    processing_layout.addRow(QLabel("Generate new entries:"), _wrap_row(new_row))

    parent_layout.addWidget(main_window.processing_group)


def _build_conversation_section(main_window, parent_layout):
    """Build the conversation configuration section"""
    conversation_group = QGroupBox("💬 Conversation Configuration")
    conversation_layout = QFormLayout()
    conversation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    conversation_layout.setHorizontalSpacing(10)
    conversation_layout.setVerticalSpacing(6)
    conversation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    conversation_group.setLayout(conversation_layout)

    # System message (used as system prompt for both modes)
    main_window.system_message_edit = QPlainTextEdit()
    main_window.system_message_edit.setPlaceholderText("System message / character prompt for the conversation")
    main_window.system_message_edit.setFixedHeight(60)
    # Alias for processing
    main_window.proc_system_prompt_edit = main_window.system_message_edit
    conversation_layout.addRow(QLabel("System Message:"), main_window.system_message_edit)

    # User first message (for generation mode)
    main_window.user_first_message_edit = QPlainTextEdit()
    main_window.user_first_message_edit.setPlaceholderText("First user message (for generation mode)")
    main_window.user_first_message_edit.setFixedHeight(60)
    conversation_layout.addRow(QLabel("User First Message:"), main_window.user_first_message_edit)

    # Assistant first message (for generation mode)
    main_window.assistant_first_message_edit = QPlainTextEdit()
    main_window.assistant_first_message_edit.setPlaceholderText("First assistant message (for generation mode)")
    main_window.assistant_first_message_edit.setFixedHeight(60)
    conversation_layout.addRow(QLabel("Assistant First Message:"), main_window.assistant_first_message_edit)

    # Tags (for generation mode)
    tags_row1 = QHBoxLayout()
    main_window.user_start_tag_edit = QLineEdit()
    main_window.user_start_tag_edit.setPlaceholderText("<human_turn>")
    main_window.user_start_tag_edit.setMaximumWidth(120)
    main_window.user_end_tag_edit = QLineEdit()
    main_window.user_end_tag_edit.setPlaceholderText("</human_turn>")
    main_window.user_end_tag_edit.setMaximumWidth(120)
    tags_row1.addWidget(QLabel("User Start:"))
    tags_row1.addWidget(main_window.user_start_tag_edit)
    tags_row1.addSpacing(8)
    tags_row1.addWidget(QLabel("User End:"))
    tags_row1.addWidget(main_window.user_end_tag_edit)
    tags_row1.addStretch()
    conversation_layout.addRow(QLabel("User Tags:"), _wrap_row(tags_row1))

    tags_row2 = QHBoxLayout()
    main_window.assistant_start_tag_edit = QLineEdit()
    main_window.assistant_start_tag_edit.setPlaceholderText("<claude_turn>")
    main_window.assistant_start_tag_edit.setMaximumWidth(120)
    main_window.assistant_end_tag_edit = QLineEdit()
    main_window.assistant_end_tag_edit.setPlaceholderText("</claude_turn>")
    main_window.assistant_end_tag_edit.setMaximumWidth(120)
    tags_row2.addWidget(QLabel("Assistant Start:"))
    tags_row2.addWidget(main_window.assistant_start_tag_edit)
    tags_row2.addSpacing(8)
    tags_row2.addWidget(QLabel("Assistant End:"))
    tags_row2.addWidget(main_window.assistant_end_tag_edit)
    tags_row2.addStretch()
    conversation_layout.addRow(QLabel("Assistant Tags:"), _wrap_row(tags_row2))

    # Options
    main_window.is_instruct_check = QCheckBox("Instruct Mode (min_turns=0, start_index=0, stopPercentage=0.25)")
    main_window.is_instruct_check.setChecked(False)
    conversation_layout.addRow("", main_window.is_instruct_check)

    main_window.proc_reply_in_character_check = QCheckBox("Reply in character & inject system prompt")
    main_window.proc_reply_in_character_check.setChecked(False)
    conversation_layout.addRow("", main_window.proc_reply_in_character_check)

    main_window.proc_dynamic_names_check = QCheckBox("Dynamic Names Mode (auto-generate & cache names)")
    main_window.proc_dynamic_names_check.setChecked(False)
    conversation_layout.addRow("", main_window.proc_dynamic_names_check)

    parent_layout.addWidget(conversation_group)


def _build_filter_section(main_window, parent_layout):
    """Build the filter settings section"""
    filter_group = QGroupBox("🚫 Filter Settings")
    filter_layout = QFormLayout()
    filter_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    filter_layout.setHorizontalSpacing(10)
    filter_layout.setVerticalSpacing(6)
    filter_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    filter_group.setLayout(filter_layout)

    main_window.refusal_phrases_edit = QPlainTextEdit()
    main_window.refusal_phrases_edit.setPlaceholderText("One phrase per line\nExample:\nUpon further reflection\nI can't engage")
    main_window.refusal_phrases_edit.setFixedHeight(60)
    filter_layout.addRow(QLabel("Refusal Phrases:"), main_window.refusal_phrases_edit)

    main_window.force_retry_phrases_edit = QPlainTextEdit()
    main_window.force_retry_phrases_edit.setPlaceholderText("One phrase per line\nExample:\nshivers down")
    main_window.force_retry_phrases_edit.setFixedHeight(60)
    filter_layout.addRow(QLabel("Force Retry Phrases:"), main_window.force_retry_phrases_edit)

    parent_layout.addWidget(filter_group)


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def _update_mode_ui(main_window, mode_text):
    """Update UI visibility based on selected mode"""
    is_generate = mode_text == "Generate New"
    
    # Show/hide generation-specific settings
    main_window.generation_group.setVisible(is_generate)
    
    # Show/hide processing-specific settings
    main_window.processing_group.setVisible(not is_generate)
    
    # Update input file requirement text
    if is_generate:
        main_window.proc_input_edit.setPlaceholderText("(Optional) Input JSONL for examples")
    else:
        main_window.proc_input_edit.setPlaceholderText("Path to input JSONL to process")


def _browse_input_file(main_window):
    """Browse for input JSONL file"""
    filename, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select input JSONL file",
        "",
        "JSONL files (*.jsonl);;All files (*.*)",
    )
    if filename:
        main_window.proc_input_edit.setText(filename)


def _browse_output(main_window):
    """Browse for output directory or file"""
    mode = main_window.synth_mode_combo.currentText()
    
    if mode == "Generate New":
        # Browse for directory
        directory = QFileDialog.getExistingDirectory(
            main_window,
            "Select output directory",
            "",
        )
        if directory:
            main_window.output_dir_edit.setText(directory)
    else:
        # Browse for file
        default_path = main_window.output_dir_edit.text().strip()
        if not default_path or os.path.isdir(default_path):
            default_path = os.path.join(default_path or "", "processed_output.jsonl")
        
        filename, _ = QFileDialog.getSaveFileName(
            main_window,
            "Select output JSONL file",
            default_path,
            "JSONL files (*.jsonl);;All files (*.*)",
        )
        if filename:
            main_window.output_dir_edit.setText(filename)


def _start_action(main_window):
    """Start generation or processing based on current mode"""
    mode = main_window.synth_mode_combo.currentText()
    
    if mode == "Generate New":
        _start_generation(main_window)
    else:
        _start_processing(main_window)


def _stop_action(main_window):
    """Stop the current operation"""
    main_window.stop_flag.set()
    if hasattr(main_window, 'proc_stop_flag'):
        main_window.proc_stop_flag.set()
    main_window._append_log("Stopping...")
    main_window.stop_button.setEnabled(False)


def _start_generation(main_window):
    """Start the generation process"""
    # Validate inputs
    api_key = main_window.api_key_edit.text().strip()
    endpoint = main_window.endpoint_edit.text().strip()
    model = main_window.model_combo.currentText().strip()
    output_dir = main_window.output_dir_edit.text().strip()

    if not api_key:
        main_window._show_error("Please enter your API key in the API Config tab.")
        return
    if not endpoint:
        main_window._show_error("Please enter the API endpoint in the API Config tab.")
        return
    if not model:
        main_window._show_error("Please enter the model name in the API Config tab.")
        return
    if not output_dir:
        main_window._show_error("Please enter the output directory.")
        return

    # Save config
    main_window._save_config()

    # Reset UI state
    main_window.log_view.clear()
    main_window._append_log("=== Generation started ===")
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.setWindowTitle(f"{APP_TITLE} - Generating...")
    main_window.start_button.setEnabled(False)
    main_window.stop_button.setEnabled(True)
    main_window.stop_flag.clear()

    main_window.queue = queue.Queue()

    # Collect all configuration values
    system_message = main_window.system_message_edit.toPlainText().strip()
    user_first_message = main_window.user_first_message_edit.toPlainText().strip()
    assistant_first_message = main_window.assistant_first_message_edit.toPlainText().strip()
    user_start_tag = main_window.user_start_tag_edit.text().strip()
    user_end_tag = main_window.user_end_tag_edit.text().strip()
    assistant_start_tag = main_window.assistant_start_tag_edit.text().strip()
    assistant_end_tag = main_window.assistant_end_tag_edit.text().strip()
    is_instruct = main_window.is_instruct_check.isChecked()
    min_delay = main_window.min_delay_spin.value()
    max_delay = main_window.max_delay_spin.value()
    stop_percentage = main_window.stop_percentage_spin.value()
    min_turns = main_window.min_turns_spin.value()
    refusal_phrases = [p.strip() for p in main_window.refusal_phrases_edit.toPlainText().split('\n') if p.strip()]
    force_retry_phrases = [p.strip() for p in main_window.force_retry_phrases_edit.toPlainText().split('\n') if p.strip()]
    api_type = main_window.api_type_combo.currentText()

    # Start worker thread
    main_window.worker_thread = threading.Thread(
        target=worker,
        args=(
            api_key,
            endpoint,
            model,
            output_dir,
            system_message,
            user_first_message,
            assistant_first_message,
            user_start_tag,
            user_end_tag,
            assistant_start_tag,
            assistant_end_tag,
            is_instruct,
            min_delay,
            max_delay,
            stop_percentage,
            min_turns,
            refusal_phrases,
            force_retry_phrases,
            api_type,
            main_window.stop_flag,
            main_window.queue,
        ),
        daemon=True,
    )
    main_window.worker_thread.start()
    main_window.timer.start()


def _start_processing(main_window):
    """Start the processing operation"""
    input_file = main_window.proc_input_edit.text().strip()
    output_file = main_window.output_dir_edit.text().strip()
    api_key = main_window.api_key_edit.text().strip()
    system_prompt = main_window.system_message_edit.toPlainText().strip()
    start_line_str = main_window.proc_start_line_edit.text().strip()
    end_line_str = main_window.proc_end_line_edit.text().strip()
    do_rewrite = main_window.proc_rewrite_check.isChecked()
    reply_in_character = main_window.proc_reply_in_character_check.isChecked()
    dynamic_names_mode = main_window.proc_dynamic_names_check.isChecked()
    num_new_str = main_window.proc_num_new_edit.text().strip()
    
    num_new = 0
    if num_new_str:
        try:
            num_new = int(num_new_str)
            if num_new < 0:
                raise ValueError("Negative value")
        except ValueError:
            main_window._show_error("Number of new entries must be a non-negative integer.")
            return
    extra_pairs = int(main_window.proc_extra_pairs_spin.value())

    # Auto-generate output filename if not provided or is directory
    if not output_file or os.path.isdir(output_file):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        outputs_dir = output_file if output_file and os.path.isdir(output_file) else os.path.join(repo_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        if input_file:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(outputs_dir, f"{base_name}_processed.jsonl")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(outputs_dir, f"processed_{timestamp}.jsonl")
        
        main_window.output_dir_edit.setText(output_file)
        main_window._append_log(f"Auto-generated output file: {output_file}")

    if not api_key:
        main_window._show_error("Please enter your API key in the API Config tab.")
        return

    # Parse line range
    start_line = None
    if start_line_str:
        try:
            start_line = int(start_line_str)
            if start_line < 1:
                raise ValueError
        except ValueError:
            main_window._show_error("Start line must be a positive integer.")
            return

    end_line = None
    if end_line_str:
        try:
            end_line = int(end_line_str)
            if end_line < 1:
                raise ValueError
        except ValueError:
            main_window._show_error("End line must be a positive integer.")
            return

    if start_line is not None and end_line is not None and end_line < start_line:
        main_window._show_error("End line cannot be less than start line.")
        return

    model_name = main_window.model_combo.currentText().strip()
    if not model_name:
        main_window._show_error("Please select a model in the API Config tab.")
        return
    
    api_type = main_window.api_type_combo.currentText()
    endpoint = main_window.endpoint_edit.text().strip()

    # Reset UI state
    main_window.log_view.clear()
    main_window._append_log("=== Processing started ===")
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.setWindowTitle(f"{APP_TITLE} - Processing...")
    main_window.start_button.setEnabled(False)
    main_window.stop_button.setEnabled(True)
    
    main_window._save_config()

    main_window.proc_queue = queue.Queue()
    main_window.proc_stop_flag = threading.Event()
    
    main_window._append_log(f"Input: {input_file or 'None'}")
    main_window._append_log(f"Output: {output_file}")
    main_window._append_log(f"Model: {model_name}")
    main_window._append_log(f"Settings: rewrite={do_rewrite}, extra_pairs={extra_pairs}, num_new={num_new}")

    def worker_wrapper(*args, **kwargs):
        try:
            main_window.proc_queue.put(("log", "Worker thread started"))
            processing_worker(*args, **kwargs)
            if hasattr(main_window, 'proc_stop_flag') and main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Processing stopped by user"))
            else:
                main_window.proc_queue.put(("success", "Processing completed successfully"))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            main_window.proc_queue.put(("error", f"Worker crashed: {e}"))
            main_window.proc_queue.put(("log", f"Traceback:\n{tb}"))

    t = threading.Thread(
        target=worker_wrapper,
        args=(
            input_file,
            output_file,
            num_new,
            api_key,
            model_name,
            system_prompt,
            start_line,
            end_line,
            do_rewrite,
            extra_pairs,
            reply_in_character,
            dynamic_names_mode,
            main_window.proc_queue,
            main_window.proc_stop_flag,
            api_type,
            endpoint,
        ),
        daemon=True,
    )
    t.start()
    main_window.timer.start()
