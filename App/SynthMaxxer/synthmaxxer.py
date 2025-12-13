"""
SynthMaxxer module - Generation tab functionality
Handles conversation generation UI and logic
"""
import os
import queue
import threading
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QGroupBox, QScrollArea,
    QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt

from App.SynthMaxxer.worker import worker


def build_synthmaxxer_tab(main_window):
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    """Build the SynthMaxxer (Generation) tab UI"""
    generation_tab = QWidget()
    generation_layout = QVBoxLayout()
    generation_layout.setContentsMargins(0, 0, 0, 0)
    generation_tab.setLayout(generation_layout)
    
    gen_header = QHBoxLayout()
    main_window.start_button = QPushButton("Start Generation")
    main_window.start_button.clicked.connect(lambda: start_generation(main_window))
    main_window.stop_button = QPushButton("Stop")
    main_window.stop_button.clicked.connect(lambda: stop_generation(main_window))
    main_window.stop_button.setEnabled(False)
    gen_header.addStretch()
    gen_header.addWidget(main_window.start_button)
    gen_header.addWidget(main_window.stop_button)
    generation_layout.addLayout(gen_header)

    main_split = QHBoxLayout()
    main_split.setSpacing(14)
    generation_layout.addLayout(main_split, stretch=1)

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

    right_container = QWidget()
    right_panel = QVBoxLayout()
    right_panel.setSpacing(10)
    right_container.setLayout(right_panel)

    main_split.addWidget(right_container, stretch=2)

    # Output Configuration
    output_group = QGroupBox("📁 Output Configuration")
    output_layout = QFormLayout()
    output_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    output_layout.setHorizontalSpacing(10)
    output_layout.setVerticalSpacing(6)
    output_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    output_group.setLayout(output_layout)

    output_dir_row, _ = create_file_browse_row(
        line_edit_name="output_dir_edit",
        placeholder_text="outputs",
        on_browse_clicked=lambda: browse_output_dir(main_window)
    )
    main_window.output_dir_edit = output_dir_row.itemAt(0).widget()
    output_layout.addRow(QLabel("Output Directory:"), _wrap_row(output_dir_row))

    left_panel.addWidget(output_group)

    # Generation Settings
    generation_group = QGroupBox("⚙️ Generation Settings")
    generation_layout = QFormLayout()
    generation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    generation_layout.setHorizontalSpacing(10)
    generation_layout.setVerticalSpacing(6)
    generation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    generation_group.setLayout(generation_layout)

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

    main_window.min_turns_spin = QSpinBox()
    main_window.min_turns_spin.setRange(0, 100)
    main_window.min_turns_spin.setValue(1)
    main_window.min_turns_spin.setMaximumWidth(80)
    main_window.min_turns_spin.setToolTip("Minimum number of assistant turns before stopping")
    min_turns_row = QHBoxLayout()
    min_turns_row.addWidget(main_window.min_turns_spin)
    min_turns_row.addStretch()
    generation_layout.addRow(QLabel("Min Turns:"), _wrap_row(min_turns_row))

    left_panel.addWidget(generation_group)

    # Conversation Configuration
    conversation_group = QGroupBox("💬 Conversation Configuration")
    conversation_layout = QFormLayout()
    conversation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    conversation_layout.setHorizontalSpacing(10)
    conversation_layout.setVerticalSpacing(6)
    conversation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    conversation_group.setLayout(conversation_layout)

    main_window.system_message_edit = QPlainTextEdit()
    main_window.system_message_edit.setPlaceholderText("System message for the conversation")
    main_window.system_message_edit.setFixedHeight(60)
    conversation_layout.addRow(QLabel("System Message:"), main_window.system_message_edit)

    main_window.user_first_message_edit = QPlainTextEdit()
    main_window.user_first_message_edit.setPlaceholderText("First user message")
    main_window.user_first_message_edit.setFixedHeight(60)
    conversation_layout.addRow(QLabel("User First Message:"), main_window.user_first_message_edit)

    main_window.assistant_first_message_edit = QPlainTextEdit()
    main_window.assistant_first_message_edit.setPlaceholderText("First assistant message")
    main_window.assistant_first_message_edit.setFixedHeight(60)
    conversation_layout.addRow(QLabel("Assistant First Message:"), main_window.assistant_first_message_edit)

    # Tags
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

    main_window.is_instruct_check = QCheckBox("Instruct Mode (min_turns=0, start_index=0, stopPercentage=0.25)")
    main_window.is_instruct_check.setChecked(False)
    conversation_layout.addRow("", main_window.is_instruct_check)

    left_panel.addWidget(conversation_group)

    # Filter Settings
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

    left_panel.addWidget(filter_group)
    left_panel.addStretch(1)

    # Right panel - Logs
    progress_group, main_window.log_view = create_log_view()
    right_panel.addWidget(progress_group, stretch=1)
    
    return generation_tab


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def browse_output_dir(main_window):
    """Browse for output directory"""
    from PyQt5.QtWidgets import QFileDialog
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select output directory",
        "",
    )
    if directory:
        main_window.output_dir_edit.setText(directory)


def start_generation(main_window):
    """Start the generation process"""
    # Validate inputs
    api_key = main_window.api_key_edit.text().strip()
    endpoint = main_window.endpoint_edit.text().strip()
    model = main_window.model_combo.currentText().strip()
    output_dir = main_window.output_dir_edit.text().strip()

    if not api_key:
        main_window._show_error("Please enter your API key.")
        return
    if not endpoint:
        main_window._show_error("Please enter the API endpoint.")
        return
    if not model:
        main_window._show_error("Please enter the model name.")
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


def stop_generation(main_window):
    """Stop the generation process"""
    main_window.stop_flag.set()
    main_window._append_log("Stopping generation...")
    main_window.stop_button.setEnabled(False)
