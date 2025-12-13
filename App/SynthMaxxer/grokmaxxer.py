"""
GrokMaxxer module - Processing tab functionality
Handles JSONL file processing, improvement, extension, and human cache management
"""
import os
import queue
import threading
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QPlainTextEdit, QGroupBox, QScrollArea, QFrame,
    QSizePolicy, QMessageBox, QInputDialog
)
from PyQt5.QtCore import Qt

from App.SynthMaxxer.processing_worker import processing_worker

GLOBAL_HUMAN_CACHE_FILE = str(Path(__file__).parent / "global_human_cache.json")


def build_grokmaxxer_tab(main_window):
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    """Build the GrokMaxxer (Processing) tab UI"""
    processing_tab = QWidget()
    processing_layout = QVBoxLayout()
    processing_layout.setContentsMargins(0, 0, 0, 0)
    processing_tab.setLayout(processing_layout)
    
    proc_header = QHBoxLayout()
    main_window.process_button = QPushButton("Process")
    main_window.process_button.clicked.connect(lambda: process_files(main_window))
    main_window.proc_stop_button = QPushButton("Stop")
    main_window.proc_stop_button.clicked.connect(lambda: stop_processing(main_window))
    main_window.proc_stop_button.setEnabled(False)
    proc_header.addStretch()
    proc_header.addWidget(main_window.process_button)
    proc_header.addWidget(main_window.proc_stop_button)
    processing_layout.addLayout(proc_header)
    
    proc_split = QHBoxLayout()
    proc_split.setSpacing(14)
    processing_layout.addLayout(proc_split, stretch=1)
    
    proc_left_scroll = QScrollArea()
    proc_left_scroll.setWidgetResizable(True)
    proc_left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    proc_left_scroll.setFrameShape(QFrame.NoFrame)
    
    proc_left_container = QWidget()
    proc_left_panel = QVBoxLayout()
    proc_left_panel.setSpacing(10)
    proc_left_container.setLayout(proc_left_panel)
    proc_left_scroll.setWidget(proc_left_container)
    
    proc_split.addWidget(proc_left_scroll, stretch=3)
    
    proc_right_container = QWidget()
    proc_right_panel = QVBoxLayout()
    proc_right_panel.setSpacing(10)
    proc_right_container.setLayout(proc_right_panel)
    
    proc_split.addWidget(proc_right_container, stretch=4)
    
    # Build processing UI elements
    _build_processing_ui(main_window, proc_left_panel, proc_right_panel)
    
    return processing_tab


def _build_processing_ui(main_window, left_panel, right_panel):
    """Build the processing UI components"""
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    # Files group
    files_group = QGroupBox("📁 Files")
    files_layout = QFormLayout()
    files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    files_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    files_layout.setHorizontalSpacing(10)
    files_layout.setVerticalSpacing(6)
    files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    files_group.setLayout(files_layout)

    input_row, _ = create_file_browse_row(
        line_edit_name="proc_input_edit",
        placeholder_text="Path to input JSONL (optional for generation-only)",
        on_browse_clicked=lambda: browse_proc_input(main_window)
    )
    main_window.proc_input_edit = input_row.itemAt(0).widget()
    files_layout.addRow(QLabel("Input JSONL:"), _wrap_row(input_row))

    # Set default to outputs folder
    repo_root = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(repo_root, "outputs")
    default_output = os.path.join(outputs_dir, "processed_output.jsonl")
    output_row, _ = create_file_browse_row(
        line_edit_name="proc_output_edit",
        placeholder_text="Leave empty to auto-generate in outputs folder",
        default_text=default_output,
        on_browse_clicked=lambda: browse_proc_output(main_window)
    )
    main_window.proc_output_edit = output_row.itemAt(0).widget()
    files_layout.addRow(QLabel("Output JSONL:"), _wrap_row(output_row))
    left_panel.addWidget(files_group)

    # Processing mode group
    processing_group = QGroupBox("⚙️ Processing Mode")
    processing_layout = QFormLayout()
    processing_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    processing_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    processing_layout.setHorizontalSpacing(10)
    processing_layout.setVerticalSpacing(6)
    processing_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    processing_group.setLayout(processing_layout)

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

    main_window.proc_rewrite_check = QCheckBox("Rewrite existing entries in range")
    main_window.proc_rewrite_check.setChecked(True)
    processing_layout.addRow("", main_window.proc_rewrite_check)

    extra_row = QHBoxLayout()
    main_window.proc_extra_pairs_spin = QSpinBox()
    main_window.proc_extra_pairs_spin.setRange(0, 100)
    main_window.proc_extra_pairs_spin.setValue(0)
    main_window.proc_extra_pairs_spin.setMaximumWidth(80)
    extra_row.addWidget(main_window.proc_extra_pairs_spin)
    extra_row.addStretch()
    processing_layout.addRow(QLabel("Extra pairs per entry:"), _wrap_row(extra_row))

    new_row = QHBoxLayout()
    main_window.proc_num_new_edit = QLineEdit()
    main_window.proc_num_new_edit.setPlaceholderText("0")
    main_window.proc_num_new_edit.setMaximumWidth(80)
    new_row.addWidget(main_window.proc_num_new_edit)
    new_row.addStretch()
    processing_layout.addRow(QLabel("Generate new entries:"), _wrap_row(new_row))

    left_panel.addWidget(processing_group)

    # Prompt group
    prompt_group = QGroupBox("System / Character Prompt")
    prompt_layout = QFormLayout()
    prompt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    prompt_layout.setHorizontalSpacing(10)
    prompt_layout.setVerticalSpacing(6)
    prompt_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    prompt_group.setLayout(prompt_layout)

    main_window.proc_system_prompt_edit = QPlainTextEdit()
    main_window.proc_system_prompt_edit.setPlaceholderText("Character sheet or global system prompt (optional)")
    main_window.proc_system_prompt_edit.setFixedHeight(100)
    prompt_layout.addRow(QLabel("System prompt:"), main_window.proc_system_prompt_edit)

    main_window.proc_reply_in_character_check = QCheckBox(
        "Reply in character & inject prompt as the first system message"
    )
    main_window.proc_reply_in_character_check.setChecked(False)
    prompt_layout.addRow(QLabel("Character mode:"), main_window.proc_reply_in_character_check)

    main_window.proc_dynamic_names_check = QCheckBox("Dynamic Names Mode (auto-generate & cache names)")
    main_window.proc_dynamic_names_check.setChecked(False)
    prompt_layout.addRow(QLabel("Dynamic names:"), main_window.proc_dynamic_names_check)

    left_panel.addWidget(prompt_group)

    # Cache group
    cache_group = QGroupBox("🧠 Human Cache Management")
    cache_layout = QFormLayout()
    cache_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    cache_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    cache_layout.setHorizontalSpacing(10)
    cache_layout.setVerticalSpacing(6)
    cache_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    cache_group.setLayout(cache_layout)

    controls_row = QHBoxLayout()
    main_window.generate_cache_button = QPushButton("Generate Human Cache")
    main_window.generate_cache_button.clicked.connect(lambda: generate_human_cache(main_window))
    main_window.generate_cache_button.setToolTip("Generate new human turns for the cache")
    
    main_window.improve_cache_button = QPushButton("Improve Human Cache")
    main_window.improve_cache_button.clicked.connect(lambda: improve_human_cache(main_window))
    main_window.improve_cache_button.setToolTip("Rewrite existing human turns for better quality")
    
    main_window.proc_concurrency_spin = QSpinBox()
    main_window.proc_concurrency_spin.setRange(1, 20)
    main_window.proc_concurrency_spin.setValue(20)
    main_window.proc_concurrency_spin.setMaximumWidth(60)
    main_window.proc_concurrency_spin.setToolTip("Concurrent batches (safe: 1-20)")
    
    main_window.proc_batch_size_spin = QSpinBox()
    main_window.proc_batch_size_spin.setRange(16, 64)
    main_window.proc_batch_size_spin.setValue(16)
    main_window.proc_batch_size_spin.setMaximumWidth(60)
    main_window.proc_batch_size_spin.setToolTip("Batch size per API call (16 optimal for cache improvement)")
    
    controls_row.addWidget(main_window.generate_cache_button)
    controls_row.addSpacing(10)
    controls_row.addWidget(main_window.improve_cache_button)
    controls_row.addSpacing(10)
    controls_row.addWidget(QLabel("Concurrency:"))
    controls_row.addWidget(main_window.proc_concurrency_spin)
    controls_row.addSpacing(20)
    controls_row.addWidget(QLabel("Batch:"))
    controls_row.addWidget(main_window.proc_batch_size_spin)
    controls_row.addStretch()
    
    cache_layout.addRow("", _wrap_row(controls_row))
    left_panel.addWidget(cache_group)

    left_panel.addStretch(1)

    # Right panel - Logs
    proc_progress_group, main_window.proc_log_view = create_log_view()
    right_panel.addWidget(proc_progress_group, stretch=1)


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def browse_proc_input(main_window):
    """Browse for processing input file"""
    from PyQt5.QtWidgets import QFileDialog
    filename, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select input JSONL file",
        "",
        "JSONL files (*.jsonl);;All files (*.*)",
    )
    if filename:
        main_window.proc_input_edit.setText(filename)


def browse_proc_output(main_window):
    """Browse for processing output file"""
    from PyQt5.QtWidgets import QFileDialog
    # Default to outputs folder in repo root
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    # Get current value or use default
    current_path = main_window.proc_output_edit.text().strip()
    if current_path and os.path.dirname(current_path):
        start_dir = os.path.dirname(current_path)
    else:
        start_dir = default_path
    
    filename, _ = QFileDialog.getSaveFileName(
        main_window,
        "Select output JSONL file",
        os.path.join(start_dir, "processed_output.jsonl"),
        "JSONL files (*.jsonl);;All files (*.*)",
    )
    if filename:
        main_window.proc_output_edit.setText(filename)


def process_files(main_window):
    """Process files using the processing worker"""
    input_file = main_window.proc_input_edit.text().strip()
    output_file = main_window.proc_output_edit.text().strip()
    api_key = main_window.proc_api_key_edit.text().strip()
    system_prompt = main_window.proc_system_prompt_edit.toPlainText().strip()
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

    # Auto-generate output filename if not provided
    if not output_file:
        # Default to outputs folder in repo root
        repo_root = os.path.dirname(os.path.dirname(__file__))
        outputs_dir = os.path.join(repo_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Generate filename based on input file or timestamp
        if input_file:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(outputs_dir, f"{base_name}_processed.jsonl")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(outputs_dir, f"processed_{timestamp}.jsonl")
        
        main_window.proc_output_edit.setText(output_file)
        main_window._append_proc_log(f"Auto-generated output file: {output_file}")

    if not api_key:
        main_window._show_error("Please enter your Grok API key.")
        return

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

    model_name = main_window.proc_model_edit.text().strip()
    if not model_name:
        model_name = "grok-4.1-fast-non-reasoning"

    # Reset UI state for new run
    main_window.proc_log_view.clear()
    main_window._append_proc_log("=== New run started ===")
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.setWindowTitle(f"{APP_TITLE} - Processing...")
    main_window.process_button.setEnabled(False)
    main_window.proc_stop_button.setEnabled(True)
    
    # Save config before processing
    main_window._save_config()

    main_window.proc_queue = queue.Queue()
    main_window.proc_stop_flag = threading.Event()
    
    # Immediately log that we're starting
    main_window._append_proc_log("Creating worker thread...")
    main_window._append_proc_log(f"Input: {input_file or 'None'}")
    main_window._append_proc_log(f"Output: {output_file}")
    main_window._append_proc_log(f"API Key: {'***' + api_key[-4:] if len(api_key) > 4 else 'None'}")
    main_window._append_proc_log(f"Model: {model_name}")
    main_window._append_proc_log(f"Settings: rewrite={do_rewrite}, extra_pairs={extra_pairs}, num_new={num_new}")

    def worker_wrapper(*args, **kwargs):
        """Wrapper to catch any exceptions during thread startup"""
        try:
            main_window.proc_queue.put(("log", "Worker thread function called"))
            processing_worker(*args, **kwargs)
            # Check if stopped after worker completes
            if hasattr(main_window, 'proc_stop_flag') and main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Processing stopped by user"))
            else:
                main_window.proc_queue.put(("success", "Processing completed successfully"))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"CRITICAL: Worker thread crashed: {e}"
            main_window.proc_queue.put(("error", error_msg))
            main_window.proc_queue.put(("log", f"Traceback:\n{tb}"))
            print(f"WORKER CRASH: {error_msg}")
            print(f"Traceback:\n{tb}")

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
            False,  # rewrite_cache=False for process mode
            main_window.proc_queue,
            main_window.proc_stop_flag,
        ),
        daemon=True,
    )
    main_window._append_proc_log("Starting worker thread...")
    t.start()
    main_window._append_proc_log("Worker thread started, timer starting...")
    main_window.timer.start()
    main_window._append_proc_log("Timer started - queue should be checked every 100ms")


def stop_processing(main_window):
    """Stop the processing operation"""
    if hasattr(main_window, 'proc_stop_flag'):
        main_window.proc_stop_flag.set()
    if hasattr(main_window, '_append_proc_log'):
        main_window._append_proc_log("Stopping processing...")
    elif hasattr(main_window, 'proc_log_view'):
        main_window.proc_log_view.appendPlainText("Stopping processing...")
    if hasattr(main_window, 'proc_stop_button'):
        main_window.proc_stop_button.setEnabled(False)


def generate_human_cache(main_window):
    """Generate new human turns for the cache"""
    api_key = main_window.proc_api_key_edit.text().strip()
    system_prompt = main_window.proc_system_prompt_edit.toPlainText().strip()
    model_name = main_window.proc_model_edit.text().strip() or "grok-4.1-fast-non-reasoning"

    if not api_key:
        main_window._show_error("Please enter your Grok API key.")
        return

    cache_file = GLOBAL_HUMAN_CACHE_FILE
    
    # Ask user how many to generate
    num_turns, ok = QInputDialog.getInt(
        main_window,
        "Generate Human Cache",
        "How many human turns to generate?",
        value=100,
        min=1,
        max=1000,
        step=10
    )
    if not ok:
        return

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.proc_log_view.clear()
    main_window._append_proc_log("=== Generating Human Cache ===")
    main_window.setWindowTitle(f"{APP_TITLE} - Generating cache...")
    main_window.generate_cache_button.setEnabled(False)
    main_window.improve_cache_button.setEnabled(False)
    main_window.proc_stop_button.setEnabled(True)

    main_window.proc_queue = queue.Queue()
    main_window.proc_stop_flag = threading.Event()

    def worker_thread():
        try:
            from App.SynthMaxxer.llm_helpers import generate_human_turns
            from xai_sdk import Client
            import json
            import os

            client = Client(api_key=api_key, timeout=300)
            main_window.proc_queue.put(("log", f"Generating {num_turns} human turns..."))
            
            if main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Generation stopped by user"))
                return
            
            new_turns = generate_human_turns(
                client,
                model_name,
                system_prompt,
                num_turns=num_turns,
                temperature=1.0
            )
            
            if main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Generation stopped by user"))
                return
            
            # Load existing cache if it exists
            existing_turns = []
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        existing_turns = json.load(f)
                    main_window.proc_queue.put(("log", f"Loaded {len(existing_turns)} existing turns from cache"))
                except Exception as e:
                    main_window.proc_queue.put(("log", f"Could not load existing cache: {e}"))
            
            if main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Generation stopped by user"))
                return
            
            # Combine and save
            all_turns = existing_turns + new_turns
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(all_turns, f, indent=2, ensure_ascii=False)
            
            if main_window.proc_stop_flag.is_set():
                main_window.proc_queue.put(("stopped", "Generation stopped by user"))
                return
            
            main_window.proc_queue.put(("log", f"✅ Generated {len(new_turns)} new human turns"))
            main_window.proc_queue.put(("log", f"✅ Cache now contains {len(all_turns)} total human turns"))
            main_window.proc_queue.put(("success", f"Successfully generated {len(new_turns)} human turns"))
            
        except Exception as e:
            main_window.proc_queue.put(("error", f"Failed to generate human cache: {str(e)}"))
        finally:
            main_window.proc_queue.put(("stopped", None))

    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()
    main_window.timer.start()


def improve_human_cache(main_window):
    """Improve existing human turns in the cache"""
    api_key = main_window.proc_api_key_edit.text().strip()
    system_prompt = main_window.proc_system_prompt_edit.toPlainText().strip()
    model_name = main_window.proc_model_edit.text().strip() or "grok-4.1-fast-non-reasoning"

    if not api_key:
        main_window._show_error("Please enter your Grok API key.")
        return

    cache_file = GLOBAL_HUMAN_CACHE_FILE
    try:
        import json
        import os
        from xai_sdk import Client
        from xai_sdk.chat import system
        from json_repair import repair_json

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                human_turns = json.load(f)
            main_window._append_proc_log(f"Loaded {len(human_turns)} human turns from {cache_file}")
        else:
            human_turns = []
            main_window._append_proc_log(f"No cache file found at {cache_file}, starting empty")

        if not human_turns:
            # Don't block - just inform user, but allow them to proceed if they want
            reply = QMessageBox.question(
                main_window,
                "Empty Cache",
                "Human cache is empty. Would you like to generate some human turns first, or proceed with improvement?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                # Generate some human turns first
                main_window.proc_log_view.clear()
                main_window._append_proc_log("Generating initial human turns for cache...")
                try:
                    from App.SynthMaxxer.llm_helpers import generate_human_turns
                    from xai_sdk import Client
                    client = Client(api_key=api_key, timeout=300)
                    new_turns = generate_human_turns(
                        client,
                        model_name,
                        system_prompt,
                        num_turns=50,
                        temperature=1.0
                    )
                    human_turns = new_turns
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(human_turns, f, indent=2, ensure_ascii=False)
                    main_window._append_proc_log(f"Generated and saved {len(human_turns)} human turns to cache")
                    main_window._show_info(f"Generated {len(human_turns)} human turns. You can now improve them.")
                    # Continue with improvement after generation
                except Exception as gen_e:
                    main_window._show_error(f"Failed to generate human turns: {gen_e}")
                    return
            else:
                main_window._show_info("Cannot improve empty cache. Please generate entries first or generate human turns.")
                return

        # Reset UI state
        from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
        main_window.proc_log_view.clear()
        main_window._append_proc_log("=== Improving human cache ===")
        main_window.setWindowTitle(f"{APP_TITLE} - Improving cache...")
        main_window.process_button.setEnabled(False)
        main_window.improve_cache_button.setEnabled(False)
        main_window.proc_stop_button.setEnabled(True)

        main_window.proc_queue = queue.Queue()
        main_window.proc_stop_flag = threading.Event()

        def worker_thread():
            try:
                import concurrent.futures
                client = Client(api_key=api_key, timeout=300)
                
                BATCH_SIZE = main_window.proc_batch_size_spin.value()
                MAX_CONCURRENCY = main_window.proc_concurrency_spin.value()
                original_human_turns = human_turns.copy()
                num_batches = (len(original_human_turns) + BATCH_SIZE - 1) // BATCH_SIZE
                
                main_window.proc_queue.put(("log", f"Processing {len(original_human_turns)} human turns in {num_batches} batches of {BATCH_SIZE} (concurrency: {MAX_CONCURRENCY})..."))
                
                def process_batch(batch_idx):
                    from App.SynthMaxxer.llm_helpers import improve_entry
                    batch_start = batch_idx * BATCH_SIZE
                    batch = original_human_turns[batch_start:batch_start + BATCH_SIZE]
                    messages_text = "\n".join([
                        f"{i+1}. Domain: {item['domain']}\nMessage: {item['message']}"
                        for i, item in enumerate(batch)
                    ])

                    base_prompt = (
                        "You are a professional dataset curator creating PREMIUM quality human-LLM conversations. "
                        "Your job is to transform RAW, broken, synthetic text into AUTHENTIC human questions that "
                        "REAL people would type into ChatGPT/Claude/Grok. PRIORITIZE QUALITY over speed.\n\n"
                        "🚫 PROBLEMS TO ELIMINATE:\n"
                        "❌ Comma-delimited lists: 'feature A, feature B, feature C'\n"
                        "❌ Broken fragments: 'how to do X? Y? Z?'\n"
                        "❌ Robotic repetition: 'I want X. I need X. Please give X.'\n"
                        "❌ Synthetic phrasing: 'As an AI language model...'\n"
                        "❌ Bullet-point style: '- point 1 - point 2'\n\n"
                        "✅ REAL HUMAN PATTERNS:\n"
                        "• Complete sentences with natural flow\n"
                        "• Casual contractions: 'I'm', 'it's', 'you're', 'there's'\n"
                        "• Personal context: 'I've been trying...', 'In my project...'\n"
                        "• Specific scenarios: 'Yesterday I was working on...'\n"
                        "• Natural curiosity: 'Do you think...', 'What's the best way...'\n"
                        "• Enthusiasm/urgency: 'Really need help with...', 'Super excited to...'\n"
                        "• Typos/fillers optional: 'kinda', 'sorta', 'tbh'\n\n"
                        "📝 EXAMPLE TRANSFORMATIONS:\n"
                        "BAD: 'machine learning, neural networks, training data'\n"
                        "GOOD: \"Hey, I'm building a ML model but struggling with training data quality. "
                        "The neural network keeps overfitting even with dropout. Any tips on data augmentation?\"\n\n"
                        "BAD: 'write python code, web scraping, beautifulsoup'\n"
                        "GOOD: \"Can you help me write a Python web scraper? I want to pull product prices "
                        "from an ecommerce site using BeautifulSoup but keep getting parsing errors. Here's my code...\"\n\n"
                        "🎯 YOUR JOB: Rewrite EVERY message above using these exact patterns. "
                        "Make them 150-450 words. Add personality/context. PRESERVE DOMAIN exactly.\n\n"
                        f"RAW INPUT MESSAGES:\n{messages_text}\n\n"
                        "📤 RESPOND with ONLY valid JSON array, same order/format:\n"
                        '[{"domain": "EXACT SAME DOMAIN", "message": "FULL REALISTIC HUMAN QUESTION"}, ...]\n'
                        "NO OTHER TEXT. VALID JSON ONLY."
                    )
                    
                    prompt = base_prompt
                    if system_prompt.strip() and main_window.proc_reply_in_character_check.isChecked():
                        prompt = system_prompt.strip() + "\n\n" + base_prompt

                    try:
                        if main_window.proc_stop_flag.is_set():
                            return batch
                        main_window.proc_queue.put(("log", f"Sending batch {batch_idx + 1}/{num_batches} ({len(batch)} items)..."))
                        chat = client.chat.create(model=model_name, temperature=0.8)
                        chat.append(system(prompt))
                        response = chat.sample()
                        if main_window.proc_stop_flag.is_set():
                            return batch
                    
                        content = response.content
                        repaired = repair_json(content)
                        batch_improved = []
                        if repaired:
                            try:
                                batch_improved = json.loads(repaired)
                            except:
                                batch_improved = []

                        # Validate and filter improved turns
                        valid_batch = []
                        skipped = 0
                        for improved in batch_improved:
                            if isinstance(improved, dict) and 'message' in improved and 'domain' in improved and improved['message'].strip():
                                valid_batch.append(improved)
                            else:
                                skipped += 1
                        
                        # Pad incomplete batch with originals
                        for i in range(len(valid_batch), len(batch)):
                            valid_batch.append(batch[i])
                        
                        main_window.proc_queue.put(("log", f"Batch {batch_idx + 1}/{num_batches}: got {len(batch_improved)} items, {skipped} skipped, using {len(valid_batch)} valid"))
                        return valid_batch
                    except Exception as e:
                        main_window.proc_queue.put(("log", f"Batch {batch_idx + 1} failed: {str(e)}"))
                        return batch

                improved_turns = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
                    batch_futures = {
                        executor.submit(process_batch, i): i 
                        for i in range(num_batches)
                    }
                    
                    for future in concurrent.futures.as_completed(batch_futures):
                        if main_window.proc_stop_flag.is_set():
                            main_window.proc_queue.put(("stopped", "Improvement stopped by user"))
                            return
                        batch_idx = batch_futures[future]
                        try:
                            valid_batch = future.result()
                            improved_turns.extend(valid_batch)
                        except Exception as e:
                            main_window.proc_queue.put(("log", f"Batch {batch_idx + 1} exception: {str(e)}"))

                if main_window.proc_stop_flag.is_set():
                    main_window.proc_queue.put(("stopped", "Improvement stopped by user"))
                    return
                
                # Save updated cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(improved_turns, f, indent=2, ensure_ascii=False)
                
                if main_window.proc_stop_flag.is_set():
                    main_window.proc_queue.put(("stopped", "Improvement stopped by user"))
                    return
                
                main_window.proc_queue.put(("log", f"Successfully rewrote {len(improved_turns)} human turns"))
                if improved_turns and original_human_turns:
                    main_window.proc_queue.put(("log", f"Sample before: {original_human_turns[0]['message'][:100]}..."))
                    main_window.proc_queue.put(("log", f"Sample after:  {improved_turns[0]['message'][:100]}..."))
                main_window.proc_queue.put(("success", f"Human cache improved! {len(improved_turns)} turns updated in {cache_file} (concurrency: {MAX_CONCURRENCY})"))

            except Exception as e:
                main_window.proc_queue.put(("error", f"Cache improvement failed: {str(e)}"))

        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        main_window.timer.start()

    except ImportError as e:
        main_window._show_error(f"Missing dependency: {str(e)}")
    except Exception as e:
        main_window._show_error(f"Setup error: {str(e)}")

