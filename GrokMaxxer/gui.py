import os
import re
import threading
import queue

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont, QTextCharFormat, QTextCursor, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
    QGroupBox,
    QSizePolicy,
    QScrollArea,
    QFrame,
)

# Ensure we can import as a package - add parent directory to path if needed
import sys
from pathlib import Path
_grokmaxxer_dir = Path(__file__).parent
_parent_dir = _grokmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from GrokMaxxer.model_config import (
    APP_TITLE,
    ICON_FILE,
    DEFAULT_MODEL,
    load_config,
    save_config,
)
from GrokMaxxer.worker import worker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)

        self.setMinimumSize(1280, 720)

        if os.path.exists(ICON_FILE):
            try:
                self.setWindowIcon(QIcon(ICON_FILE))
            except Exception:
                pass

        self._setup_style()

        self.queue = None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.check_queue)

        self._build_ui()
        self._load_initial_config()

    def _setup_style(self):

        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
                background-color: #000000;
                color: #F9FAFB;
                font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
                font-size: 12pt;
            }
            QLabel {
                color: #E5E7EB;
            }
            QGroupBox {
                border: 1px solid #111827;
                border-radius: 8px;
                margin-top: 18px;
                padding: 10px;
                background-color: #050505;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #9CA3AF;
                font-weight: 600;
                font-size: 16pt;
            }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #050505;
                color: #F9FAFB;
                border: 1px solid #1F2937;
                border-radius: 4px;
                padding: 4px 6px;
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QLineEdit::placeholder {
                color: #6B7280;
            }
            QPlainTextEdit {
                background-color: #020202;
                color: #D1D5DB;
                border: 1px solid #1F2937;
                border-radius: 8px;
                font-family: Consolas, "Fira Code", monospace;
                font-size: 12px;
                padding: 6px;
            }
            QPushButton {
                background-color: #020617;
                color: #F9FAFB;
                border: 1px solid #1F2937;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #111827;
            }
            QPushButton:pressed {
                background-color: #030712;
            }
            QPushButton:disabled {
                color: #6B7280;
                border-color: #111827;
                background-color: #020202;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
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
            QComboBox QAbstractItemView {
                background-color: #050505;
                border: 1px solid #1F2937;
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        central.setLayout(root_layout)

        header_row = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)

        subtitle_label = QLabel("Simplified ShareGPT-style JSONL processor")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 12pt;")

        title_container = QVBoxLayout()
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_row.addLayout(title_container)
        header_row.addStretch()

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_files)
        header_row.addWidget(self.process_button, alignment=Qt.AlignRight)

        root_layout.addLayout(header_row)

        main_split = QHBoxLayout()
        main_split.setSpacing(14)
        root_layout.addLayout(main_split, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_container = QWidget()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        left_container.setLayout(left_panel)
        left_scroll.setWidget(left_container)

        main_split.addWidget(left_scroll, stretch=3)

        right_container = QWidget()
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        right_container.setLayout(right_panel)

        main_split.addWidget(right_container, stretch=4)

        # 1. FILES - Input/Output only
        files_group = QGroupBox("📁 Files")
        files_layout = QFormLayout()
        files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        files_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        files_layout.setHorizontalSpacing(10)
        files_layout.setVerticalSpacing(6)
        files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        files_group.setLayout(files_layout)

        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Path to input JSONL (optional for generation-only)")
        input_browse_btn = QPushButton("Browse")
        input_browse_btn.setFixedWidth(80)
        input_browse_btn.clicked.connect(self.browse_input)
        input_row.addWidget(self.input_edit)
        input_row.addWidget(input_browse_btn)
        files_layout.addRow(QLabel("Input JSONL:"), self._wrap_row(input_row))

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Path to output JSONL (required)")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self.browse_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(output_browse_btn)
        files_layout.addRow(QLabel("Output JSONL:"), self._wrap_row(output_row))
        left_panel.addWidget(files_group)

        # 2. PROCESSING - Range & transformation controls
        processing_group = QGroupBox("⚙️ Processing Mode")
        processing_layout = QFormLayout()
        processing_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        processing_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        processing_layout.setHorizontalSpacing(10)
        processing_layout.setVerticalSpacing(6)
        processing_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        processing_group.setLayout(processing_layout)
        processing_group.setToolTip("Controls how existing entries are processed")

        range_row = QHBoxLayout()
        self.start_line_edit = QLineEdit()
        self.start_line_edit.setPlaceholderText("from")
        self.start_line_edit.setMaximumWidth(80)
        self.end_line_edit = QLineEdit()
        self.end_line_edit.setPlaceholderText("to")
        self.end_line_edit.setMaximumWidth(80)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self.start_line_edit)
        range_row.addSpacing(8)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.end_line_edit)
        range_row.addStretch()
        processing_layout.addRow(QLabel("Process lines (1-based):"), self._wrap_row(range_row))

        self.rewrite_check = QCheckBox("Rewrite existing entries in range")
        self.rewrite_check.setChecked(True)
        self.rewrite_check.setToolTip("Improve quality of entries in selected line range")
        processing_layout.addRow("", self.rewrite_check)

        extra_row = QHBoxLayout()
        self.extra_pairs_spin = QSpinBox()
        self.extra_pairs_spin.setRange(0, 100)
        self.extra_pairs_spin.setValue(0)
        self.extra_pairs_spin.setMaximumWidth(80)
        self.extra_pairs_spin.setToolTip("Add extra human/GPT turn pairs to each entry")
        extra_row.addWidget(self.extra_pairs_spin)
        extra_row.addStretch()
        processing_layout.addRow(QLabel("Extra pairs per entry:"), self._wrap_row(extra_row))

        new_row = QHBoxLayout()
        self.num_new_edit = QLineEdit()
        self.num_new_edit.setPlaceholderText("0")
        self.num_new_edit.setMaximumWidth(80)
        self.num_new_edit.setToolTip("Generate this many completely new entries")
        new_row.addWidget(self.num_new_edit)
        new_row.addStretch()
        processing_layout.addRow(QLabel("Generate new entries:"), self._wrap_row(new_row))

        left_panel.addWidget(processing_group)

        # 3. CACHE - Human cache management
        cache_group = QGroupBox("🧠 Human Cache Management")
        cache_layout = QFormLayout()
        cache_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cache_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        cache_layout.setHorizontalSpacing(10)
        cache_layout.setVerticalSpacing(6)
        cache_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        cache_group.setLayout(cache_layout)
        cache_group.setToolTip("Manage human turns cache used for new entry generation")

        controls_row = QHBoxLayout()
        self.improve_cache_button = QPushButton("Improve Human Cache")
        self.improve_cache_button.clicked.connect(self.improve_human_cache)
        self.improve_cache_button.setToolTip("Rewrite existing human turns for better quality")
        
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 20)
        self.concurrency_spin.setValue(20)
        self.concurrency_spin.setMaximumWidth(60)
        self.concurrency_spin.setToolTip("Concurrent batches (safe: 1-20)")
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 64)
        self.batch_size_spin.setValue(16)
        self.batch_size_spin.setMaximumWidth(60)
        self.batch_size_spin.setToolTip("Batch size per API call (16 optimal for cache improvement)")
        
        controls_row.addWidget(self.improve_cache_button)
        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Concurrency:"))
        controls_row.addWidget(self.concurrency_spin)
        controls_row.addSpacing(20)
        controls_row.addWidget(QLabel("Batch:"))
        controls_row.addWidget(self.batch_size_spin)
        
        cache_layout.addRow("", self._wrap_row(controls_row))
        left_panel.addWidget(cache_group)

        prompt_group = QGroupBox("System / Character Prompt")
        prompt_layout = QFormLayout()
        prompt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prompt_layout.setHorizontalSpacing(10)
        prompt_layout.setVerticalSpacing(6)
        prompt_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        prompt_group.setLayout(prompt_layout)

        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setPlaceholderText("Character sheet or global system prompt (optional)")
        self.system_prompt_edit.setFixedHeight(100)
        prompt_layout.addRow(QLabel("System prompt:"), self.system_prompt_edit)

        self.reply_in_character_check = QCheckBox(
            "Reply in character & inject prompt as the first system message"
        )
        self.reply_in_character_check.setChecked(False)
        prompt_layout.addRow(QLabel("Character mode:"), self.reply_in_character_check)

        self.dynamic_names_check = QCheckBox("Dynamic Names Mode (auto-generate & cache names)")
        self.dynamic_names_check.setToolTip("Automatically generate and cache diverse names for new entries (scales to 10% of num_new, min 20). Improves variety.")
        self.dynamic_names_check.setChecked(False)
        prompt_layout.addRow(QLabel("Dynamic names:"), self.dynamic_names_check)

        left_panel.addWidget(prompt_group)

        api_group = QGroupBox("API & Model")
        api_layout = QFormLayout()
        api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        api_layout.setHorizontalSpacing(10)
        api_layout.setVerticalSpacing(6)
        api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        api_group.setLayout(api_layout)

        api_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-...")
        self.show_key_check = QCheckBox("Show")
        self.show_key_check.toggled.connect(self.toggle_api_visibility)
        api_row.addWidget(self.api_key_edit)
        api_row.addWidget(self.show_key_check)
        api_layout.addRow(QLabel("Grok API key:"), self._wrap_row(api_row))

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Enter model name (e.g., grok-beta)")
        self.model_edit.setText(DEFAULT_MODEL)

        api_layout.addRow(QLabel("Model:"), self.model_edit)

        left_panel.addWidget(api_group)
        left_panel.addStretch(1)

        progress_group = QGroupBox("Run Status")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(6)
        progress_group.setLayout(progress_layout)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setPlaceholderText("Logs will appear here...")
        progress_layout.addWidget(QLabel("Logs:"))
        progress_layout.addWidget(self.log_view, stretch=1)

        right_panel.addWidget(progress_group, stretch=1)

    def _wrap_row(self, layout):

        container = QWidget()
        container.setLayout(layout)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        container.setSizePolicy(sp)
        return container

    def _get_log_format(self, line):
        line_lower = line.lower()
        if any(word in line_lower for word in ['error', 'failed', 'invalid', 'fatal']):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(255, 100, 100))
            return fmt
        elif 'warning' in line_lower:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(255, 255, 0))
            return fmt
        elif any(word in line_lower for word in ['applied', 'success', 'generated', 'processed', 'preview']):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(100, 255, 100))
            return fmt
        elif any(word in line_lower for word in ['info', 'progress', 'processing', 'starting']):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(100, 150, 255))
            return fmt
        else:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(200, 200, 200))
            return fmt

    def _load_initial_config(self):
        cfg = load_config()
        api_key = cfg.get("api_key", "")
        input_path = cfg.get("last_input", "")
        output_path = cfg.get("last_output", "")
        system_prompt = cfg.get("system_prompt", "")
        saved_models_csv = cfg.get("models_csv", "")
        start_line = cfg.get("start_line")
        end_line = cfg.get("end_line")
        rewrite_existing = cfg.get("rewrite_existing", True)
        extra_pairs = cfg.get("extra_pairs", 0)
        num_new = cfg.get("num_new", 0)
        reply_in_character = cfg.get("reply_in_character", False)

        if api_key:
            self.api_key_edit.setText(api_key)
        if input_path:
            self.input_edit.setText(input_path)
        if output_path:
            self.output_edit.setText(output_path)
        if system_prompt:
            self.system_prompt_edit.setPlainText(system_prompt)
        if start_line:
            self.start_line_edit.setText(str(start_line))
        if end_line:
            self.end_line_edit.setText(str(end_line))

        self.rewrite_check.setChecked(bool(rewrite_existing))
        self.extra_pairs_spin.setValue(int(extra_pairs) if isinstance(extra_pairs, (int, float)) else 0)
        if num_new:
            self.num_new_edit.setText(str(int(num_new)))
        self.reply_in_character_check.setChecked(bool(reply_in_character))

        dynamic_names_mode = cfg.get("dynamic_names_mode", False)
        self.dynamic_names_check.setChecked(bool(dynamic_names_mode))

        model_name = saved_models_csv.split(',')[0].strip() if saved_models_csv else DEFAULT_MODEL
        self.model_edit.setText(model_name)

    def toggle_api_visibility(self, checked):
        if checked:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def browse_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select input JSONL file",
            "",
            "JSONL files (*.jsonl);;All files (*.*)",
        )
        if filename:
            self.input_edit.setText(filename)

    def browse_output(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select output JSONL file",
            "",
            "JSONL files (*.jsonl);;All files (*.*)",
        )
        if filename:
            self.output_edit.setText(filename)

    def _show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    def _show_info(self, msg):
        QMessageBox.information(self, "Success", msg)

    def strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', str(text))

    def _append_log(self, text):
        if not text:
            return
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_view.setTextCursor(cursor)
        ansi_stripped = self.strip_ansi(text)
        lines = ansi_stripped.split('\n')
        for line in lines:
            if line.strip():
                fmt = self._get_log_format(line)
                cursor.insertText(line + '\n', fmt)
            else:
                cursor.insertText('\n')
        self.log_view.ensureCursorVisible()

    def process_files(self):
        input_file = self.input_edit.text().strip()
        output_file = self.output_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        start_line_str = self.start_line_edit.text().strip()
        end_line_str = self.end_line_edit.text().strip()
        do_rewrite = self.rewrite_check.isChecked()
        reply_in_character = self.reply_in_character_check.isChecked()

        dynamic_names_mode = self.dynamic_names_check.isChecked()

        num_new_str = self.num_new_edit.text().strip()
        num_new = 0
        if num_new_str:
            try:
                num_new = int(num_new_str)
                if num_new < 0:
                    raise ValueError("Negative value")
            except ValueError:
                self._show_error("Number of new entries must be a non-negative integer.")
                return
        extra_pairs = int(self.extra_pairs_spin.value())

        if not output_file:
            self._show_error("Please select an output JSONL file.")

        if not api_key:
            self._show_error("Please enter your Grok API key.")
            return

        start_line = None
        if start_line_str:
            try:
                start_line = int(start_line_str)
                if start_line < 1:
                    raise ValueError
            except ValueError:
                self._show_error("Start line must be a positive integer.")
                return

        end_line = None
        if end_line_str:
            try:
                end_line = int(end_line_str)
                if end_line < 1:
                    raise ValueError
            except ValueError:
                self._show_error("End line must be a positive integer.")
                return

        if start_line is not None and end_line is not None and end_line < start_line:
            self._show_error("End line cannot be less than start line.")
            return

        model_name = self.model_edit.text().strip()
        if not model_name:
            model_name = DEFAULT_MODEL

        cfg = {
            "api_key": api_key,
            "last_input": input_file,
            "last_output": output_file,
            "system_prompt": system_prompt,
            "models_csv": model_name,
            "start_line": start_line,
            "end_line": end_line,
            "rewrite_existing": do_rewrite,
            "extra_pairs": extra_pairs,
            "num_new": num_new,
            "reply_in_character": reply_in_character,
            "dynamic_names_mode": dynamic_names_mode,
        }
        save_config(cfg)

        # Reset UI state for new run
        self.log_view.clear()
        self._append_log("=== New run started ===")
        self.setWindowTitle(f"{APP_TITLE} - Processing...")
        self.process_button.setEnabled(False)

        self.queue = queue.Queue()

        t = threading.Thread(
            target=worker,
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
                self.queue,
            ),
            daemon=True,
        )
        t.start()
        self.timer.start()

    def check_queue(self):
        if not self.queue:
            return

        try:
            while True:
                msg_type, msg = self.queue.get_nowait()
                if msg_type == "log":
                    print(str(msg))
                    self._append_log(msg)
                elif msg_type == "success":
                    self.setWindowTitle(f"{APP_TITLE} - Done")
                    self.process_button.setEnabled(True)
                    self.improve_cache_button.setEnabled(True)
                    self.timer.stop()
                    self._append_log(str(msg))
                    self._show_info(str(msg))
                elif msg_type == "error":
                    self.setWindowTitle(f"{APP_TITLE} - Error")
                    self.process_button.setEnabled(True)
                    self.improve_cache_button.setEnabled(True)
                    self.timer.stop()
                    self._append_log(str(msg))
                    self._show_error(str(msg))
        except queue.Empty:
            pass

        if not self.process_button.isEnabled():
            return
        else:
            self.timer.stop()

    def improve_human_cache(self):
        api_key = self.api_key_edit.text().strip()
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        model_name = self.model_edit.text().strip() or DEFAULT_MODEL

        if not api_key:
            self._show_error("Please enter your Grok API key.")
            return

        cache_file = "global_human_cache.json"
        try:
            import json
            import os
            from xai_sdk import Client
            from xai_sdk.chat import system
            from json_repair import repair_json

            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    human_turns = json.load(f)
                self._append_log(f"Loaded {len(human_turns)} human turns from {cache_file}")
            else:
                human_turns = []
                self._append_log(f"No cache file found at {cache_file}, starting empty")

            if not human_turns:
                self._show_info("Human cache is empty. Run a generation job first to populate it.")
                return

            # Reset UI state
            self.log_view.clear()
            self._append_log("=== Improving human cache ===")
            self.setWindowTitle(f"{APP_TITLE} - Improving cache...")
            self.process_button.setEnabled(False)
            self.improve_cache_button.setEnabled(False)

            self.queue = queue.Queue()

            def worker_thread():
                try:
                    import concurrent.futures
                    client = Client(api_key=api_key, timeout=300)
                    
                    BATCH_SIZE = self.batch_size_spin.value()
                    MAX_CONCURRENCY = self.concurrency_spin.value()
                    original_human_turns = human_turns.copy()
                    num_batches = (len(original_human_turns) + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    self.queue.put(("log", f"Processing {len(original_human_turns)} human turns in {num_batches} batches of {BATCH_SIZE} (concurrency: {MAX_CONCURRENCY})..."))
                    
                    def process_batch(batch_idx):
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
                        if system_prompt.strip() and self.reply_in_character_check.isChecked():
                            prompt = system_prompt.strip() + "\n\n" + base_prompt

                        try:
                            self.queue.put(("log", f"Sending batch {batch_idx + 1}/{num_batches} ({len(batch)} items)..."))
                            chat = client.chat.create(model=model_name, temperature=0.8)
                            chat.append(system(prompt))
                            response = chat.sample()
                        
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
                            
                            self.queue.put(("log", f"Batch {batch_idx + 1}/{num_batches}: got {len(batch_improved)} items, {skipped} skipped, using {len(valid_batch)} valid"))
                            return valid_batch
                        except Exception as e:
                            self.queue.put(("log", f"Batch {batch_idx + 1} failed: {str(e)}"))
                            # Fallback to originals
                            return batch

                    improved_turns = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
                        # Submit all batch futures
                        batch_futures = {
                            executor.submit(process_batch, i): i 
                            for i in range(num_batches)
                        }
                        
                        # Collect results in order
                        for future in concurrent.futures.as_completed(batch_futures):
                            batch_idx = batch_futures[future]
                            try:
                                valid_batch = future.result()
                                improved_turns.extend(valid_batch)
                            except Exception as e:
                                self.queue.put(("log", f"Batch {batch_idx + 1} exception: {str(e)}"))

                    # Save updated cache
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(improved_turns, f, indent=2, ensure_ascii=False)
                    
                    self.queue.put(("log", f"Successfully rewrote {len(improved_turns)} human turns"))
                    self.queue.put(("log", f"Sample before: {original_human_turns[0]['message'][:100]}..."))
                    self.queue.put(("log", f"Sample after:  {improved_turns[0]['message'][:100]}..."))
                    self.queue.put(("success", f"Human cache improved! {len(improved_turns)} turns updated in {cache_file} (concurrency: {MAX_CONCURRENCY})"))

                except Exception as e:
                    self.queue.put(("error", f"Cache improvement failed: {str(e)}"))

            t = threading.Thread(target=worker_thread, daemon=True)
            t.start()
            self.timer.start()

        except ImportError as e:
            self._show_error(f"Missing dependency: {str(e)}")
        except Exception as e:
            self._show_error(f"Setup error: {str(e)}")


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
