import os
import re
import threading
import queue
import json
import random
import time
import pathlib
import requests
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QFont, QTextCharFormat, QTextCursor, QColor, QImage, QPixmap
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
    QDoubleSpinBox,
    QTabWidget,
)

# Ensure we can import as a package - add parent directory to path if needed
import sys
from pathlib import Path
_synthmaxxer_dir = Path(__file__).parent
_parent_dir = _synthmaxxer_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from App.SynthMaxxer.worker import worker
from App.SynthMaxxer.processing_worker import processing_worker


APP_TITLE = "SynthMaxxer"
ICON_FILE = str(Path(__file__).parent.parent / "Assets" / "icon.ico")
CONFIG_FILE = str(Path(__file__).parent / "synthmaxxer_config.json")
GLOBAL_HUMAN_CACHE_FILE = str(Path(__file__).parent / "global_human_cache.json")


class ModelFetcher(QObject):
    """Helper class to emit signals from background thread"""
    models_ready = pyqtSignal(list)

class MultimodalModelFetcher(QObject):
    """Helper class to emit signals from background thread for multimodal tab"""
    models_ready = pyqtSignal(list)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1280, 720)
        
        # Create model fetcher for thread-safe signals
        self.model_fetcher = ModelFetcher()
        self.model_fetcher.models_ready.connect(self._on_models_ready)
        
        # Create multimodal model fetcher
        self.mm_model_fetcher = MultimodalModelFetcher()
        self.mm_model_fetcher.models_ready.connect(self._on_mm_models_ready)

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
        
        self.stop_flag = threading.Event()
        self.worker_thread = None

        self._build_ui()
        self._load_initial_config()
        # Don't auto-fetch models on startup - let user click refresh when ready
    
    def closeEvent(self, event):
        """Save config when window is closed"""
        try:
            self._save_config()
        except Exception as e:
            print(f"Error saving config on close: {e}")
        event.accept()

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
            QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox {
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
            QTabWidget::pane {
                border: 1px solid #1F2937;
                background-color: #000000;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #050505;
                color: #9CA3AF;
                border: 1px solid #1F2937;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #000000;
                color: #F9FAFB;
                border-color: #2563EB;
            }
            QTabBar::tab:hover {
                background-color: #111827;
                color: #E5E7EB;
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

        subtitle_label = QLabel("Synthetic Data Generator & Processor")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 12pt;")

        title_container = QVBoxLayout()
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        header_row.addLayout(title_container)
        header_row.addStretch()

        root_layout.addLayout(header_row)

        # Create tab widget
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)

        # Generation tab
        generation_tab = QWidget()
        generation_layout = QVBoxLayout()
        generation_layout.setContentsMargins(0, 0, 0, 0)
        generation_tab.setLayout(generation_layout)
        
        gen_header = QHBoxLayout()
        self.start_button = QPushButton("Start Generation")
        self.start_button.clicked.connect(self.start_generation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)
        gen_header.addStretch()
        gen_header.addWidget(self.start_button)
        gen_header.addWidget(self.stop_button)
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

        main_split.addWidget(left_scroll, stretch=3)

        right_container = QWidget()
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        right_container.setLayout(right_panel)

        main_split.addWidget(right_container, stretch=4)

        # 1. API Configuration
        api_group = QGroupBox("🔑 API Configuration")
        api_layout = QFormLayout()
        api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        api_layout.setHorizontalSpacing(10)
        api_layout.setVerticalSpacing(6)
        api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        api_group.setLayout(api_layout)

        api_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Your API key")
        self.show_key_check = QCheckBox("Show")
        self.show_key_check.toggled.connect(self.toggle_api_visibility)
        api_row.addWidget(self.api_key_edit)
        api_row.addWidget(self.show_key_check)
        api_layout.addRow(QLabel("API Key:"), self._wrap_row(api_row))

        self.endpoint_edit = QLineEdit()
        self.endpoint_edit.setPlaceholderText("https://api.example.com/v1/messages (or /v1/chat/completions or /v1/completions)")
        api_layout.addRow(QLabel("API Endpoint:"), self.endpoint_edit)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)  # Allow custom model names
        self.model_combo.setPlaceholderText("Select or enter model name")
        self.model_combo.addItem("(Click Refresh to load models)")
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.setFixedWidth(80)
        self.refresh_models_button.setToolTip("Refresh available models from API")
        self.refresh_models_button.setEnabled(True)  # Always enabled by default
        self.refresh_models_button.clicked.connect(self._refresh_models)
        print("DEBUG: Refresh button created and connected")
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.refresh_models_button)
        api_layout.addRow(QLabel("Model:"), self._wrap_row(model_row))

        self.api_type_combo = QComboBox()
        self.api_type_combo.addItems([
            "Anthropic Claude",
            "OpenAI Official",
            "OpenAI Chat Completions",
            "OpenAI Text Completions",
            "Grok (xAI)",
            "Gemini (Google)",
            "OpenRouter",
            "DeepSeek"
        ])
        self.api_type_combo.setToolTip("Select the API format to use")
        api_layout.addRow(QLabel("API Type:"), self.api_type_combo)
        
        # Update endpoint placeholder and API key when API type changes (after combo is created)
        self.api_type_combo.currentTextChanged.connect(self._update_endpoint_placeholder)
        self.api_type_combo.currentTextChanged.connect(self._update_api_key_for_type)
        # Don't auto-fetch models on API type change - let user click refresh

        left_panel.addWidget(api_group)

        # 2. Output Configuration
        output_group = QGroupBox("📁 Output Configuration")
        output_layout = QFormLayout()
        output_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_layout.setHorizontalSpacing(10)
        output_layout.setVerticalSpacing(6)
        output_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        output_group.setLayout(output_layout)

        output_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("outputs")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_row.addWidget(self.output_dir_edit)
        output_dir_row.addWidget(output_browse_btn)
        output_layout.addRow(QLabel("Output Directory:"), self._wrap_row(output_dir_row))

        left_panel.addWidget(output_group)

        # 3. Generation Settings
        generation_group = QGroupBox("⚙️ Generation Settings")
        generation_layout = QFormLayout()
        generation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        generation_layout.setHorizontalSpacing(10)
        generation_layout.setVerticalSpacing(6)
        generation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        generation_group.setLayout(generation_layout)

        delay_row = QHBoxLayout()
        self.min_delay_spin = QDoubleSpinBox()
        self.min_delay_spin.setRange(0.0, 10.0)
        self.min_delay_spin.setValue(0.1)
        self.min_delay_spin.setSingleStep(0.1)
        self.min_delay_spin.setDecimals(2)
        self.min_delay_spin.setMaximumWidth(80)
        self.max_delay_spin = QDoubleSpinBox()
        self.max_delay_spin.setRange(0.0, 10.0)
        self.max_delay_spin.setValue(0.5)
        self.max_delay_spin.setSingleStep(0.1)
        self.max_delay_spin.setDecimals(2)
        self.max_delay_spin.setMaximumWidth(80)
        delay_row.addWidget(QLabel("Min:"))
        delay_row.addWidget(self.min_delay_spin)
        delay_row.addSpacing(8)
        delay_row.addWidget(QLabel("Max:"))
        delay_row.addWidget(self.max_delay_spin)
        delay_row.addStretch()
        generation_layout.addRow(QLabel("Delay (seconds):"), self._wrap_row(delay_row))

        self.stop_percentage_spin = QDoubleSpinBox()
        self.stop_percentage_spin.setRange(0.0, 1.0)
        self.stop_percentage_spin.setValue(0.05)
        self.stop_percentage_spin.setSingleStep(0.01)
        self.stop_percentage_spin.setDecimals(2)
        self.stop_percentage_spin.setMaximumWidth(80)
        self.stop_percentage_spin.setToolTip("Probability of stopping after minimum turns")
        stop_row = QHBoxLayout()
        stop_row.addWidget(self.stop_percentage_spin)
        stop_row.addStretch()
        generation_layout.addRow(QLabel("Stop Probability:"), self._wrap_row(stop_row))

        self.min_turns_spin = QSpinBox()
        self.min_turns_spin.setRange(0, 100)
        self.min_turns_spin.setValue(1)
        self.min_turns_spin.setMaximumWidth(80)
        self.min_turns_spin.setToolTip("Minimum number of assistant turns before stopping")
        min_turns_row = QHBoxLayout()
        min_turns_row.addWidget(self.min_turns_spin)
        min_turns_row.addStretch()
        generation_layout.addRow(QLabel("Min Turns:"), self._wrap_row(min_turns_row))

        left_panel.addWidget(generation_group)

        # 4. Conversation Configuration
        conversation_group = QGroupBox("💬 Conversation Configuration")
        conversation_layout = QFormLayout()
        conversation_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conversation_layout.setHorizontalSpacing(10)
        conversation_layout.setVerticalSpacing(6)
        conversation_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        conversation_group.setLayout(conversation_layout)

        self.system_message_edit = QPlainTextEdit()
        self.system_message_edit.setPlaceholderText("System message for the conversation")
        self.system_message_edit.setFixedHeight(60)
        conversation_layout.addRow(QLabel("System Message:"), self.system_message_edit)

        self.user_first_message_edit = QPlainTextEdit()
        self.user_first_message_edit.setPlaceholderText("First user message")
        self.user_first_message_edit.setFixedHeight(60)
        conversation_layout.addRow(QLabel("User First Message:"), self.user_first_message_edit)

        self.assistant_first_message_edit = QPlainTextEdit()
        self.assistant_first_message_edit.setPlaceholderText("First assistant message")
        self.assistant_first_message_edit.setFixedHeight(60)
        conversation_layout.addRow(QLabel("Assistant First Message:"), self.assistant_first_message_edit)

        # Tags
        tags_row1 = QHBoxLayout()
        self.user_start_tag_edit = QLineEdit()
        self.user_start_tag_edit.setPlaceholderText("<human_turn>")
        self.user_start_tag_edit.setMaximumWidth(120)
        self.user_end_tag_edit = QLineEdit()
        self.user_end_tag_edit.setPlaceholderText("</human_turn>")
        self.user_end_tag_edit.setMaximumWidth(120)
        tags_row1.addWidget(QLabel("User Start:"))
        tags_row1.addWidget(self.user_start_tag_edit)
        tags_row1.addSpacing(8)
        tags_row1.addWidget(QLabel("User End:"))
        tags_row1.addWidget(self.user_end_tag_edit)
        tags_row1.addStretch()
        conversation_layout.addRow(QLabel("User Tags:"), self._wrap_row(tags_row1))

        tags_row2 = QHBoxLayout()
        self.assistant_start_tag_edit = QLineEdit()
        self.assistant_start_tag_edit.setPlaceholderText("<claude_turn>")
        self.assistant_start_tag_edit.setMaximumWidth(120)
        self.assistant_end_tag_edit = QLineEdit()
        self.assistant_end_tag_edit.setPlaceholderText("</claude_turn>")
        self.assistant_end_tag_edit.setMaximumWidth(120)
        tags_row2.addWidget(QLabel("Assistant Start:"))
        tags_row2.addWidget(self.assistant_start_tag_edit)
        tags_row2.addSpacing(8)
        tags_row2.addWidget(QLabel("Assistant End:"))
        tags_row2.addWidget(self.assistant_end_tag_edit)
        tags_row2.addStretch()
        conversation_layout.addRow(QLabel("Assistant Tags:"), self._wrap_row(tags_row2))

        self.is_instruct_check = QCheckBox("Instruct Mode (min_turns=0, start_index=0, stopPercentage=0.25)")
        self.is_instruct_check.setChecked(False)
        conversation_layout.addRow("", self.is_instruct_check)

        left_panel.addWidget(conversation_group)

        # 5. Filter Settings
        filter_group = QGroupBox("🚫 Filter Settings")
        filter_layout = QFormLayout()
        filter_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        filter_layout.setHorizontalSpacing(10)
        filter_layout.setVerticalSpacing(6)
        filter_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        filter_group.setLayout(filter_layout)

        self.refusal_phrases_edit = QPlainTextEdit()
        self.refusal_phrases_edit.setPlaceholderText("One phrase per line\nExample:\nUpon further reflection\nI can't engage")
        self.refusal_phrases_edit.setFixedHeight(60)
        filter_layout.addRow(QLabel("Refusal Phrases:"), self.refusal_phrases_edit)

        self.force_retry_phrases_edit = QPlainTextEdit()
        self.force_retry_phrases_edit.setPlaceholderText("One phrase per line\nExample:\nshivers down")
        self.force_retry_phrases_edit.setFixedHeight(60)
        filter_layout.addRow(QLabel("Force Retry Phrases:"), self.force_retry_phrases_edit)

        left_panel.addWidget(filter_group)
        left_panel.addStretch(1)

        # Right panel - Logs
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
        
        # Add generation tab to tabs widget
        self.tabs.addTab(generation_tab, "🔄 Generation")
        
        # Processing tab
        processing_tab = QWidget()
        processing_layout = QVBoxLayout()
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_tab.setLayout(processing_layout)
        
        proc_header = QHBoxLayout()
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_files)
        proc_header.addStretch()
        proc_header.addWidget(self.process_button)
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
        
        # Processing UI elements (based on GrokMaxxer)
        self._build_processing_ui(proc_left_panel, proc_right_panel)
        
        # Add processing tab to tabs widget
        self.tabs.addTab(processing_tab, "⚙️ Processing")
        
        # Multimodal tab
        multimodal_tab = QWidget()
        multimodal_layout = QVBoxLayout()
        multimodal_layout.setContentsMargins(0, 0, 0, 0)
        multimodal_tab.setLayout(multimodal_layout)
        
        mm_header = QHBoxLayout()
        self.mm_start_button = QPushButton("Start Captioning")
        self.mm_start_button.clicked.connect(self.start_image_captioning)
        self.mm_stop_button = QPushButton("Stop")
        self.mm_stop_button.clicked.connect(self.stop_image_captioning)
        self.mm_stop_button.setEnabled(False)
        self.civitai_start_button = QPushButton("Start Civitai Download")
        self.civitai_start_button.clicked.connect(self.start_civitai_download)
        self.civitai_stop_button = QPushButton("Stop Civitai")
        self.civitai_stop_button.clicked.connect(self.stop_civitai_download)
        self.civitai_stop_button.setEnabled(False)
        mm_header.addStretch()
        mm_header.addWidget(self.mm_start_button)
        mm_header.addWidget(self.mm_stop_button)
        mm_header.addWidget(self.civitai_start_button)
        mm_header.addWidget(self.civitai_stop_button)
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
        self._build_multimodal_ui(mm_left_panel, mm_right_panel)
        
        # Add multimodal tab to tabs widget
        self.tabs.addTab(multimodal_tab, "🖼️ Multimodal")
        
        # Processing log view is created in _build_processing_ui

    def _build_processing_ui(self, left_panel, right_panel):
        """Build the processing UI (from GrokMaxxer)"""
        # Files group
        files_group = QGroupBox("📁 Files")
        files_layout = QFormLayout()
        files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        files_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        files_layout.setHorizontalSpacing(10)
        files_layout.setVerticalSpacing(6)
        files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        files_group.setLayout(files_layout)

        input_row = QHBoxLayout()
        self.proc_input_edit = QLineEdit()
        self.proc_input_edit.setPlaceholderText("Path to input JSONL (optional for generation-only)")
        input_browse_btn = QPushButton("Browse")
        input_browse_btn.setFixedWidth(80)
        input_browse_btn.clicked.connect(self.browse_proc_input)
        input_row.addWidget(self.proc_input_edit)
        input_row.addWidget(input_browse_btn)
        files_layout.addRow(QLabel("Input JSONL:"), self._wrap_row(input_row))

        output_row = QHBoxLayout()
        self.proc_output_edit = QLineEdit()
        # Set default to outputs folder
        repo_root = os.path.dirname(os.path.dirname(__file__))
        outputs_dir = os.path.join(repo_root, "outputs")
        default_output = os.path.join(outputs_dir, "processed_output.jsonl")
        self.proc_output_edit.setPlaceholderText(f"Leave empty to auto-generate in outputs folder")
        self.proc_output_edit.setText(default_output)
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self.browse_proc_output)
        output_row.addWidget(self.proc_output_edit)
        output_row.addWidget(output_browse_btn)
        files_layout.addRow(QLabel("Output JSONL:"), self._wrap_row(output_row))
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
        self.proc_start_line_edit = QLineEdit()
        self.proc_start_line_edit.setPlaceholderText("from")
        self.proc_start_line_edit.setMaximumWidth(80)
        self.proc_end_line_edit = QLineEdit()
        self.proc_end_line_edit.setPlaceholderText("to")
        self.proc_end_line_edit.setMaximumWidth(80)
        range_row.addWidget(QLabel("from"))
        range_row.addWidget(self.proc_start_line_edit)
        range_row.addSpacing(8)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.proc_end_line_edit)
        range_row.addStretch()
        processing_layout.addRow(QLabel("Process lines (1-based):"), self._wrap_row(range_row))

        self.proc_rewrite_check = QCheckBox("Rewrite existing entries in range")
        self.proc_rewrite_check.setChecked(True)
        processing_layout.addRow("", self.proc_rewrite_check)

        extra_row = QHBoxLayout()
        self.proc_extra_pairs_spin = QSpinBox()
        self.proc_extra_pairs_spin.setRange(0, 100)
        self.proc_extra_pairs_spin.setValue(0)
        self.proc_extra_pairs_spin.setMaximumWidth(80)
        extra_row.addWidget(self.proc_extra_pairs_spin)
        extra_row.addStretch()
        processing_layout.addRow(QLabel("Extra pairs per entry:"), self._wrap_row(extra_row))

        new_row = QHBoxLayout()
        self.proc_num_new_edit = QLineEdit()
        self.proc_num_new_edit.setPlaceholderText("0")
        self.proc_num_new_edit.setMaximumWidth(80)
        new_row.addWidget(self.proc_num_new_edit)
        new_row.addStretch()
        processing_layout.addRow(QLabel("Generate new entries:"), self._wrap_row(new_row))

        left_panel.addWidget(processing_group)

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
        self.generate_cache_button = QPushButton("Generate Human Cache")
        self.generate_cache_button.clicked.connect(self.generate_human_cache)
        self.generate_cache_button.setToolTip("Generate new human turns for the cache")
        
        self.improve_cache_button = QPushButton("Improve Human Cache")
        self.improve_cache_button.clicked.connect(self.improve_human_cache)
        self.improve_cache_button.setToolTip("Rewrite existing human turns for better quality")
        
        self.proc_concurrency_spin = QSpinBox()
        self.proc_concurrency_spin.setRange(1, 20)
        self.proc_concurrency_spin.setValue(20)
        self.proc_concurrency_spin.setMaximumWidth(60)
        self.proc_concurrency_spin.setToolTip("Concurrent batches (safe: 1-20)")
        
        self.proc_batch_size_spin = QSpinBox()
        self.proc_batch_size_spin.setRange(16, 64)
        self.proc_batch_size_spin.setValue(16)
        self.proc_batch_size_spin.setMaximumWidth(60)
        self.proc_batch_size_spin.setToolTip("Batch size per API call (16 optimal for cache improvement)")
        
        controls_row.addWidget(self.generate_cache_button)
        controls_row.addSpacing(10)
        controls_row.addWidget(self.improve_cache_button)
        controls_row.addSpacing(10)
        controls_row.addWidget(QLabel("Concurrency:"))
        controls_row.addWidget(self.proc_concurrency_spin)
        controls_row.addSpacing(20)
        controls_row.addWidget(QLabel("Batch:"))
        controls_row.addWidget(self.proc_batch_size_spin)
        controls_row.addStretch()
        
        cache_layout.addRow("", self._wrap_row(controls_row))
        left_panel.addWidget(cache_group)

        # Prompt group
        prompt_group = QGroupBox("System / Character Prompt")
        prompt_layout = QFormLayout()
        prompt_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prompt_layout.setHorizontalSpacing(10)
        prompt_layout.setVerticalSpacing(6)
        prompt_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        prompt_group.setLayout(prompt_layout)

        self.proc_system_prompt_edit = QPlainTextEdit()
        self.proc_system_prompt_edit.setPlaceholderText("Character sheet or global system prompt (optional)")
        self.proc_system_prompt_edit.setFixedHeight(100)
        prompt_layout.addRow(QLabel("System prompt:"), self.proc_system_prompt_edit)

        self.proc_reply_in_character_check = QCheckBox(
            "Reply in character & inject prompt as the first system message"
        )
        self.proc_reply_in_character_check.setChecked(False)
        prompt_layout.addRow(QLabel("Character mode:"), self.proc_reply_in_character_check)

        self.proc_dynamic_names_check = QCheckBox("Dynamic Names Mode (auto-generate & cache names)")
        self.proc_dynamic_names_check.setChecked(False)
        prompt_layout.addRow(QLabel("Dynamic names:"), self.proc_dynamic_names_check)

        left_panel.addWidget(prompt_group)

        # API group for processing
        proc_api_group = QGroupBox("API & Model")
        proc_api_layout = QFormLayout()
        proc_api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        proc_api_layout.setHorizontalSpacing(10)
        proc_api_layout.setVerticalSpacing(6)
        proc_api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        proc_api_group.setLayout(proc_api_layout)

        proc_api_row = QHBoxLayout()
        self.proc_api_key_edit = QLineEdit()
        self.proc_api_key_edit.setEchoMode(QLineEdit.Password)
        self.proc_api_key_edit.setPlaceholderText("sk-...")
        self.proc_show_key_check = QCheckBox("Show")
        self.proc_show_key_check.toggled.connect(self.toggle_proc_api_visibility)
        proc_api_row.addWidget(self.proc_api_key_edit)
        proc_api_row.addWidget(self.proc_show_key_check)
        proc_api_layout.addRow(QLabel("Grok API key:"), self._wrap_row(proc_api_row))

        self.proc_model_edit = QLineEdit()
        self.proc_model_edit.setPlaceholderText("Enter model name (e.g., grok-beta)")
        self.proc_model_edit.setText("grok-4.1-fast-non-reasoning")

        proc_api_layout.addRow(QLabel("Model:"), self.proc_model_edit)

        left_panel.addWidget(proc_api_group)
        left_panel.addStretch(1)

        # Right panel - Logs for processing
        proc_progress_group = QGroupBox("Run Status")
        proc_progress_layout = QVBoxLayout()
        proc_progress_layout.setSpacing(6)
        proc_progress_group.setLayout(proc_progress_layout)

        self.proc_log_view = QPlainTextEdit()
        self.proc_log_view.setReadOnly(True)
        self.proc_log_view.setMaximumBlockCount(1000)
        self.proc_log_view.setPlaceholderText("Logs will appear here...")
        proc_progress_layout.addWidget(QLabel("Logs:"))
        proc_progress_layout.addWidget(self.proc_log_view, stretch=1)

        right_panel.addWidget(proc_progress_group, stretch=1)

    def _build_multimodal_ui(self, left_panel, right_panel):
        """Build the multimodal image captioning UI"""
        # Files group
        mm_files_group = QGroupBox("📁 Image Input")
        mm_files_layout = QFormLayout()
        mm_files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        mm_files_layout.setHorizontalSpacing(10)
        mm_files_layout.setVerticalSpacing(6)
        mm_files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        mm_files_group.setLayout(mm_files_layout)

        image_dir_row = QHBoxLayout()
        self.mm_image_dir_edit = QLineEdit()
        self.mm_image_dir_edit.setPlaceholderText("Path to folder containing images")
        image_dir_browse_btn = QPushButton("Browse")
        image_dir_browse_btn.setFixedWidth(80)
        image_dir_browse_btn.clicked.connect(self.browse_mm_image_dir)
        image_dir_row.addWidget(self.mm_image_dir_edit)
        image_dir_row.addWidget(image_dir_browse_btn)
        mm_files_layout.addRow(QLabel("Image Folder:"), self._wrap_row(image_dir_row))

        output_row = QHBoxLayout()
        self.mm_output_edit = QLineEdit()
        # Get repo root (go up from App/SynthMaxxer to repo root)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(repo_root, "outputs")
        default_output = os.path.join(outputs_dir, "image_captions.parquet")
        self.mm_output_edit.setPlaceholderText(f"Leave empty to auto-generate in outputs folder (will be .parquet)")
        self.mm_output_edit.setText(default_output)
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self.browse_mm_output)
        output_row.addWidget(self.mm_output_edit)
        output_row.addWidget(output_browse_btn)
        mm_files_layout.addRow(QLabel("Output JSONL:"), self._wrap_row(output_row))
        left_panel.addWidget(mm_files_group)

        # API Configuration
        mm_api_group = QGroupBox("🔑 API Configuration")
        mm_api_layout = QFormLayout()
        mm_api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        mm_api_layout.setHorizontalSpacing(10)
        mm_api_layout.setVerticalSpacing(6)
        mm_api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        mm_api_group.setLayout(mm_api_layout)

        mm_api_row = QHBoxLayout()
        self.mm_api_key_edit = QLineEdit()
        self.mm_api_key_edit.setEchoMode(QLineEdit.Password)
        self.mm_api_key_edit.setPlaceholderText("Your API key")
        self.mm_show_key_check = QCheckBox("Show")
        self.mm_show_key_check.toggled.connect(self.toggle_mm_api_visibility)
        mm_api_row.addWidget(self.mm_api_key_edit)
        mm_api_row.addWidget(self.mm_show_key_check)
        mm_api_layout.addRow(QLabel("API Key:"), self._wrap_row(mm_api_row))

        self.mm_endpoint_edit = QLineEdit()
        self.mm_endpoint_edit.setPlaceholderText("https://api.openai.com/v1/chat/completions")
        mm_api_layout.addRow(QLabel("API Endpoint:"), self.mm_endpoint_edit)

        mm_model_row = QHBoxLayout()
        self.mm_model_combo = QComboBox()
        self.mm_model_combo.setEditable(True)
        self.mm_model_combo.setPlaceholderText("Select or enter model name")
        self.mm_model_combo.addItem("(Click Refresh to load models)")
        self.mm_refresh_models_button = QPushButton("Refresh")
        self.mm_refresh_models_button.setFixedWidth(80)
        self.mm_refresh_models_button.setToolTip("Refresh available models from API")
        self.mm_refresh_models_button.setEnabled(True)
        self.mm_refresh_models_button.clicked.connect(self._refresh_mm_models)
        mm_model_row.addWidget(self.mm_model_combo)
        mm_model_row.addWidget(self.mm_refresh_models_button)
        mm_api_layout.addRow(QLabel("Model:"), self._wrap_row(mm_model_row))

        self.mm_api_type_combo = QComboBox()
        self.mm_api_type_combo.addItems([
            "OpenAI Vision",
            "Anthropic Claude",
            "Grok (xAI)",
            "OpenRouter"
        ])
        self.mm_api_type_combo.setToolTip("Select the API format to use")
        self.mm_api_type_combo.currentTextChanged.connect(self._update_mm_endpoint_placeholder)
        self.mm_api_type_combo.currentTextChanged.connect(self._update_mm_api_key_for_type)
        mm_api_layout.addRow(QLabel("API Type:"), self.mm_api_type_combo)
        left_panel.addWidget(mm_api_group)

        # HuggingFace Dataset Input
        hf_dataset_group = QGroupBox("🤗 HuggingFace Dataset (Optional)")
        hf_dataset_layout = QFormLayout()
        hf_dataset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hf_dataset_layout.setHorizontalSpacing(10)
        hf_dataset_layout.setVerticalSpacing(6)
        hf_dataset_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        hf_dataset_group.setLayout(hf_dataset_layout)

        self.mm_hf_dataset_edit = QLineEdit()
        self.mm_hf_dataset_edit.setPlaceholderText("e.g., dataset_name or org/dataset_name (leave empty to use image folder)")
        hf_dataset_layout.addRow(QLabel("HF Dataset:"), self.mm_hf_dataset_edit)

        hf_token_row = QHBoxLayout()
        self.mm_hf_token_edit = QLineEdit()
        self.mm_hf_token_edit.setEchoMode(QLineEdit.Password)
        self.mm_hf_token_edit.setPlaceholderText("hf_... (optional, for private/gated datasets)")
        self.mm_hf_show_token_check = QCheckBox("Show")
        self.mm_hf_show_token_check.toggled.connect(self.toggle_mm_hf_token_visibility)
        hf_token_row.addWidget(self.mm_hf_token_edit)
        hf_token_row.addWidget(self.mm_hf_show_token_check)
        hf_dataset_layout.addRow(QLabel("HF Token:"), self._wrap_row(hf_token_row))

        self.mm_use_hf_dataset_check = QCheckBox("Use HuggingFace dataset instead of image folder")
        self.mm_use_hf_dataset_check.setChecked(False)
        self.mm_use_hf_dataset_check.toggled.connect(self._toggle_hf_dataset_mode)
        # Initialize state
        self.mm_hf_dataset_edit.setEnabled(False)
        self.mm_hf_token_edit.setEnabled(False)
        hf_dataset_layout.addRow("", self.mm_use_hf_dataset_check)
        left_panel.addWidget(hf_dataset_group)

        # Caption Settings
        mm_caption_group = QGroupBox("📝 Caption Settings")
        mm_caption_layout = QFormLayout()
        mm_caption_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        mm_caption_layout.setHorizontalSpacing(10)
        mm_caption_layout.setVerticalSpacing(6)
        mm_caption_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        mm_caption_group.setLayout(mm_caption_layout)

        self.mm_caption_prompt_edit = QPlainTextEdit()
        self.mm_caption_prompt_edit.setPlaceholderText("Describe what you see in this image in detail. Include all important elements, objects, people, text, colors, composition, and context.")
        self.mm_caption_prompt_edit.setFixedHeight(80)
        mm_caption_layout.addRow(QLabel("Caption Prompt:"), self.mm_caption_prompt_edit)

        self.mm_max_tokens_spin = QSpinBox()
        self.mm_max_tokens_spin.setRange(50, 4000)
        self.mm_max_tokens_spin.setValue(500)
        self.mm_max_tokens_spin.setMaximumWidth(100)
        self.mm_max_tokens_spin.setToolTip("Maximum tokens for caption generation")
        max_tokens_row = QHBoxLayout()
        max_tokens_row.addWidget(self.mm_max_tokens_spin)
        max_tokens_row.addStretch()
        mm_caption_layout.addRow(QLabel("Max Tokens:"), self._wrap_row(max_tokens_row))

        self.mm_temperature_spin = QDoubleSpinBox()
        self.mm_temperature_spin.setRange(0.0, 2.0)
        self.mm_temperature_spin.setValue(0.7)
        self.mm_temperature_spin.setSingleStep(0.1)
        self.mm_temperature_spin.setDecimals(1)
        self.mm_temperature_spin.setMaximumWidth(100)
        self.mm_temperature_spin.setToolTip("Temperature for caption generation")
        temp_row = QHBoxLayout()
        temp_row.addWidget(self.mm_temperature_spin)
        temp_row.addStretch()
        mm_caption_layout.addRow(QLabel("Temperature:"), self._wrap_row(temp_row))

        self.mm_batch_size_spin = QSpinBox()
        self.mm_batch_size_spin.setRange(1, 20)
        self.mm_batch_size_spin.setValue(1)
        self.mm_batch_size_spin.setMaximumWidth(100)
        self.mm_batch_size_spin.setToolTip("Number of images to process in parallel (1 recommended for most APIs)")
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.mm_batch_size_spin)
        batch_row.addStretch()
        mm_caption_layout.addRow(QLabel("Batch Size:"), self._wrap_row(batch_row))

        self.mm_max_captions_spin = QSpinBox()
        self.mm_max_captions_spin.setRange(0, 100000)
        self.mm_max_captions_spin.setValue(0)
        self.mm_max_captions_spin.setSpecialValueText("Unlimited")
        self.mm_max_captions_spin.setMaximumWidth(100)
        self.mm_max_captions_spin.setToolTip("Maximum number of captions to generate (0 = unlimited, processes all images)")
        max_captions_row = QHBoxLayout()
        max_captions_row.addWidget(self.mm_max_captions_spin)
        max_captions_row.addStretch()
        mm_caption_layout.addRow(QLabel("Max Captions:"), self._wrap_row(max_captions_row))
        left_panel.addWidget(mm_caption_group)
        
        # Civitai Image Downloader
        civitai_group = QGroupBox("🎨 Civitai Image Downloader")
        civitai_layout = QFormLayout()
        civitai_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        civitai_layout.setHorizontalSpacing(10)
        civitai_layout.setVerticalSpacing(6)
        civitai_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        civitai_group.setLayout(civitai_layout)
        
        civitai_api_row = QHBoxLayout()
        self.civitai_api_key_edit = QLineEdit()
        self.civitai_api_key_edit.setEchoMode(QLineEdit.Password)
        self.civitai_api_key_edit.setPlaceholderText("Your Civitai API key")
        self.civitai_show_key_check = QCheckBox("Show")
        self.civitai_show_key_check.toggled.connect(self.toggle_civitai_api_visibility)
        civitai_api_row.addWidget(self.civitai_api_key_edit)
        civitai_api_row.addWidget(self.civitai_show_key_check)
        civitai_layout.addRow(QLabel("API Key:"), self._wrap_row(civitai_api_row))
        
        # Get repo root for default output directory
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        default_civitai_output = os.path.join(repo_root, "Outputs", "images")
        
        civitai_output_row = QHBoxLayout()
        self.civitai_output_edit = QLineEdit()
        self.civitai_output_edit.setPlaceholderText("Output folder for downloaded images")
        self.civitai_output_edit.setText(default_civitai_output)
        civitai_output_browse_btn = QPushButton("Browse")
        civitai_output_browse_btn.setFixedWidth(80)
        civitai_output_browse_btn.clicked.connect(self.browse_civitai_output)
        civitai_output_row.addWidget(self.civitai_output_edit)
        civitai_output_row.addWidget(civitai_output_browse_btn)
        civitai_layout.addRow(QLabel("Output Folder:"), self._wrap_row(civitai_output_row))
        
        self.civitai_max_images_spin = QSpinBox()
        self.civitai_max_images_spin.setRange(1, 100000)
        self.civitai_max_images_spin.setValue(100)
        self.civitai_max_images_spin.setMaximumWidth(100)
        self.civitai_max_images_spin.setToolTip("Maximum number of images to download")
        max_images_row = QHBoxLayout()
        max_images_row.addWidget(self.civitai_max_images_spin)
        max_images_row.addStretch()
        civitai_layout.addRow(QLabel("Max Images:"), self._wrap_row(max_images_row))
        
        self.civitai_min_width_spin = QSpinBox()
        self.civitai_min_width_spin.setRange(0, 10000)
        self.civitai_min_width_spin.setValue(0)
        self.civitai_min_width_spin.setMaximumWidth(100)
        self.civitai_min_width_spin.setToolTip("Minimum image width (0 = no filter)")
        min_width_row = QHBoxLayout()
        min_width_row.addWidget(self.civitai_min_width_spin)
        min_width_row.addStretch()
        civitai_layout.addRow(QLabel("Min Width:"), self._wrap_row(min_width_row))
        
        self.civitai_min_height_spin = QSpinBox()
        self.civitai_min_height_spin.setRange(0, 10000)
        self.civitai_min_height_spin.setValue(0)
        self.civitai_min_height_spin.setMaximumWidth(100)
        self.civitai_min_height_spin.setToolTip("Minimum image height (0 = no filter)")
        min_height_row = QHBoxLayout()
        min_height_row.addWidget(self.civitai_min_height_spin)
        min_height_row.addStretch()
        civitai_layout.addRow(QLabel("Min Height:"), self._wrap_row(min_height_row))
        
        self.civitai_nsfw_combo = QComboBox()
        self.civitai_nsfw_combo.addItems(["Any (no filter)", "None (SFW only)", "Soft", "Mature", "X (explicit)"])
        self.civitai_nsfw_combo.setToolTip("NSFW filter level")
        civitai_layout.addRow(QLabel("NSFW Level:"), self.civitai_nsfw_combo)
        
        self.civitai_sort_combo = QComboBox()
        self.civitai_sort_combo.addItems(["Newest", "Most Reactions", "Most Comments"])
        self.civitai_sort_combo.setToolTip("Sort mode for image selection")
        civitai_layout.addRow(QLabel("Sort Mode:"), self.civitai_sort_combo)
        
        self.civitai_include_edit = QLineEdit()
        self.civitai_include_edit.setPlaceholderText("Comma-separated terms (ANY match passes)")
        self.civitai_include_edit.setToolTip("Include terms: comma-separated list. At least one term must match (OR logic)")
        civitai_layout.addRow(QLabel("Include Terms:"), self.civitai_include_edit)
        
        self.civitai_exclude_edit = QLineEdit()
        self.civitai_exclude_edit.setPlaceholderText("Comma-separated terms (ANY match blocks)")
        self.civitai_exclude_edit.setToolTip("Exclude terms: comma-separated list. Any matching term will block the image (OR logic)")
        civitai_layout.addRow(QLabel("Exclude Terms:"), self.civitai_exclude_edit)
        
        self.civitai_save_meta_check = QCheckBox("Save metadata JSONL")
        self.civitai_save_meta_check.setChecked(True)
        self.civitai_save_meta_check.setToolTip("Save metadata JSONL file with image information")
        civitai_layout.addRow("", self.civitai_save_meta_check)
        
        self.civitai_batch_size_spin = QSpinBox()
        self.civitai_batch_size_spin.setRange(1, 500)
        self.civitai_batch_size_spin.setValue(200)
        self.civitai_batch_size_spin.setMaximumWidth(100)
        self.civitai_batch_size_spin.setToolTip("Number of images per API request (batch size)")
        batch_size_row = QHBoxLayout()
        batch_size_row.addWidget(self.civitai_batch_size_spin)
        batch_size_row.addStretch()
        civitai_layout.addRow(QLabel("Batch Size:"), self._wrap_row(batch_size_row))
        
        self.civitai_max_empty_batches_spin = QSpinBox()
        self.civitai_max_empty_batches_spin.setRange(1, 1000)
        self.civitai_max_empty_batches_spin.setValue(40)
        self.civitai_max_empty_batches_spin.setMaximumWidth(100)
        self.civitai_max_empty_batches_spin.setToolTip("Maximum number of empty batches before stopping (prevents infinite loops)")
        max_empty_batches_row = QHBoxLayout()
        max_empty_batches_row.addWidget(self.civitai_max_empty_batches_spin)
        max_empty_batches_row.addStretch()
        civitai_layout.addRow(QLabel("Max Empty Batches:"), self._wrap_row(max_empty_batches_row))
        
        self.civitai_wait_time_spin = QDoubleSpinBox()
        self.civitai_wait_time_spin.setRange(0.0, 300.0)
        self.civitai_wait_time_spin.setValue(0.0)
        self.civitai_wait_time_spin.setSingleStep(0.5)
        self.civitai_wait_time_spin.setDecimals(1)
        self.civitai_wait_time_spin.setMaximumWidth(100)
        self.civitai_wait_time_spin.setSuffix(" s")
        self.civitai_wait_time_spin.setToolTip("Wait time in seconds between page requests (0 = no wait)")
        wait_time_row = QHBoxLayout()
        wait_time_row.addWidget(self.civitai_wait_time_spin)
        wait_time_row.addStretch()
        civitai_layout.addRow(QLabel("Wait Between Pages:"), self._wrap_row(wait_time_row))
        
        left_panel.addWidget(civitai_group)
        left_panel.addStretch(1)

        # Right panel - Logs
        mm_progress_group = QGroupBox("Run Status")
        mm_progress_layout = QVBoxLayout()
        mm_progress_layout.setSpacing(6)
        mm_progress_group.setLayout(mm_progress_layout)

        self.mm_log_view = QPlainTextEdit()
        self.mm_log_view.setReadOnly(True)
        self.mm_log_view.setMaximumBlockCount(1000)
        self.mm_log_view.setPlaceholderText("Logs will appear here...")
        mm_progress_layout.addWidget(QLabel("Logs:"))
        mm_progress_layout.addWidget(self.mm_log_view, stretch=1)

        right_panel.addWidget(mm_progress_group, stretch=1)

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
        elif any(word in line_lower for word in ['saved', 'success', 'generated', 'processing', 'preview']):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(100, 255, 100))
            return fmt
        elif any(word in line_lower for word in ['info', 'progress', 'starting', 'waiting']):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(100, 150, 255))
            return fmt
        else:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(200, 200, 200))
            return fmt

    def _load_initial_config(self):
        # Load from a config file if it exists, or use defaults
        config_file = CONFIG_FILE
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                
                # Load API key for current API type, or use general api_key as fallback
                api_type = cfg.get("api_type", "Anthropic Claude")
                api_keys = cfg.get("api_keys", {})  # Dictionary of api_type -> api_key
                api_key = api_keys.get(api_type, cfg.get("api_key", ""))
                self.api_key_edit.setText(api_key)
                
                self.endpoint_edit.setText(cfg.get("endpoint", ""))
                output_dir = cfg.get("output_dir", "outputs")
                self.output_dir_edit.setText(output_dir)
                model_name = cfg.get("model", "")
                if model_name:
                    # Try to set the model, add it if not in list (since combo is editable)
                    index = self.model_combo.findText(model_name)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    else:
                        self.model_combo.setCurrentText(model_name)
                self.output_dir_edit.setText(cfg.get("output_dir", ""))
                self.system_message_edit.setPlainText(cfg.get("system_message", ""))
                self.user_first_message_edit.setPlainText(cfg.get("user_first_message", ""))
                self.assistant_first_message_edit.setPlainText(cfg.get("assistant_first_message", ""))
                self.user_start_tag_edit.setText(cfg.get("user_start_tag", "<human_turn>"))
                self.user_end_tag_edit.setText(cfg.get("user_end_tag", "</human_turn>"))
                self.assistant_start_tag_edit.setText(cfg.get("assistant_start_tag", "<claude_turn>"))
                self.assistant_end_tag_edit.setText(cfg.get("assistant_end_tag", "</claude_turn>"))
                self.is_instruct_check.setChecked(cfg.get("is_instruct", False))
                self.min_delay_spin.setValue(cfg.get("min_delay", 0.1))
                self.max_delay_spin.setValue(cfg.get("max_delay", 0.5))
                self.stop_percentage_spin.setValue(cfg.get("stop_percentage", 0.05))
                self.min_turns_spin.setValue(cfg.get("min_turns", 1))
                self.refusal_phrases_edit.setPlainText("\n".join(cfg.get("refusal_phrases", [])))
                self.force_retry_phrases_edit.setPlainText("\n".join(cfg.get("force_retry_phrases", [])))
                api_type = cfg.get("api_type", "Anthropic Claude")
                index = self.api_type_combo.findText(api_type)
                if index >= 0:
                    self.api_type_combo.setCurrentIndex(index)
            except Exception as e:
                self._append_log(f"Error loading config: {e}")
        else:
            # Set defaults
            self.output_dir_edit.setText("outputs")
            self.user_start_tag_edit.setText("<human_turn>")
            self.user_end_tag_edit.setText("</human_turn>")
            self.assistant_start_tag_edit.setText("<claude_turn>")
            self.assistant_end_tag_edit.setText("</claude_turn>")
            self.refusal_phrases_edit.setPlainText("Upon further reflection\nI can't engage")
            self.force_retry_phrases_edit.setPlainText("shivers down")
        
        # Load processing config - try main config first, then fallback to old GrokMaxxer config
        proc_cfg = {}
        
        # Try main config file first
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    main_cfg = json.load(f)
                    # Load processing settings from main config
                    # Try proc_api_key first, then check api_keys dict for Grok
                    if main_cfg.get("proc_api_key"):
                        proc_cfg["api_key"] = main_cfg.get("proc_api_key")
                    elif main_cfg.get("api_keys"):
                        api_keys = main_cfg.get("api_keys", {})
                        # Try "Grok (xAI)" first, then "Grok" as fallback
                        proc_cfg["api_key"] = api_keys.get("Grok (xAI)", api_keys.get("Grok", ""))
                    if main_cfg.get("proc_last_input"):
                        proc_cfg["last_input"] = main_cfg.get("proc_last_input")
                    if main_cfg.get("proc_last_output"):
                        proc_cfg["last_output"] = main_cfg.get("proc_last_output")
                    if main_cfg.get("proc_system_prompt"):
                        proc_cfg["system_prompt"] = main_cfg.get("proc_system_prompt")
                    if main_cfg.get("proc_model"):
                        proc_cfg["model"] = main_cfg.get("proc_model")
                    if main_cfg.get("proc_start_line"):
                        proc_cfg["start_line"] = main_cfg.get("proc_start_line")
                    if main_cfg.get("proc_end_line"):
                        proc_cfg["end_line"] = main_cfg.get("proc_end_line")
                    if "proc_rewrite_existing" in main_cfg:
                        proc_cfg["rewrite_existing"] = main_cfg.get("proc_rewrite_existing")
                    if "proc_extra_pairs" in main_cfg:
                        proc_cfg["extra_pairs"] = main_cfg.get("proc_extra_pairs")
                    if main_cfg.get("proc_num_new"):
                        proc_cfg["num_new"] = main_cfg.get("proc_num_new")
                    if "proc_reply_in_character" in main_cfg:
                        proc_cfg["reply_in_character"] = main_cfg.get("proc_reply_in_character")
                    if "proc_dynamic_names_mode" in main_cfg:
                        proc_cfg["dynamic_names_mode"] = main_cfg.get("proc_dynamic_names_mode")
                    if "proc_concurrency" in main_cfg:
                        proc_cfg["concurrency"] = main_cfg.get("proc_concurrency")
                    if "proc_batch_size" in main_cfg:
                        proc_cfg["batch_size"] = main_cfg.get("proc_batch_size")
            except Exception:
                pass
        
        # Fallback to old GrokMaxxer config if main config doesn't have processing settings
        if not proc_cfg.get("api_key"):
            proc_config_file = str(Path(__file__).parent / "grok_tool_config.json")
            if os.path.exists(proc_config_file):
                try:
                    with open(proc_config_file, 'r', encoding='utf-8') as f:
                        old_proc_cfg = json.load(f)
                        proc_cfg.update(old_proc_cfg)
                except Exception:
                    pass
        
        # Apply loaded config
        if proc_cfg:
            if proc_cfg.get("api_key"):
                self.proc_api_key_edit.setText(proc_cfg.get("api_key", ""))
            if proc_cfg.get("last_input"):
                self.proc_input_edit.setText(proc_cfg.get("last_input", ""))
            if proc_cfg.get("last_output"):
                # Default to outputs folder if no saved output
                saved_output = proc_cfg.get("last_output", "")
                if saved_output:
                    self.proc_output_edit.setText(saved_output)
                else:
                    # Set default to outputs folder
                    repo_root = os.path.dirname(os.path.dirname(__file__))
                    outputs_dir = os.path.join(repo_root, "outputs")
                    default_output = os.path.join(outputs_dir, "processed_output.jsonl")
                    self.proc_output_edit.setText(default_output)
            if proc_cfg.get("system_prompt"):
                self.proc_system_prompt_edit.setPlainText(proc_cfg.get("system_prompt", ""))
            if proc_cfg.get("model"):
                self.proc_model_edit.setText(proc_cfg.get("model", "grok-4.1-fast-non-reasoning"))
            elif proc_cfg.get("models_csv"):
                model_name = proc_cfg.get("models_csv", "").split(',')[0].strip() if proc_cfg.get("models_csv") else "grok-4.1-fast-non-reasoning"
                self.proc_model_edit.setText(model_name)
            if proc_cfg.get("start_line"):
                self.proc_start_line_edit.setText(str(proc_cfg.get("start_line", "")))
            if proc_cfg.get("end_line"):
                self.proc_end_line_edit.setText(str(proc_cfg.get("end_line", "")))
            
            if "rewrite_existing" in proc_cfg:
                self.proc_rewrite_check.setChecked(bool(proc_cfg.get("rewrite_existing", True)))
            if "extra_pairs" in proc_cfg:
                self.proc_extra_pairs_spin.setValue(int(proc_cfg.get("extra_pairs", 0)) if isinstance(proc_cfg.get("extra_pairs"), (int, float)) else 0)
            if proc_cfg.get("num_new"):
                self.proc_num_new_edit.setText(str(int(proc_cfg.get("num_new", 0))))
            if "reply_in_character" in proc_cfg:
                self.proc_reply_in_character_check.setChecked(bool(proc_cfg.get("reply_in_character", False)))
            if "dynamic_names_mode" in proc_cfg:
                self.proc_dynamic_names_check.setChecked(bool(proc_cfg.get("dynamic_names_mode", False)))
            if "concurrency" in proc_cfg:
                self.proc_concurrency_spin.setValue(int(proc_cfg.get("concurrency", 20)))
            if "batch_size" in proc_cfg:
                self.proc_batch_size_spin.setValue(int(proc_cfg.get("batch_size", 16)))
        
        # Load multimodal config
        if hasattr(self, 'mm_image_dir_edit'):
            if cfg.get("mm_image_dir"):
                self.mm_image_dir_edit.setText(cfg.get("mm_image_dir", ""))
            if cfg.get("mm_output"):
                saved_output = cfg.get("mm_output", "")
                # Fix old paths that point to App\outputs
                if "App\\outputs" in saved_output or "App/outputs" in saved_output:
                    # Replace with correct repo root outputs
                    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    outputs_dir = os.path.join(repo_root, "outputs")
                    filename = os.path.basename(saved_output)
                    saved_output = os.path.join(outputs_dir, filename)
                self.mm_output_edit.setText(saved_output)
            if cfg.get("mm_api_key"):
                self.mm_api_key_edit.setText(cfg.get("mm_api_key", ""))
            elif cfg.get("api_keys"):
                api_keys = cfg.get("api_keys", {})
                # Try to get API key for multimodal (default to OpenAI Vision)
                mm_api_type = cfg.get("mm_api_type", "OpenAI Vision")
                mm_api_key = api_keys.get(mm_api_type, "")
                if mm_api_key:
                    self.mm_api_key_edit.setText(mm_api_key)
            if cfg.get("mm_endpoint"):
                self.mm_endpoint_edit.setText(cfg.get("mm_endpoint", ""))
            if cfg.get("mm_model"):
                model_name = cfg.get("mm_model", "")
                if model_name:
                    index = self.mm_model_combo.findText(model_name)
                    if index >= 0:
                        self.mm_model_combo.setCurrentIndex(index)
                    else:
                        self.mm_model_combo.setCurrentText(model_name)
            if cfg.get("mm_api_type"):
                api_type = cfg.get("mm_api_type", "OpenAI Vision")
                index = self.mm_api_type_combo.findText(api_type)
                if index >= 0:
                    self.mm_api_type_combo.setCurrentIndex(index)
                    # Only update endpoint if it wasn't saved (to preserve custom endpoints)
                    if not cfg.get("mm_endpoint"):
                        self._update_mm_endpoint_placeholder(api_type)
            if cfg.get("mm_caption_prompt"):
                self.mm_caption_prompt_edit.setPlainText(cfg.get("mm_caption_prompt", ""))
            if "mm_max_tokens" in cfg:
                self.mm_max_tokens_spin.setValue(int(cfg.get("mm_max_tokens", 500)))
            if "mm_temperature" in cfg:
                self.mm_temperature_spin.setValue(float(cfg.get("mm_temperature", 0.7)))
            if "mm_batch_size" in cfg:
                self.mm_batch_size_spin.setValue(int(cfg.get("mm_batch_size", 1)))
            if "mm_max_captions" in cfg:
                self.mm_max_captions_spin.setValue(int(cfg.get("mm_max_captions", 0)))
            if cfg.get("mm_hf_dataset"):
                self.mm_hf_dataset_edit.setText(cfg.get("mm_hf_dataset", ""))
            if "mm_use_hf_dataset" in cfg:
                use_hf = bool(cfg.get("mm_use_hf_dataset", False))
                self.mm_use_hf_dataset_check.setChecked(use_hf)
                self._toggle_hf_dataset_mode(use_hf)
            if cfg.get("mm_hf_token"):
                self.mm_hf_token_edit.setText(cfg.get("mm_hf_token", ""))
        
        # Load Civitai config
        if hasattr(self, 'civitai_api_key_edit'):
            # Load API key from api_keys dict with "Civitai" key
            if cfg.get("api_keys"):
                api_keys = cfg.get("api_keys", {})
                civitai_api_key = api_keys.get("Civitai", "")
                if civitai_api_key:
                    self.civitai_api_key_edit.setText(civitai_api_key)
            # Also check for legacy civitai_api_key
            elif cfg.get("civitai_api_key"):
                self.civitai_api_key_edit.setText(cfg.get("civitai_api_key", ""))
            
            if cfg.get("civitai_output_dir"):
                saved_output = cfg.get("civitai_output_dir", "")
                # Fix old paths
                if "App\\outputs" in saved_output or "App/outputs" in saved_output:
                    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    outputs_dir = os.path.join(repo_root, "Outputs", "images")
                    saved_output = outputs_dir
                self.civitai_output_edit.setText(saved_output)
            else:
                # Set default
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                default_output = os.path.join(repo_root, "Outputs", "images")
                self.civitai_output_edit.setText(default_output)
            
            if "civitai_max_images" in cfg:
                self.civitai_max_images_spin.setValue(int(cfg.get("civitai_max_images", 100)))
            if "civitai_min_width" in cfg:
                self.civitai_min_width_spin.setValue(int(cfg.get("civitai_min_width", 0)))
            if "civitai_min_height" in cfg:
                self.civitai_min_height_spin.setValue(int(cfg.get("civitai_min_height", 0)))
            if cfg.get("civitai_nsfw_level"):
                nsfw_level = cfg.get("civitai_nsfw_level", "Any (no filter)")
                index = self.civitai_nsfw_combo.findText(nsfw_level)
                if index >= 0:
                    self.civitai_nsfw_combo.setCurrentIndex(index)
            if cfg.get("civitai_sort_mode"):
                sort_mode = cfg.get("civitai_sort_mode", "Newest")
                index = self.civitai_sort_combo.findText(sort_mode)
                if index >= 0:
                    self.civitai_sort_combo.setCurrentIndex(index)
            if cfg.get("civitai_include_terms"):
                self.civitai_include_edit.setText(cfg.get("civitai_include_terms", ""))
            if cfg.get("civitai_exclude_terms"):
                self.civitai_exclude_edit.setText(cfg.get("civitai_exclude_terms", ""))
            if "civitai_save_meta_jsonl" in cfg:
                self.civitai_save_meta_check.setChecked(bool(cfg.get("civitai_save_meta_jsonl", True)))
            if "civitai_batch_size" in cfg:
                self.civitai_batch_size_spin.setValue(int(cfg.get("civitai_batch_size", 200)))
            if "civitai_max_empty_batches" in cfg:
                self.civitai_max_empty_batches_spin.setValue(int(cfg.get("civitai_max_empty_batches", 40)))
            if "civitai_wait_time" in cfg:
                self.civitai_wait_time_spin.setValue(float(cfg.get("civitai_wait_time", 0.0)))

    def _save_config(self):
        config_file = CONFIG_FILE
        
        # Load existing config to preserve API keys for other providers
        existing_cfg = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    existing_cfg = json.load(f)
            except Exception:
                pass
        
        # Get existing API keys dict or create new one
        api_keys = existing_cfg.get("api_keys", {})
        # Save current API key for current API type
        current_api_type = self.api_type_combo.currentText()
        api_keys[current_api_type] = self.api_key_edit.text().strip()
        
        cfg = {
            "api_key": self.api_key_edit.text().strip(),  # Keep for backward compatibility
            "api_keys": api_keys,  # Per-provider API keys
            "endpoint": self.endpoint_edit.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "system_message": self.system_message_edit.toPlainText().strip(),
            "user_first_message": self.user_first_message_edit.toPlainText().strip(),
            "assistant_first_message": self.assistant_first_message_edit.toPlainText().strip(),
            "user_start_tag": self.user_start_tag_edit.text().strip(),
            "user_end_tag": self.user_end_tag_edit.text().strip(),
            "assistant_start_tag": self.assistant_start_tag_edit.text().strip(),
            "assistant_end_tag": self.assistant_end_tag_edit.text().strip(),
            "is_instruct": self.is_instruct_check.isChecked(),
            "min_delay": self.min_delay_spin.value(),
            "max_delay": self.max_delay_spin.value(),
            "stop_percentage": self.stop_percentage_spin.value(),
            "min_turns": self.min_turns_spin.value(),
            "refusal_phrases": [p.strip() for p in self.refusal_phrases_edit.toPlainText().split('\n') if p.strip()],
            "force_retry_phrases": [p.strip() for p in self.force_retry_phrases_edit.toPlainText().split('\n') if p.strip()],
            "api_type": self.api_type_combo.currentText(),
            # Processing tab config
            "proc_last_input": self.proc_input_edit.text().strip(),
            "proc_last_output": self.proc_output_edit.text().strip(),
            "proc_system_prompt": self.proc_system_prompt_edit.toPlainText().strip(),
            "proc_model": self.proc_model_edit.text().strip(),
            "proc_start_line": self.proc_start_line_edit.text().strip(),
            "proc_end_line": self.proc_end_line_edit.text().strip(),
            "proc_rewrite_existing": self.proc_rewrite_check.isChecked(),
            "proc_extra_pairs": self.proc_extra_pairs_spin.value(),
            "proc_num_new": self.proc_num_new_edit.text().strip(),
            "proc_reply_in_character": self.proc_reply_in_character_check.isChecked(),
            "proc_dynamic_names_mode": self.proc_dynamic_names_check.isChecked(),
            "proc_concurrency": self.proc_concurrency_spin.value(),
            "proc_batch_size": self.proc_batch_size_spin.value(),
            # Multimodal tab config
            "mm_image_dir": self.mm_image_dir_edit.text().strip() if hasattr(self, 'mm_image_dir_edit') else "",
            "mm_output": self.mm_output_edit.text().strip() if hasattr(self, 'mm_output_edit') else "",
            "mm_api_key": self.mm_api_key_edit.text().strip() if hasattr(self, 'mm_api_key_edit') else "",
            "mm_endpoint": self.mm_endpoint_edit.text().strip() if hasattr(self, 'mm_endpoint_edit') else "",
            "mm_model": self.mm_model_combo.currentText().strip() if hasattr(self, 'mm_model_combo') else "",
            "mm_api_type": self.mm_api_type_combo.currentText() if hasattr(self, 'mm_api_type_combo') else "",
            "mm_caption_prompt": self.mm_caption_prompt_edit.toPlainText().strip() if hasattr(self, 'mm_caption_prompt_edit') else "",
            "mm_max_tokens": self.mm_max_tokens_spin.value() if hasattr(self, 'mm_max_tokens_spin') else 500,
            "mm_temperature": self.mm_temperature_spin.value() if hasattr(self, 'mm_temperature_spin') else 0.7,
            "mm_batch_size": self.mm_batch_size_spin.value() if hasattr(self, 'mm_batch_size_spin') else 1,
            "mm_max_captions": self.mm_max_captions_spin.value() if hasattr(self, 'mm_max_captions_spin') else 0,
            "mm_hf_dataset": self.mm_hf_dataset_edit.text().strip() if hasattr(self, 'mm_hf_dataset_edit') else "",
            "mm_use_hf_dataset": self.mm_use_hf_dataset_check.isChecked() if hasattr(self, 'mm_use_hf_dataset_check') else False,
            "mm_hf_token": self.mm_hf_token_edit.text().strip() if hasattr(self, 'mm_hf_token_edit') else "",
        }
        
        # Save processing API key if it exists
        if hasattr(self, 'proc_api_key_edit'):
            proc_api_key = self.proc_api_key_edit.text().strip()
            if proc_api_key:
                # Save processing API key (defaults to Grok for processing tab)
                proc_api_type = "Grok (xAI)"  # Default for processing tab
                api_keys[proc_api_type] = proc_api_key
                cfg["api_keys"] = api_keys
                cfg["proc_api_key"] = proc_api_key  # Also save directly for backward compat
        
        # Save multimodal API key if it exists
        if hasattr(self, 'mm_api_key_edit'):
            mm_api_key = self.mm_api_key_edit.text().strip()
            if mm_api_key:
                mm_api_type = self.mm_api_type_combo.currentText() if hasattr(self, 'mm_api_type_combo') else "OpenAI Vision"
                api_keys[mm_api_type] = mm_api_key
                cfg["api_keys"] = api_keys
        
        # Save Civitai API key if it exists
        if hasattr(self, 'civitai_api_key_edit'):
            civitai_api_key = self.civitai_api_key_edit.text().strip()
            if civitai_api_key:
                api_keys["Civitai"] = civitai_api_key
                cfg["api_keys"] = api_keys
            # Save Civitai settings
            cfg["civitai_output_dir"] = self.civitai_output_edit.text().strip() if hasattr(self, 'civitai_output_edit') else ""
            cfg["civitai_max_images"] = self.civitai_max_images_spin.value() if hasattr(self, 'civitai_max_images_spin') else 100
            cfg["civitai_min_width"] = self.civitai_min_width_spin.value() if hasattr(self, 'civitai_min_width_spin') else 0
            cfg["civitai_min_height"] = self.civitai_min_height_spin.value() if hasattr(self, 'civitai_min_height_spin') else 0
            cfg["civitai_nsfw_level"] = self.civitai_nsfw_combo.currentText() if hasattr(self, 'civitai_nsfw_combo') else "Any (no filter)"
            cfg["civitai_sort_mode"] = self.civitai_sort_combo.currentText() if hasattr(self, 'civitai_sort_combo') else "Newest"
            cfg["civitai_include_terms"] = self.civitai_include_edit.text().strip() if hasattr(self, 'civitai_include_edit') else ""
            cfg["civitai_exclude_terms"] = self.civitai_exclude_edit.text().strip() if hasattr(self, 'civitai_exclude_edit') else ""
            cfg["civitai_save_meta_jsonl"] = self.civitai_save_meta_check.isChecked() if hasattr(self, 'civitai_save_meta_check') else True
            cfg["civitai_batch_size"] = self.civitai_batch_size_spin.value() if hasattr(self, 'civitai_batch_size_spin') else 200
            cfg["civitai_max_empty_batches"] = self.civitai_max_empty_batches_spin.value() if hasattr(self, 'civitai_max_empty_batches_spin') else 40
            cfg["civitai_wait_time"] = self.civitai_wait_time_spin.value() if hasattr(self, 'civitai_wait_time_spin') else 0.0
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._append_log(f"Error saving config: {e}")

    def toggle_api_visibility(self, checked):
        if checked:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def _update_api_key_for_type(self, api_type):
        """Update API key field when API type changes"""
        # Load saved API key for this provider
        config_file = CONFIG_FILE
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                api_keys = cfg.get("api_keys", {})
                api_key = api_keys.get(api_type, "")
                if api_key:
                    self.api_key_edit.setText(api_key)
            except Exception:
                pass
    
    def _refresh_models(self):
        """Manually refresh models for the current API type"""
        print(f"DEBUG: _refresh_models called")
        api_type = self.api_type_combo.currentText()
        api_key = self.api_key_edit.text().strip()
        print(f"DEBUG: API type = {api_type}, API key present = {bool(api_key)}")
        self._update_models(api_type)
    
    def _update_models(self, api_type):
        """Fetch and update model dropdown based on selected API type"""
        print(f"DEBUG: _update_models called with api_type={api_type}")
        api_key = self.api_key_edit.text().strip()
        
        # Check if we have an API key before fetching
        if not api_key:
            # No API key - show message but keep button enabled
            print("DEBUG: No API key, showing message")
            self.model_combo.clear()
            self.model_combo.addItem("(Enter API key and click Refresh)")
            self.model_combo.setEnabled(True)
            self.refresh_models_button.setEnabled(True)
            return
        
        # We have an API key, start fetching
        print(f"DEBUG: Starting model fetch for {api_type}")
        self.model_combo.clear()
        self.model_combo.addItem("Loading models...")
        self.model_combo.setEnabled(False)
        self.refresh_models_button.setEnabled(False)
        
        # Fetch models in a separate thread to avoid blocking UI
        thread = threading.Thread(
            target=self._fetch_models,
            args=(api_type, api_key),
            daemon=True
        )
        thread.start()
        print(f"DEBUG: Thread started: {thread.is_alive()}")
    
    def _fetch_models(self, api_type, api_key):
        """Fetch available models from the API"""
        print(f"DEBUG: _fetch_models started for {api_type}")
        models = []
        error_msg = None
        
        try:
            if api_type == "Anthropic Claude":
                # Anthropic doesn't have a public models endpoint, use common models
                models = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ]
            elif api_type in ["OpenAI Official", "OpenAI Chat Completions"]:
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            # Filter for chat models
                            if any(x in model_id for x in ["gpt-4", "gpt-3.5"]):
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"OpenAI API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing OpenAI response: {str(e)}"
                if not models:
                    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            elif api_type == "OpenAI Text Completions":
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            # Filter for text completion models
                            if "instruct" in model_id or "davinci" in model_id:
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"OpenAI API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing OpenAI response: {str(e)}"
                if not models:
                    models = ["gpt-3.5-turbo-instruct", "text-davinci-003", "text-davinci-002"]
            elif api_type == "Grok (xAI)":
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.x.ai/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        # Grok API might return data in different formats
                        items = data.get("data") or data.get("models") or data.get("model_list") or []
                        if isinstance(items, list):
                            for m in items:
                                if isinstance(m, dict):
                                    mid = m.get("id") or m.get("name") or m.get("model_id")
                                    if mid and isinstance(mid, str) and "image" not in mid.lower():
                                        models.append(mid)
                        elif isinstance(data, list):
                            for m in data:
                                if isinstance(m, dict):
                                    mid = m.get("id") or m.get("name") or m.get("model_id")
                                    if mid and isinstance(mid, str) and "image" not in mid.lower():
                                        models.append(mid)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"Grok API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing Grok response: {str(e)}"
                if not models:
                    models = ["grok-4.1-fast-non-reasoning", "grok-4-fast-non-reasoning", "grok-beta"]
            elif api_type == "Gemini (Google)":
                # Gemini models are typically hardcoded, but we can try to fetch
                models = [
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-flash-8b",
                    "gemini-pro",
                ]
            elif api_type == "OpenRouter":
                if api_key:
                    try:
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "HTTP-Referer": "https://github.com/ShareGPT-Formaxxing",
                            "X-Title": "SynthMaxxer"
                        }
                        resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            if model_id:
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"OpenRouter API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing OpenRouter response: {str(e)}"
                if not models:
                    models = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
            elif api_type == "DeepSeek":
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.deepseek.com/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            if model_id:
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"DeepSeek API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing DeepSeek response: {str(e)}"
                if not models:
                    models = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
        
        # Update UI on main thread - ALWAYS update, even on error
        print(f"DEBUG: Fetch complete. Models found: {len(models)}, Error: {error_msg}")
        if not models and not error_msg:
            models = ["(No models found - enter manually)"]
        elif error_msg:
            models = [f"(Error: {error_msg[:50]})"]
        
        # Emit signal to update UI on main thread
        print(f"DEBUG: Emitting models_ready signal with {len(models)} models")
        self.model_fetcher.models_ready.emit(models)
        print("DEBUG: Signal emitted")
    
    def _on_models_ready(self, models):
        """Handle models ready signal on main thread"""
        try:
            print(f"DEBUG: _on_models_ready called with {len(models)} models")
            self.model_combo.clear()
            self.model_combo.addItems(models)
            self.model_combo.setEnabled(True)
            self.refresh_models_button.setEnabled(True)  # Always re-enable refresh button
            if models and not models[0].startswith("(Error:") and models[0] != "(No models found - enter manually)":
                self.model_combo.setCurrentIndex(0)
            print("DEBUG: UI update complete")
        except Exception as e:
            # On any error, make sure controls are enabled
            print(f"DEBUG: UI update error: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.model_combo.clear()
                self.model_combo.addItem(f"(UI Error: {str(e)})")
                self.model_combo.setEnabled(True)
                self.refresh_models_button.setEnabled(True)
            except:
                pass

    def _update_endpoint_placeholder(self, api_type):
        """Update endpoint and assistant tags based on selected API type"""
        if api_type == "Anthropic Claude":
            endpoint = "https://api.anthropic.com/v1/messages"
            assistant_start = "<claude_turn>"
            assistant_end = "</claude_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "OpenAI Official":
            endpoint = "https://api.openai.com/v1/chat/completions"
            assistant_start = "<gpt_turn>"
            assistant_end = "</gpt_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "OpenAI Chat Completions":
            endpoint = "http://localhost:8000/v1/chat/completions"
            assistant_start = "<gpt_turn>"
            assistant_end = "</gpt_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "OpenAI Text Completions":
            endpoint = "http://localhost:8000/v1/completions"
            assistant_start = "<gpt_turn>"
            assistant_end = "</gpt_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "Grok (xAI)":
            endpoint = "https://api.x.ai/v1/chat/completions"
            assistant_start = "<grok_turn>"
            assistant_end = "</grok_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "Gemini (Google)":
            # Gemini endpoint includes model name - use a default model
            default_model = "gemini-2.0-flash-exp"
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{default_model}:streamGenerateContent"
            assistant_start = "<gemini_turn>"
            assistant_end = "</gemini_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
            # Also update model field with default
            if not self.model_combo.currentText().strip():
                self.model_combo.setCurrentText(default_model)
        elif api_type == "OpenRouter":
            endpoint = "https://openrouter.ai/api/v1/chat/completions"
            assistant_start = "<gpt_turn>"
            assistant_end = "</gpt_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)
        elif api_type == "DeepSeek":
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            assistant_start = "<deepseek_turn>"
            assistant_end = "</deepseek_turn>"
            self.endpoint_edit.setPlaceholderText(endpoint)
            self.endpoint_edit.setText(endpoint)
            self.assistant_start_tag_edit.setText(assistant_start)
            self.assistant_end_tag_edit.setText(assistant_end)

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            "",
        )
        if directory:
            self.output_dir_edit.setText(directory)

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

    def start_generation(self):
        # Validate inputs
        api_key = self.api_key_edit.text().strip()
        endpoint = self.endpoint_edit.text().strip()
        model = self.model_combo.currentText().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not api_key:
            self._show_error("Please enter your API key.")
            return
        if not endpoint:
            self._show_error("Please enter the API endpoint.")
            return
        if not model:
            self._show_error("Please enter the model name.")
            return
        if not output_dir:
            self._show_error("Please enter the output directory.")
            return

        # Save config
        self._save_config()

        # Reset UI state
        self.log_view.clear()
        self._append_log("=== Generation started ===")
        self.setWindowTitle(f"{APP_TITLE} - Generating...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_flag.clear()

        self.queue = queue.Queue()

        # Collect all configuration values
        system_message = self.system_message_edit.toPlainText().strip()
        user_first_message = self.user_first_message_edit.toPlainText().strip()
        assistant_first_message = self.assistant_first_message_edit.toPlainText().strip()
        user_start_tag = self.user_start_tag_edit.text().strip()
        user_end_tag = self.user_end_tag_edit.text().strip()
        assistant_start_tag = self.assistant_start_tag_edit.text().strip()
        assistant_end_tag = self.assistant_end_tag_edit.text().strip()
        is_instruct = self.is_instruct_check.isChecked()
        min_delay = self.min_delay_spin.value()
        max_delay = self.max_delay_spin.value()
        stop_percentage = self.stop_percentage_spin.value()
        min_turns = self.min_turns_spin.value()
        refusal_phrases = [p.strip() for p in self.refusal_phrases_edit.toPlainText().split('\n') if p.strip()]
        force_retry_phrases = [p.strip() for p in self.force_retry_phrases_edit.toPlainText().split('\n') if p.strip()]
        api_type = self.api_type_combo.currentText()

        # Start worker thread using SynthMaxxer.worker
        self.worker_thread = threading.Thread(
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
                self.stop_flag,
                self.queue,
            ),
            daemon=True,
        )
        self.worker_thread.start()
        self.timer.start()

    def stop_generation(self):
        self.stop_flag.set()
        self._append_log("Stopping generation...")
        self.stop_button.setEnabled(False)

    def check_queue(self):
        # Check multimodal queue
        if hasattr(self, 'mm_queue') and self.mm_queue:
            try:
                while True:
                    msg_type, msg = self.mm_queue.get_nowait()
                    if msg_type == "log":
                        self._append_mm_log(str(msg))
                    elif msg_type == "success":
                        self.setWindowTitle(f"{APP_TITLE} - Done")
                        self.mm_start_button.setEnabled(True)
                        self.mm_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.mm_start_button.setEnabled(True)
                        self.mm_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "preview_image":
                        # Display the preview image
                        self._show_preview_image(msg)
                    elif msg_type == "stopped":
                        self.mm_start_button.setEnabled(True)
                        self.mm_stop_button.setEnabled(False)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
                        if msg:  # Only log if there's a message
                            self._append_mm_log(str(msg))
            except queue.Empty:
                pass
        
        # Check Civitai queue
        if hasattr(self, 'civitai_queue') and self.civitai_queue:
            try:
                while True:
                    msg_type, msg = self.civitai_queue.get_nowait()
                    if msg_type == "log":
                        self._append_mm_log(str(msg))
                    elif msg_type == "success":
                        self.setWindowTitle(f"{APP_TITLE} - Done")
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "stopped":
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
                        if msg:  # Only log if there's a message
                            self._append_mm_log(str(msg))
            except queue.Empty:
                pass
        
        # Always check proc_queue if it exists, even if main queue doesn't
        proc_queue_checked = False
        if hasattr(self, 'proc_queue') and self.proc_queue:
            proc_queue_checked = True
            try:
                while True:
                    msg_type, msg = self.proc_queue.get_nowait()
                    print(f"PROC_QUEUE_MSG: {msg_type} = {msg}")  # Debug output
                    if msg_type == "log":
                        self._append_proc_log(str(msg))
                    elif msg_type == "success":
                        self.setWindowTitle(f"{APP_TITLE} - Done")
                        self.process_button.setEnabled(True)
                        self.generate_cache_button.setEnabled(True)
                        self.improve_cache_button.setEnabled(True)
                        self.timer.stop()
                        self._append_proc_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.process_button.setEnabled(True)
                        self.generate_cache_button.setEnabled(True)
                        self.improve_cache_button.setEnabled(True)
                        self.timer.stop()
                        self._append_proc_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "stopped":
                        self.process_button.setEnabled(True)
                        self.generate_cache_button.setEnabled(True)
                        self.improve_cache_button.setEnabled(True)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
            except queue.Empty:
                pass
        
        # Now check main queue for generation tab
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
                    self.start_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    self.timer.stop()
                    self._append_log(str(msg))
                elif msg_type == "error":
                    self.setWindowTitle(f"{APP_TITLE} - Error")
                    self.start_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    self.timer.stop()
                    self._append_log(str(msg))
                    self._show_error(str(msg))
                elif msg_type == "stopped":
                    self.setWindowTitle(f"{APP_TITLE} - Stopped")
                    self.start_button.setEnabled(True)
                    self.stop_button.setEnabled(False)
                    self.timer.stop()
                    self._append_log(str(msg))
        except queue.Empty:
            pass

    def browse_proc_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select input JSONL file",
            "",
            "JSONL files (*.jsonl);;All files (*.*)",
        )
        if filename:
            self.proc_input_edit.setText(filename)

    def browse_proc_output(self):
        # Default to outputs folder in repo root
        default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        if not os.path.exists(default_path):
            os.makedirs(default_path, exist_ok=True)
        
        # Get current value or use default
        current_path = self.proc_output_edit.text().strip()
        if current_path and os.path.dirname(current_path):
            start_dir = os.path.dirname(current_path)
        else:
            start_dir = default_path
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select output JSONL file",
            os.path.join(start_dir, "processed_output.jsonl"),
            "JSONL files (*.jsonl);;All files (*.*)",
        )
        if filename:
            self.proc_output_edit.setText(filename)

    def toggle_proc_api_visibility(self, checked):
        if checked:
            self.proc_api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.proc_api_key_edit.setEchoMode(QLineEdit.Password)

    def browse_mm_image_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select image folder",
            "",
        )
        if directory:
            self.mm_image_dir_edit.setText(directory)

    def browse_mm_output(self):
        # Get repo root (go up from App/SynthMaxxer to repo root)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        default_path = os.path.join(repo_root, "outputs")
        if not os.path.exists(default_path):
            os.makedirs(default_path, exist_ok=True)
        
        current_path = self.mm_output_edit.text().strip()
        if current_path and os.path.dirname(current_path):
            start_dir = os.path.dirname(current_path)
        else:
            start_dir = default_path
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select output Parquet file",
            os.path.join(start_dir, "image_captions.parquet"),
            "Parquet files (*.parquet);;All files (*.*)",
        )
        if filename:
            self.mm_output_edit.setText(filename)

    def toggle_mm_api_visibility(self, checked):
        if checked:
            self.mm_api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.mm_api_key_edit.setEchoMode(QLineEdit.Password)
    
    def toggle_civitai_api_visibility(self, checked):
        if checked:
            self.civitai_api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.civitai_api_key_edit.setEchoMode(QLineEdit.Password)
    
    def browse_civitai_output(self):
        """Browse for Civitai output directory"""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        default_path = os.path.join(repo_root, "Outputs", "images")
        if not os.path.exists(default_path):
            os.makedirs(default_path, exist_ok=True)
        
        current_path = self.civitai_output_edit.text().strip()
        if current_path and os.path.exists(current_path):
            start_dir = current_path
        else:
            start_dir = default_path
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for Civitai images",
            start_dir,
        )
        if directory:
            self.civitai_output_edit.setText(directory)

    def _update_mm_endpoint_placeholder(self, api_type):
        """Update endpoint placeholder based on selected API type"""
        if api_type == "OpenAI Vision":
            endpoint = "https://api.openai.com/v1/chat/completions"
            self.mm_endpoint_edit.setPlaceholderText(endpoint)
            self.mm_endpoint_edit.setText(endpoint)
        elif api_type == "Anthropic Claude":
            endpoint = "https://api.anthropic.com/v1/messages"
            self.mm_endpoint_edit.setPlaceholderText(endpoint)
            self.mm_endpoint_edit.setText(endpoint)
        elif api_type == "Grok (xAI)":
            endpoint = "https://api.x.ai/v1/chat/completions"
            self.mm_endpoint_edit.setPlaceholderText(endpoint)
            self.mm_endpoint_edit.setText(endpoint)
        elif api_type == "OpenRouter":
            endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self.mm_endpoint_edit.setPlaceholderText(endpoint)
            self.mm_endpoint_edit.setText(endpoint)

    def _update_mm_api_key_for_type(self, api_type):
        """Update API key field when API type changes"""
        config_file = CONFIG_FILE
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                api_keys = cfg.get("api_keys", {})
                api_key = api_keys.get(api_type, "")
                if api_key:
                    self.mm_api_key_edit.setText(api_key)
            except Exception:
                pass

    def _refresh_mm_models(self):
        """Manually refresh models for the current API type in multimodal tab"""
        api_type = self.mm_api_type_combo.currentText()
        api_key = self.mm_api_key_edit.text().strip()
        self._update_mm_models(api_type, api_key)

    def _update_mm_models(self, api_type, api_key):
        """Fetch and update model dropdown based on selected API type for multimodal tab"""
        if not api_key:
            self.mm_model_combo.clear()
            self.mm_model_combo.addItem("(Enter API key and click Refresh)")
            self.mm_model_combo.setEnabled(True)
            self.mm_refresh_models_button.setEnabled(True)
            return

        self.mm_model_combo.clear()
        self.mm_model_combo.addItem("Loading models...")
        self.mm_model_combo.setEnabled(False)
        self.mm_refresh_models_button.setEnabled(False)

        # Fetch models in a separate thread
        thread = threading.Thread(
            target=self._fetch_mm_models,
            args=(api_type, api_key),
            daemon=True
        )
        thread.start()

    def _fetch_mm_models(self, api_type, api_key):
        """Fetch available models from the API for multimodal tab"""
        models = []
        error_msg = None

        try:
            if api_type == "Anthropic Claude":
                # Anthropic doesn't have a public models endpoint, use common models
                models = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ]
            elif api_type == "OpenAI Vision":
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            # Filter for vision-capable models
                            if any(x in model_id for x in ["gpt-4o", "gpt-4-vision"]):
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"OpenAI API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing OpenAI response: {str(e)}"
                if not models:
                    models = ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"]
            elif api_type == "Grok (xAI)":
                if api_key:
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"}
                        resp = requests.get("https://api.x.ai/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        items = data.get("data") or data.get("models") or data.get("model_list") or []
                        if isinstance(items, list):
                            for m in items:
                                if isinstance(m, dict):
                                    mid = m.get("id") or m.get("name") or m.get("model_id")
                                    if mid and isinstance(mid, str):
                                        models.append(mid)
                        elif isinstance(data, list):
                            for m in data:
                                if isinstance(m, dict):
                                    mid = m.get("id") or m.get("name") or m.get("model_id")
                                    if mid and isinstance(mid, str):
                                        models.append(mid)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"Grok API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing Grok response: {str(e)}"
                if not models:
                    # Default to grok-4.1-fast-non-reasoning (has vision, latest, faster, cheaper)
                    models = ["grok-4.1-fast-non-reasoning"]
            elif api_type == "OpenRouter":
                if api_key:
                    try:
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "HTTP-Referer": "https://github.com/ShareGPT-Formaxxing",
                            "X-Title": "SynthMaxxer"
                        }
                        resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
                        resp.raise_for_status()
                        data = resp.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            if model_id:
                                models.append(model_id)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"OpenRouter API error: {str(e)}"
                    except Exception as e:
                        error_msg = f"Error parsing OpenRouter response: {str(e)}"
                if not models:
                    models = ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"]
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"

        # Update UI on main thread
        if not models and not error_msg:
            models = ["(No models found - enter manually)"]
        elif error_msg:
            models = [f"(Error: {error_msg[:50]})"]

        # Emit signal to update UI on main thread
        self.mm_model_fetcher.models_ready.emit(models)

    def _on_mm_models_ready(self, models):
        """Handle models ready signal on main thread for multimodal tab"""
        try:
            self.mm_model_combo.clear()
            self.mm_model_combo.addItems(models)
            self.mm_model_combo.setEnabled(True)
            self.mm_refresh_models_button.setEnabled(True)
            if models and not models[0].startswith("(Error:") and models[0] != "(No models found - enter manually)":
                self.mm_model_combo.setCurrentIndex(0)
        except Exception as e:
            try:
                self.mm_model_combo.clear()
                self.mm_model_combo.addItem(f"(UI Error: {str(e)})")
                self.mm_model_combo.setEnabled(True)
                self.mm_refresh_models_button.setEnabled(True)
            except:
                pass

    def _toggle_hf_dataset_mode(self, checked):
        """Enable/disable image folder input based on HF dataset mode"""
        if checked:
            self.mm_image_dir_edit.setEnabled(False)
            self.mm_hf_dataset_edit.setEnabled(True)
            self.mm_hf_token_edit.setEnabled(True)
        else:
            self.mm_image_dir_edit.setEnabled(True)
            self.mm_hf_dataset_edit.setEnabled(False)
            self.mm_hf_token_edit.setEnabled(False)

    def toggle_mm_hf_token_visibility(self, checked):
        if checked:
            self.mm_hf_token_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.mm_hf_token_edit.setEchoMode(QLineEdit.Password)

    def _append_mm_log(self, text):
        if not text:
            return
        cursor = self.mm_log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.mm_log_view.setTextCursor(cursor)
        ansi_stripped = self.strip_ansi(text)
        lines = ansi_stripped.split('\n')
        for line in lines:
            if line.strip():
                fmt = self._get_log_format(line)
                cursor.insertText(line + '\n', fmt)
            else:
                cursor.insertText('\n')
        self.mm_log_view.ensureCursorVisible()

    def _show_preview_image(self, pil_image):
        """Display a preview image in a popup window"""
        try:
            from PIL import Image as PILImage
            
            if not pil_image or not isinstance(pil_image, PILImage.Image):
                return
            
            # Convert PIL Image to QPixmap
            data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
            qim = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qim)
            
            # Create preview window
            preview_window = QMainWindow(self)
            preview_window.setWindowTitle("🖼️ Captioned Image Preview")
            preview_window.resize(800, 600)
            
            # Apply dark theme
            preview_window.setStyleSheet("""
                QMainWindow {
                    background-color: #000000;
                }
                QLabel {
                    background-color: #000000;
                }
            """)
            
            # Create central widget with image
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            layout.setContentsMargins(10, 10, 10, 10)
            
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            # Scale image to fit window while maintaining aspect ratio
            scaled_pix = pix.scaled(780, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pix)
            
            layout.addWidget(image_label)
            preview_window.setCentralWidget(central_widget)
            
            # Show the window
            preview_window.show()
            preview_window.raise_()
            preview_window.activateWindow()
            
        except Exception as e:
            self._append_mm_log(f"Could not display preview image: {str(e)}")

    def _append_proc_log(self, text):
        if not text:
            return
        cursor = self.proc_log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.proc_log_view.setTextCursor(cursor)
        ansi_stripped = self.strip_ansi(text)
        lines = ansi_stripped.split('\n')
        for line in lines:
            if line.strip():
                fmt = self._get_log_format(line)
                cursor.insertText(line + '\n', fmt)
            else:
                cursor.insertText('\n')
        self.proc_log_view.ensureCursorVisible()

    def process_files(self):
        input_file = self.proc_input_edit.text().strip()
        output_file = self.proc_output_edit.text().strip()
        api_key = self.proc_api_key_edit.text().strip()
        system_prompt = self.proc_system_prompt_edit.toPlainText().strip()
        start_line_str = self.proc_start_line_edit.text().strip()
        end_line_str = self.proc_end_line_edit.text().strip()
        do_rewrite = self.proc_rewrite_check.isChecked()
        reply_in_character = self.proc_reply_in_character_check.isChecked()
        dynamic_names_mode = self.proc_dynamic_names_check.isChecked()
        num_new_str = self.proc_num_new_edit.text().strip()
        num_new = 0
        if num_new_str:
            try:
                num_new = int(num_new_str)
                if num_new < 0:
                    raise ValueError("Negative value")
            except ValueError:
                self._show_error("Number of new entries must be a non-negative integer.")
                return
        extra_pairs = int(self.proc_extra_pairs_spin.value())

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
            
            self.proc_output_edit.setText(output_file)
            self._append_proc_log(f"Auto-generated output file: {output_file}")

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

        model_name = self.proc_model_edit.text().strip()
        if not model_name:
            model_name = "grok-4.1-fast-non-reasoning"

        # Reset UI state for new run
        self.proc_log_view.clear()
        self._append_proc_log("=== New run started ===")
        self.setWindowTitle(f"{APP_TITLE} - Processing...")
        self.process_button.setEnabled(False)
        
        # Save config before processing
        self._save_config()

        self.proc_queue = queue.Queue()
        
        # Immediately log that we're starting
        self._append_proc_log("Creating worker thread...")
        self._append_proc_log(f"Input: {input_file or 'None'}")
        self._append_proc_log(f"Output: {output_file}")
        self._append_proc_log(f"API Key: {'***' + api_key[-4:] if len(api_key) > 4 else 'None'}")
        self._append_proc_log(f"Model: {model_name}")
        self._append_proc_log(f"Settings: rewrite={do_rewrite}, extra_pairs={extra_pairs}, num_new={num_new}")

        def worker_wrapper(*args, **kwargs):
            """Wrapper to catch any exceptions during thread startup"""
            try:
                self.proc_queue.put(("log", "Worker thread function called"))
                processing_worker(*args, **kwargs)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = f"CRITICAL: Worker thread crashed: {e}"
                self.proc_queue.put(("error", error_msg))
                self.proc_queue.put(("log", f"Traceback:\n{tb}"))
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
                self.proc_queue,
            ),
            daemon=True,
        )
        self._append_proc_log("Starting worker thread...")
        t.start()
        self._append_proc_log("Worker thread started, timer starting...")
        self.timer.start()
        self._append_proc_log("Timer started - queue should be checked every 100ms")

    def generate_human_cache(self):
        api_key = self.proc_api_key_edit.text().strip()
        system_prompt = self.proc_system_prompt_edit.toPlainText().strip()
        model_name = self.proc_model_edit.text().strip() or "grok-4.1-fast-non-reasoning"

        if not api_key:
            self._show_error("Please enter your Grok API key.")
            return

        cache_file = GLOBAL_HUMAN_CACHE_FILE
        
        # Ask user how many to generate
        from PyQt5.QtWidgets import QInputDialog
        num_turns, ok = QInputDialog.getInt(
            self,
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
        self.proc_log_view.clear()
        self._append_proc_log("=== Generating Human Cache ===")
        self.setWindowTitle(f"{APP_TITLE} - Generating cache...")
        self.generate_cache_button.setEnabled(False)
        self.improve_cache_button.setEnabled(False)

        self.proc_queue = queue.Queue()

        def worker_thread():
            try:
                from App.SynthMaxxer.llm_helpers import generate_human_turns
                from xai_sdk import Client
                import json
                import os

                client = Client(api_key=api_key, timeout=300)
                self.proc_queue.put(("log", f"Generating {num_turns} human turns..."))
                
                new_turns = generate_human_turns(
                    client,
                    model_name,
                    system_prompt,
                    num_turns=num_turns,
                    temperature=1.0
                )
                
                # Load existing cache if it exists
                existing_turns = []
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            existing_turns = json.load(f)
                        self.proc_queue.put(("log", f"Loaded {len(existing_turns)} existing turns from cache"))
                    except Exception as e:
                        self.proc_queue.put(("log", f"Could not load existing cache: {e}"))
                
                # Combine and save
                all_turns = existing_turns + new_turns
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(all_turns, f, indent=2, ensure_ascii=False)
                
                self.proc_queue.put(("log", f"✅ Generated {len(new_turns)} new human turns"))
                self.proc_queue.put(("log", f"✅ Cache now contains {len(all_turns)} total human turns"))
                self.proc_queue.put(("success", f"Successfully generated {len(new_turns)} human turns"))
                
            except Exception as e:
                self.proc_queue.put(("error", f"Failed to generate human cache: {str(e)}"))
            finally:
                self.proc_queue.put(("stopped", None))

        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        self.timer.start()

    def improve_human_cache(self):
        api_key = self.proc_api_key_edit.text().strip()
        system_prompt = self.proc_system_prompt_edit.toPlainText().strip()
        model_name = self.proc_model_edit.text().strip() or "grok-4.1-fast-non-reasoning"

        if not api_key:
            self._show_error("Please enter your Grok API key.")
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
                self._append_proc_log(f"Loaded {len(human_turns)} human turns from {cache_file}")
            else:
                human_turns = []
                self._append_proc_log(f"No cache file found at {cache_file}, starting empty")

            if not human_turns:
                # Don't block - just inform user, but allow them to proceed if they want
                reply = QMessageBox.question(
                    self,
                    "Empty Cache",
                    "Human cache is empty. Would you like to generate some human turns first, or proceed with improvement?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    # Generate some human turns first
                    self.proc_log_view.clear()
                    self._append_proc_log("Generating initial human turns for cache...")
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
                        self._append_proc_log(f"Generated and saved {len(human_turns)} human turns to cache")
                        self._show_info(f"Generated {len(human_turns)} human turns. You can now improve them.")
                        # Continue with improvement after generation
                    except Exception as gen_e:
                        self._show_error(f"Failed to generate human turns: {gen_e}")
                        return
                else:
                    self._show_info("Cannot improve empty cache. Please generate entries first or generate human turns.")
                    return

            # Reset UI state
            self.proc_log_view.clear()
            self._append_proc_log("=== Improving human cache ===")
            self.setWindowTitle(f"{APP_TITLE} - Improving cache...")
            self.process_button.setEnabled(False)
            self.improve_cache_button.setEnabled(False)

            self.proc_queue = queue.Queue()

            def worker_thread():
                try:
                    import concurrent.futures
                    client = Client(api_key=api_key, timeout=300)
                    
                    BATCH_SIZE = self.proc_batch_size_spin.value()
                    MAX_CONCURRENCY = self.proc_concurrency_spin.value()
                    original_human_turns = human_turns.copy()
                    num_batches = (len(original_human_turns) + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    self.proc_queue.put(("log", f"Processing {len(original_human_turns)} human turns in {num_batches} batches of {BATCH_SIZE} (concurrency: {MAX_CONCURRENCY})..."))
                    
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
                        if system_prompt.strip() and self.proc_reply_in_character_check.isChecked():
                            prompt = system_prompt.strip() + "\n\n" + base_prompt

                        try:
                            self.proc_queue.put(("log", f"Sending batch {batch_idx + 1}/{num_batches} ({len(batch)} items)..."))
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
                            
                            self.proc_queue.put(("log", f"Batch {batch_idx + 1}/{num_batches}: got {len(batch_improved)} items, {skipped} skipped, using {len(valid_batch)} valid"))
                            return valid_batch
                        except Exception as e:
                            self.proc_queue.put(("log", f"Batch {batch_idx + 1} failed: {str(e)}"))
                            return batch

                    improved_turns = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
                        batch_futures = {
                            executor.submit(process_batch, i): i 
                            for i in range(num_batches)
                        }
                        
                        for future in concurrent.futures.as_completed(batch_futures):
                            batch_idx = batch_futures[future]
                            try:
                                valid_batch = future.result()
                                improved_turns.extend(valid_batch)
                            except Exception as e:
                                self.proc_queue.put(("log", f"Batch {batch_idx + 1} exception: {str(e)}"))

                    # Save updated cache
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(improved_turns, f, indent=2, ensure_ascii=False)
                    
                    self.proc_queue.put(("log", f"Successfully rewrote {len(improved_turns)} human turns"))
                    if improved_turns and original_human_turns:
                        self.proc_queue.put(("log", f"Sample before: {original_human_turns[0]['message'][:100]}..."))
                        self.proc_queue.put(("log", f"Sample after:  {improved_turns[0]['message'][:100]}..."))
                    self.proc_queue.put(("success", f"Human cache improved! {len(improved_turns)} turns updated in {cache_file} (concurrency: {MAX_CONCURRENCY})"))

                except Exception as e:
                    self.proc_queue.put(("error", f"Cache improvement failed: {str(e)}"))

            t = threading.Thread(target=worker_thread, daemon=True)
            t.start()
            self.timer.start()

        except ImportError as e:
            self._show_error(f"Missing dependency: {str(e)}")
        except Exception as e:
            self._show_error(f"Setup error: {str(e)}")

    def start_image_captioning(self):
        use_hf_dataset = self.mm_use_hf_dataset_check.isChecked()
        hf_dataset = self.mm_hf_dataset_edit.text().strip() if use_hf_dataset else None
        hf_token = self.mm_hf_token_edit.text().strip() if use_hf_dataset else None
        image_dir = self.mm_image_dir_edit.text().strip() if not use_hf_dataset else None
        output_file = self.mm_output_edit.text().strip()
        api_key = self.mm_api_key_edit.text().strip()
        endpoint = self.mm_endpoint_edit.text().strip()
        model = self.mm_model_combo.currentText().strip()
        api_type = self.mm_api_type_combo.currentText()
        caption_prompt = self.mm_caption_prompt_edit.toPlainText().strip()
        max_tokens = self.mm_max_tokens_spin.value()
        temperature = self.mm_temperature_spin.value()
        batch_size = self.mm_batch_size_spin.value()
        max_captions = self.mm_max_captions_spin.value()

        if use_hf_dataset:
            if not hf_dataset:
                self._show_error("Please enter a HuggingFace dataset name.")
                return
        else:
            if not image_dir:
                self._show_error("Please select an image folder.")
                return
            if not os.path.isdir(image_dir):
                self._show_error("Image folder path is invalid.")
                return
        
        if not api_key:
            self._show_error("Please enter your API key.")
            return
        if not endpoint:
            self._show_error("Please enter the API endpoint.")
            return
        if not model:
            self._show_error("Please enter the model name.")
            return

        # Validate model is selected
        if model == "(Click Refresh to load models)" or not model or model.startswith("("):
            self._show_error("Please select a model. Click 'Refresh' to load available models.")
            return

        # Normalize output file path (fix any App\outputs paths)
        if output_file and ("App\\outputs" in output_file or "App/outputs" in output_file):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            outputs_dir = os.path.join(repo_root, "outputs")
            filename = os.path.basename(output_file)
            output_file = os.path.join(outputs_dir, filename)
            self.mm_output_edit.setText(output_file)
        
        # Auto-generate output filename if not provided
        if not output_file:
            # Get repo root (go up from App/SynthMaxxer to repo root)
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            outputs_dir = os.path.join(repo_root, "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(outputs_dir, f"image_captions_{timestamp}.parquet")
            self.mm_output_edit.setText(output_file)

        # Save config
        self._save_config()

        # Reset UI state
        self.mm_log_view.clear()
        self._append_mm_log("=== Image Captioning started ===")
        self.setWindowTitle(f"{APP_TITLE} - Captioning...")
        self.mm_start_button.setEnabled(False)
        self.mm_stop_button.setEnabled(True)
        self.mm_stop_flag = threading.Event()

        self.mm_queue = queue.Queue()
        
        # Log path correction if it happened
        if output_file and ("App\\outputs" not in output_file and "App/outputs" not in output_file):
            # Path is correct, log it
            self._append_mm_log(f"Output file: {output_file}")

        # Start worker thread
        self.mm_worker_thread = threading.Thread(
            target=self._image_captioning_worker,
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
                self.mm_stop_flag,
                hf_dataset,
                hf_token,
                self.mm_queue,
            ),
            daemon=True,
        )
        self.mm_worker_thread.start()
        self.timer.start()

    def stop_image_captioning(self):
        if hasattr(self, 'mm_stop_flag'):
            self.mm_stop_flag.set()
        self._append_mm_log("Stopping image captioning...")
        self.mm_stop_button.setEnabled(False)
    
    def start_civitai_download(self):
        """Start Civitai image download"""
        api_key = self.civitai_api_key_edit.text().strip()
        output_dir = self.civitai_output_edit.text().strip()
        max_images = self.civitai_max_images_spin.value()
        min_width = self.civitai_min_width_spin.value()
        min_height = self.civitai_min_height_spin.value()
        nsfw_level_text = self.civitai_nsfw_combo.currentText()
        sort_mode = self.civitai_sort_combo.currentText()
        include_terms_text = self.civitai_include_edit.text().strip()
        exclude_terms_text = self.civitai_exclude_edit.text().strip()
        save_meta_jsonl = self.civitai_save_meta_check.isChecked()
        batch_size = self.civitai_batch_size_spin.value()
        max_empty_batches = self.civitai_max_empty_batches_spin.value()
        wait_time = self.civitai_wait_time_spin.value()
        
        # Parse NSFW level
        nsfw_mapping = {
            "Any (no filter)": None,
            "None (SFW only)": "None",
            "Soft": "Soft",
            "Mature": "Mature",
            "X (explicit)": "X",
        }
        nsfw_level = nsfw_mapping.get(nsfw_level_text, None)
        
        # Parse include/exclude terms
        include_terms = [x.strip().lower() for x in include_terms_text.split(",") if x.strip()]
        exclude_terms = [x.strip().lower() for x in exclude_terms_text.split(",") if x.strip()]
        
        if not api_key:
            self._show_error("Please enter your Civitai API key.")
            return
        
        if not output_dir:
            # Use default if empty
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            output_dir = os.path.join(repo_root, "Outputs", "images")
            self.civitai_output_edit.setText(output_dir)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        self._save_config()
        
        # Reset UI state
        self.mm_log_view.clear()
        self._append_mm_log("=== Civitai Image Download started ===")
        self.setWindowTitle(f"{APP_TITLE} - Civitai Downloading...")
        self.civitai_start_button.setEnabled(False)
        self.civitai_stop_button.setEnabled(True)
        self.civitai_stop_flag = threading.Event()
        
        self.civitai_queue = queue.Queue()
        
        # Start worker thread
        self.civitai_worker_thread = threading.Thread(
            target=self._civitai_download_worker,
            args=(
                api_key,
                output_dir,
                max_images,
                min_width,
                min_height,
                nsfw_level,
                include_terms,
                exclude_terms,
                sort_mode,
                save_meta_jsonl,
                self.civitai_stop_flag,
                self.civitai_queue,
                batch_size,
                max_empty_batches,
                wait_time,
            ),
            daemon=True,
        )
        self.civitai_worker_thread.start()
        self.timer.start()
    
    def stop_civitai_download(self):
        """Stop Civitai image download"""
        if hasattr(self, 'civitai_stop_flag'):
            self.civitai_stop_flag.set()
        self._append_mm_log("Stopping Civitai download...")
        self.civitai_stop_button.setEnabled(False)
    
    def _civitai_download_worker(
        self,
        api_key,
        output_dir,
        max_images,
        min_width,
        min_height,
        nsfw_level,
        include_terms,
        exclude_terms,
        sort_mode,
        save_meta_jsonl,
        stop_flag,
        q,
        batch_size,
        max_empty_batches,
        wait_time,
    ):
        """Worker function for Civitai image download"""
        try:
            from App.SynthMaxxer.multimodal_worker import civitai_image_download_worker
            civitai_image_download_worker(
                api_key,
                output_dir,
                max_images,
                min_width,
                min_height,
                nsfw_level,
                include_terms,
                exclude_terms,
                sort_mode,
                save_meta_jsonl,
                stop_flag,
                q,
                batch_size,
                max_empty_batches,
                wait_time,
            )
        except Exception as e:
            import traceback
            if q:
                q.put(("error", f"Civitai download worker error: {str(e)}"))
                q.put(("log", traceback.format_exc()))
                q.put(("stopped", "Error occurred"))

    def _image_captioning_worker(
        self,
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
        stop_flag,
        hf_dataset,
        hf_token,
        q,
    ):
        """Worker function for image captioning"""
        try:
            from App.SynthMaxxer.multimodal_worker import image_captioning_worker
            image_captioning_worker(
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
                stop_flag,
                hf_dataset,
                hf_token,
                q,
            )
        except Exception as e:
            import traceback
            error_msg = f"Image captioning worker error: {str(e)}"
            if q:
                q.put(("error", error_msg))
                q.put(("log", f"Traceback:\n{traceback.format_exc()}"))
            print(f"IMAGE_CAPTIONING_ERROR: {error_msg}")


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

