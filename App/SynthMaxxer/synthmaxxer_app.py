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

# Import tab modules
from App.SynthMaxxer import synthmaxxer as synthmaxxer_module
from App.SynthMaxxer import grokmaxxer as grokmaxxer_module
from App.SynthMaxxer import captionmaxxer as captionmaxxer_module
from App.SynthMaxxer import civitai as civitai_module
from App.SynthMaxxer import huggingface as huggingface_module

# Import galaxy background widget
from App.Other.BG import GalaxyBackgroundWidget


APP_TITLE = "SynthMaxxer"
ICON_FILE = str(Path(__file__).parent.parent / "Assets" / "icon.ico")
CONFIG_FILE = str(Path(__file__).parent / "synthmaxxer_config.json")
GLOBAL_HUMAN_CACHE_FILE = str(Path(__file__).parent / "global_human_cache.json")


# ============================================================================
# Reusable UI Components - Eliminate duplication between tabs
# ============================================================================

def create_api_config_group(
    api_key_edit_name="api_key_edit",
    endpoint_edit_name="endpoint_edit",
    model_combo_name="model_combo",
    api_type_combo_name="api_type_combo",
    refresh_button_name="refresh_models_button",
    show_key_check_name="show_key_check",
    api_types=None,
    default_endpoint_placeholder="https://api.example.com/v1/messages",
    on_api_type_changed=None,
    on_refresh_clicked=None,
    on_toggle_visibility=None,
    parent=None,
    group_title="🔑 API Configuration"
):
    """
    Create a reusable API Configuration group box to eliminate duplication.
    
    Returns a tuple: (group_box, widget_dict) where widget_dict contains all created widgets
    """
    if api_types is None:
        api_types = [
            "Anthropic Claude",
            "OpenAI Official",
            "OpenAI Chat Completions",
            "OpenAI Text Completions",
            "Grok (xAI)",
            "Gemini (Google)",
            "OpenRouter",
            "DeepSeek"
        ]
    
    api_group = QGroupBox(group_title)
    api_layout = QFormLayout()
    api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
    api_layout.setHorizontalSpacing(10)
    api_layout.setVerticalSpacing(6)
    api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
    api_group.setLayout(api_layout)
    
    widgets = {}
    
    # API Key row
    api_row = QHBoxLayout()
    api_key_edit = QLineEdit()
    api_key_edit.setEchoMode(QLineEdit.Password)
    api_key_edit.setPlaceholderText("Your API key")
    widgets[api_key_edit_name] = api_key_edit
    
    show_key_check = QCheckBox("Show")
    if on_toggle_visibility:
        show_key_check.toggled.connect(on_toggle_visibility)
    widgets[show_key_check_name] = show_key_check
    
    api_row.addWidget(api_key_edit)
    api_row.addWidget(show_key_check)
    api_layout.addRow(QLabel("API Key:"), _wrap_row_helper(api_row))
    
    # Endpoint
    endpoint_edit = QLineEdit()
    endpoint_edit.setPlaceholderText(default_endpoint_placeholder)
    widgets[endpoint_edit_name] = endpoint_edit
    api_layout.addRow(QLabel("API Endpoint:"), endpoint_edit)
    
    # Model row
    model_row = QHBoxLayout()
    model_combo = QComboBox()
    model_combo.setEditable(True)
    model_combo.setPlaceholderText("Select or enter model name")
    model_combo.addItem("(Click Refresh to load models)")
    widgets[model_combo_name] = model_combo
    
    refresh_models_button = QPushButton("Refresh")
    refresh_models_button.setFixedWidth(80)
    refresh_models_button.setToolTip("Refresh available models from API")
    refresh_models_button.setEnabled(True)
    if on_refresh_clicked:
        refresh_models_button.clicked.connect(on_refresh_clicked)
    widgets[refresh_button_name] = refresh_models_button
    
    model_row.addWidget(model_combo)
    model_row.addWidget(refresh_models_button)
    api_layout.addRow(QLabel("Model:"), _wrap_row_helper(model_row))
    
    # API Type
    api_type_combo = QComboBox()
    api_type_combo.addItems(api_types)
    api_type_combo.setToolTip("Select the API format to use")
    if on_api_type_changed:
        api_type_combo.currentTextChanged.connect(on_api_type_changed)
    widgets[api_type_combo_name] = api_type_combo
    api_layout.addRow(QLabel("API Type:"), api_type_combo)
    
    return api_group, widgets


def create_log_view(placeholder_text="Logs will appear here...", max_blocks=1000):
    """
    Create a reusable log view to eliminate duplication.
    
    Returns: (group_box, log_view)
    """
    progress_group = QGroupBox("Run Status")
    progress_layout = QVBoxLayout()
    progress_layout.setSpacing(6)
    progress_group.setLayout(progress_layout)
    
    log_view = QPlainTextEdit()
    log_view.setReadOnly(True)
    log_view.setMaximumBlockCount(max_blocks)
    log_view.setPlaceholderText(placeholder_text)
    
    progress_layout.addWidget(QLabel("Logs:"))
    progress_layout.addWidget(log_view, stretch=1)
    
    return progress_group, log_view


def create_file_browse_row(
    line_edit_name,
    placeholder_text="",
    default_text="",
    browse_button_text="Browse",
    on_browse_clicked=None,
    widgets_dict=None
):
    """
    Create a reusable file/folder browse row to eliminate duplication.
    
    Returns: (row_layout, widgets_dict) where widgets_dict contains line_edit and browse_button
    """
    if widgets_dict is None:
        widgets_dict = {}
    
    row = QHBoxLayout()
    line_edit = QLineEdit()
    if placeholder_text:
        line_edit.setPlaceholderText(placeholder_text)
    if default_text:
        line_edit.setText(default_text)
    widgets_dict[line_edit_name] = line_edit
    
    browse_btn = QPushButton(browse_button_text)
    browse_btn.setFixedWidth(80)
    if on_browse_clicked:
        browse_btn.clicked.connect(on_browse_clicked)
    widgets_dict[f"{line_edit_name}_browse"] = browse_btn
    
    row.addWidget(line_edit)
    row.addWidget(browse_btn)
    
    return row, widgets_dict


def _wrap_row_helper(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


class ModelFetcher(QObject):
    """Helper class to emit signals from background thread"""
    models_ready = pyqtSignal(list)

class MultimodalModelFetcher(QObject):
    """Helper class to emit signals from background thread for CaptionMaxxer tab"""
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
    
    def resizeEvent(self, event):
        """Handle window resize to update background widget"""
        super().resizeEvent(event)
        if hasattr(self, 'galaxy_bg') and hasattr(self, '_central_widget'):
            self.galaxy_bg.resize(self._central_widget.size())
    
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
                background-color: #05050F;
            }
            QWidget {
                background-color: transparent;
                color: #F9FAFB;
                font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
                font-size: 12pt;
            }
            QLabel {
                color: #E5E7EB;
            }
            QGroupBox {
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                margin-top: 18px;
                padding: 10px;
                background-color: rgba(5, 5, 15, 180);
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
                background-color: rgba(5, 5, 15, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 4px;
                padding: 4px 6px;
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QLineEdit::placeholder {
                color: #6B7280;
            }
            QPlainTextEdit {
                background-color: rgba(2, 2, 10, 220);
                color: #D1D5DB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 8px;
                font-family: Consolas, "Fira Code", monospace;
                font-size: 12px;
                padding: 6px;
            }
            QPushButton {
                background-color: rgba(2, 6, 23, 200);
                color: #F9FAFB;
                border: 1px solid rgba(31, 41, 55, 200);
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(17, 24, 39, 220);
            }
            QPushButton:pressed {
                background-color: rgba(3, 7, 18, 240);
            }
            QPushButton:disabled {
                color: #6B7280;
                border-color: rgba(17, 24, 39, 200);
                background-color: rgba(2, 2, 2, 200);
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
                background-color: rgba(5, 5, 15, 240);
                border: 1px solid rgba(31, 41, 55, 200);
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QTabWidget::pane {
                border: 1px solid rgba(31, 41, 55, 200);
                background-color: rgba(5, 5, 15, 180);
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: rgba(5, 5, 15, 200);
                color: #9CA3AF;
                border: 1px solid rgba(31, 41, 55, 200);
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: rgba(5, 5, 15, 180);
                color: #F9FAFB;
                border-color: #2563EB;
            }
            QTabBar::tab:hover {
                background-color: rgba(17, 24, 39, 200);
                color: #E5E7EB;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Create galaxy background widget
        self.galaxy_bg = GalaxyBackgroundWidget(central)
        self.galaxy_bg.lower()  # Put it behind everything
        
        # Store reference for resize handling
        self._central_widget = central
        
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        central.setLayout(root_layout)
        
        # Initial resize - use a timer to ensure it happens after layout is complete
        QTimer.singleShot(100, lambda: self.galaxy_bg.resize(central.size()) if hasattr(self, 'galaxy_bg') else None)

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

        # API Configuration tab (first tab - central API management)
        self._build_api_config_tab()
        
        # Generation tab (SynthMaxxer)
        generation_tab = synthmaxxer_module.build_synthmaxxer_tab(self)
        self.tabs.addTab(generation_tab, "🔄 SynthMaxxer")
        
        # Processing tab (GrokMaxxer)
        processing_tab = grokmaxxer_module.build_grokmaxxer_tab(self)
        self.tabs.addTab(processing_tab, "⚙️ GrokMaxxer")
        
        # CaptionMaxxer tab
        multimodal_tab = captionmaxxer_module.build_captionmaxxer_tab(self)
        self.tabs.addTab(multimodal_tab, "🖼️ CaptionMaxxer")
        
        # HuggingFace Dataset Download tab
        hf_tab = huggingface_module.build_huggingface_tab(self)
        self.tabs.addTab(hf_tab, "🤗 HuggingFace")
        
        # CivitAI Downloader tab
        civitai_tab = civitai_module.build_civitai_tab(self)
        self.tabs.addTab(civitai_tab, "🎨 CivitAI")

    def _build_api_config_tab(self):
        """Build the centralized API Configuration tab"""
        api_config_tab = QWidget()
        api_config_layout = QVBoxLayout()
        api_config_layout.setContentsMargins(16, 16, 16, 16)
        api_config_layout.setSpacing(12)
        api_config_tab.setLayout(api_config_layout)
        
        api_config_title = QLabel("🔑 API Configuration")
        api_config_title_font = QFont()
        api_config_title_font.setPointSize(16)
        api_config_title_font.setBold(True)
        api_config_title.setFont(api_config_title_font)
        api_config_title.setStyleSheet("color: #F9FAFB; margin-bottom: 10px;")
        api_config_layout.addWidget(api_config_title)
        
        api_config_subtitle = QLabel("Centralized API settings for all tabs - Configure once, use everywhere")
        api_config_subtitle.setStyleSheet("color: #6B7280; font-size: 12pt; margin-bottom: 20px;")
        api_config_layout.addWidget(api_config_subtitle)
        
        api_config_split = QHBoxLayout()
        api_config_split.setSpacing(14)
        api_config_layout.addLayout(api_config_split, stretch=1)
        
        api_config_left_scroll = QScrollArea()
        api_config_left_scroll.setWidgetResizable(True)
        api_config_left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        api_config_left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore
        api_config_left_scroll.setFrameShape(QFrame.NoFrame)
        api_config_left_scroll.viewport().setContentsMargins(0, 0, 0, 0)
        
        api_config_left_container = QWidget()
        api_config_left_container.setMinimumSize(0, 0)
        api_config_left_panel = QVBoxLayout()
        api_config_left_panel.setSpacing(10)
        api_config_left_panel.setContentsMargins(0, 0, 0, 0)
        api_config_left_container.setLayout(api_config_left_panel)
        api_config_left_scroll.setWidget(api_config_left_container)
        
        api_config_split.addWidget(api_config_left_scroll, stretch=3)
        
        api_config_right_container = QWidget()
        api_config_right_panel = QVBoxLayout()
        api_config_right_panel.setSpacing(10)
        api_config_right_container.setLayout(api_config_right_panel)
        
        api_config_split.addWidget(api_config_right_container, stretch=4)
        
        # Build API configuration UI
        self._build_api_config_ui(api_config_left_panel, api_config_right_panel)
        
        # Add API Configuration tab as first tab
        self.tabs.insertTab(0, api_config_tab, "🔑 API Config")

    def _build_api_config_ui(self, left_panel, right_panel):
        """Build the API configuration UI components"""
        # SynthMaxxer API Configuration
        gen_api_group, gen_api_widgets = create_api_config_group(
            api_key_edit_name="api_key_edit",
            endpoint_edit_name="endpoint_edit",
            model_combo_name="model_combo",
            api_type_combo_name="api_type_combo",
            refresh_button_name="refresh_models_button",
            show_key_check_name="show_key_check",
            default_endpoint_placeholder="https://api.example.com/v1/messages (or /v1/chat/completions or /v1/completions)",
            on_api_type_changed=None,
            on_refresh_clicked=self._refresh_models,
            on_toggle_visibility=self.toggle_api_visibility,
            parent=self,
            group_title="🔄 SynthMaxxer"
        )
        # Store widgets as instance attributes
        self.api_key_edit = gen_api_widgets["api_key_edit"]
        self.endpoint_edit = gen_api_widgets["endpoint_edit"]
        self.model_combo = gen_api_widgets["model_combo"]
        self.api_type_combo = gen_api_widgets["api_type_combo"]
        self.refresh_models_button = gen_api_widgets["refresh_models_button"]
        self.show_key_check = gen_api_widgets["show_key_check"]
        
        # Connect API type changes
        self.api_type_combo.currentTextChanged.connect(self._update_endpoint_placeholder)
        self.api_type_combo.currentTextChanged.connect(self._update_api_key_for_type)
        
        left_panel.addWidget(gen_api_group)
        
        # GrokMaxxer API Configuration
        proc_api_group = QGroupBox("⚙️ GrokMaxxer")
        proc_api_layout = QFormLayout()
        proc_api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        proc_api_layout.setHorizontalSpacing(10)
        proc_api_layout.setVerticalSpacing(6)
        proc_api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        proc_api_group.setLayout(proc_api_layout)
        
        proc_api_row = QHBoxLayout()
        self.proc_api_key_edit = QLineEdit()
        self.proc_api_key_edit.setEchoMode(QLineEdit.Password)
        self.proc_api_key_edit.setPlaceholderText("sk-...")
        self.proc_show_key_check = QCheckBox("Show")
        self.proc_show_key_check.toggled.connect(self.toggle_proc_api_visibility)
        proc_api_row.addWidget(self.proc_api_key_edit)
        proc_api_row.addWidget(self.proc_show_key_check)
        proc_api_layout.addRow(QLabel("API Key:"), self._wrap_row(proc_api_row))
        
        self.proc_model_edit = QLineEdit()
        self.proc_model_edit.setPlaceholderText("Enter model name (e.g., grok-beta)")
        self.proc_model_edit.setText("grok-4.1-fast-non-reasoning")
        proc_api_layout.addRow(QLabel("Model:"), self.proc_model_edit)
        
        left_panel.addWidget(proc_api_group)
        
        # CaptionMaxxer API Configuration
        mm_api_group, mm_api_widgets = create_api_config_group(
            api_key_edit_name="mm_api_key_edit",
            endpoint_edit_name="mm_endpoint_edit",
            model_combo_name="mm_model_combo",
            api_type_combo_name="mm_api_type_combo",
            refresh_button_name="mm_refresh_models_button",
            show_key_check_name="mm_show_key_check",
            api_types=["OpenAI Vision", "Anthropic Claude", "Grok (xAI)", "OpenRouter"],
            default_endpoint_placeholder="https://api.openai.com/v1/chat/completions",
            on_api_type_changed=None,
            on_refresh_clicked=self._refresh_mm_models,
            on_toggle_visibility=self.toggle_mm_api_visibility,
            parent=self,
            group_title="🖼️ CaptionMaxxer"
        )
        # Store widgets as instance attributes
        self.mm_api_key_edit = mm_api_widgets["mm_api_key_edit"]
        self.mm_endpoint_edit = mm_api_widgets["mm_endpoint_edit"]
        self.mm_model_combo = mm_api_widgets["mm_model_combo"]
        self.mm_api_type_combo = mm_api_widgets["mm_api_type_combo"]
        self.mm_refresh_models_button = mm_api_widgets["mm_refresh_models_button"]
        self.mm_show_key_check = mm_api_widgets["mm_show_key_check"]
        
        # Connect API type changes
        self.mm_api_type_combo.currentTextChanged.connect(self._update_mm_endpoint_placeholder)
        self.mm_api_type_combo.currentTextChanged.connect(self._update_mm_api_key_for_type)
        
        left_panel.addWidget(mm_api_group)
        
        # Civitai API Configuration
        civitai_api_group = QGroupBox("🎨 Civitai API")
        civitai_api_layout = QFormLayout()
        civitai_api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        civitai_api_layout.setHorizontalSpacing(10)
        civitai_api_layout.setVerticalSpacing(6)
        civitai_api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        civitai_api_group.setLayout(civitai_api_layout)
        
        civitai_api_row = QHBoxLayout()
        self.civitai_api_key_edit = QLineEdit()
        self.civitai_api_key_edit.setEchoMode(QLineEdit.Password)
        self.civitai_api_key_edit.setPlaceholderText("Your Civitai API key")
        self.civitai_show_key_check = QCheckBox("Show")
        self.civitai_show_key_check.toggled.connect(self.toggle_civitai_api_visibility)
        civitai_api_row.addWidget(self.civitai_api_key_edit)
        civitai_api_row.addWidget(self.civitai_show_key_check)
        civitai_api_layout.addRow(QLabel("API Key:"), self._wrap_row(civitai_api_row))
        
        left_panel.addWidget(civitai_api_group)
        
        # HuggingFace API Configuration
        hf_api_group = QGroupBox("🤗 HuggingFace API")
        hf_api_layout = QFormLayout()
        hf_api_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        hf_api_layout.setHorizontalSpacing(10)
        hf_api_layout.setVerticalSpacing(6)
        hf_api_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)  # type: ignore
        hf_api_group.setLayout(hf_api_layout)
        
        hf_token_row = QHBoxLayout()
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        self.hf_token_edit.setPlaceholderText("hf_... (optional, for private/gated datasets)")
        self.hf_show_token_check = QCheckBox("Show")
        self.hf_show_token_check.toggled.connect(self.toggle_hf_token_visibility)
        hf_token_row.addWidget(self.hf_token_edit)
        hf_token_row.addWidget(self.hf_show_token_check)
        hf_api_layout.addRow(QLabel("HF Token:"), self._wrap_row(hf_token_row))
        
        left_panel.addWidget(hf_api_group)
        left_panel.addStretch(1)
        
        # Right panel - Info/Status
        info_group = QGroupBox("ℹ️ Information")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        info_group.setLayout(info_layout)
        
        info_text = QPlainTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(
            "API Configuration\n\n"
            "This tab centralizes all API settings used across SynthMaxxer:\n\n"
            "• SynthMaxxer: Used for conversation generation\n"
            "• GrokMaxxer: Used for entry enhancement and cache generation\n"
            "• CaptionMaxxer: Used for image captioning\n"
            "• Civitai: Used for downloading images from Civitai\n"
            "• HuggingFace: Used for accessing HuggingFace datasets\n\n"
            "Configure your API keys, endpoints, and models here. "
            "These settings are automatically saved and shared across all tabs."
        )
        info_text.setStyleSheet("background-color: #020202; color: #D1D5DB; border: 1px solid #1F2937; border-radius: 8px; padding: 10px;")
        info_layout.addWidget(info_text)
        
        right_panel.addWidget(info_group, stretch=1)


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
                self.output_dir_edit.setText(output_dir)  # type: ignore
                model_name = cfg.get("model", "")
                if model_name:
                    # Try to set the model, add it if not in list (since combo is editable)
                    index = self.model_combo.findText(model_name)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    else:
                        self.model_combo.setCurrentText(model_name)
                self.output_dir_edit.setText(cfg.get("output_dir", ""))  # type: ignore
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
            self.output_dir_edit.setText("outputs")  # type: ignore
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
                self.proc_input_edit.setText(proc_cfg.get("last_input", ""))  # type: ignore
            if proc_cfg.get("last_output"):
                # Default to outputs folder if no saved output
                saved_output = proc_cfg.get("last_output", "")
                if saved_output:
                    self.proc_output_edit.setText(saved_output)  # type: ignore
                else:
                    # Set default to outputs folder
                    repo_root = os.path.dirname(os.path.dirname(__file__))
                    outputs_dir = os.path.join(repo_root, "outputs")
                    default_output = os.path.join(outputs_dir, "processed_output.jsonl")
                    self.proc_output_edit.setText(default_output)  # type: ignore
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
                self.mm_image_dir_edit.setText(cfg.get("mm_image_dir", ""))  # type: ignore
            if cfg.get("mm_output"):
                saved_output = cfg.get("mm_output", "")
                # Fix old paths that point to App\outputs
                if "App\\outputs" in saved_output or "App/outputs" in saved_output:
                    # Replace with correct repo root outputs
                    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    outputs_dir = os.path.join(repo_root, "outputs")
                    filename = os.path.basename(saved_output)
                    saved_output = os.path.join(outputs_dir, filename)
                self.mm_output_edit.setText(saved_output)  # type: ignore
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
            if cfg.get("hf_token"):
                self.hf_token_edit.setText(cfg.get("hf_token", ""))
            # Load HuggingFace upload settings (optional section)
            if hasattr(self, 'mm_upload_group') and "mm_upload_enabled" in cfg:
                self.mm_upload_group.setChecked(bool(cfg.get("mm_upload_enabled", False)))
            if hasattr(self, 'mm_hf_repo_edit') and cfg.get("mm_hf_repo"):
                self.mm_hf_repo_edit.setText(cfg.get("mm_hf_repo", ""))
            if hasattr(self, 'mm_private_repo_check') and "mm_private_repo" in cfg:
                self.mm_private_repo_check.setChecked(bool(cfg.get("mm_private_repo", True)))
            if hasattr(self, 'mm_shard_size_spin') and "mm_shard_size" in cfg:
                self.mm_shard_size_spin.setValue(int(cfg.get("mm_shard_size", 1000)))
            if hasattr(self, 'mm_upload_batch_spin') and "mm_upload_batch_size" in cfg:
                self.mm_upload_batch_spin.setValue(int(cfg.get("mm_upload_batch_size", 2000)))
            if hasattr(self, 'mm_resume_upload_check') and "mm_resume_upload" in cfg:
                self.mm_resume_upload_check.setChecked(bool(cfg.get("mm_resume_upload", True)))
        
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
            "output_dir": self.output_dir_edit.text().strip(),  # type: ignore
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
            "proc_last_input": self.proc_input_edit.text().strip(),  # type: ignore
            "proc_last_output": self.proc_output_edit.text().strip(),  # type: ignore
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
            # CaptionMaxxer tab config
            "mm_image_dir": self.mm_image_dir_edit.text().strip() if hasattr(self, 'mm_image_dir_edit') else "",  # type: ignore
            "mm_output": self.mm_output_edit.text().strip() if hasattr(self, 'mm_output_edit') else "",  # type: ignore
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
            "hf_token": self.hf_token_edit.text().strip() if hasattr(self, 'hf_token_edit') else "",
            # HuggingFace Upload settings (optional section)
            "mm_upload_enabled": self.mm_upload_group.isChecked() if hasattr(self, 'mm_upload_group') else False,
            "mm_hf_repo": self.mm_hf_repo_edit.text().strip() if hasattr(self, 'mm_hf_repo_edit') else "",
            "mm_private_repo": self.mm_private_repo_check.isChecked() if hasattr(self, 'mm_private_repo_check') else True,
            "mm_shard_size": self.mm_shard_size_spin.value() if hasattr(self, 'mm_shard_size_spin') else 1000,
            "mm_upload_batch_size": self.mm_upload_batch_spin.value() if hasattr(self, 'mm_upload_batch_spin') else 2000,
            "mm_resume_upload": self.mm_resume_upload_check.isChecked() if hasattr(self, 'mm_resume_upload_check') else True,
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
                        # Also re-enable upload buttons
                        if hasattr(self, 'mm_upload_button'):
                            self.mm_upload_button.setEnabled(True)
                        if hasattr(self, 'mm_stop_upload_button'):
                            self.mm_stop_upload_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.mm_start_button.setEnabled(True)
                        self.mm_stop_button.setEnabled(False)
                        # Also re-enable upload buttons
                        if hasattr(self, 'mm_upload_button'):
                            self.mm_upload_button.setEnabled(True)
                        if hasattr(self, 'mm_stop_upload_button'):
                            self.mm_stop_upload_button.setEnabled(False)
                        self.timer.stop()
                        self._append_mm_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "preview_image":
                        # Display the preview image
                        self._show_preview_image(msg)
                    elif msg_type == "stopped":
                        self.mm_start_button.setEnabled(True)
                        self.mm_stop_button.setEnabled(False)
                        # Also re-enable upload buttons
                        if hasattr(self, 'mm_upload_button'):
                            self.mm_upload_button.setEnabled(True)
                        if hasattr(self, 'mm_stop_upload_button'):
                            self.mm_stop_upload_button.setEnabled(False)
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
                        self._append_civitai_log(str(msg))
                    elif msg_type == "success":
                        self.setWindowTitle(f"{APP_TITLE} - Done")
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_civitai_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_civitai_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "stopped":
                        self.civitai_start_button.setEnabled(True)
                        self.civitai_stop_button.setEnabled(False)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
                        if msg:  # Only log if there's a message
                            self._append_civitai_log(str(msg))
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
                        if hasattr(self, 'proc_stop_button'):
                            self.proc_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_proc_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.process_button.setEnabled(True)
                        self.generate_cache_button.setEnabled(True)
                        self.improve_cache_button.setEnabled(True)
                        if hasattr(self, 'proc_stop_button'):
                            self.proc_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_proc_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "stopped":
                        self.process_button.setEnabled(True)
                        self.generate_cache_button.setEnabled(True)
                        self.improve_cache_button.setEnabled(True)
                        if hasattr(self, 'proc_stop_button'):
                            self.proc_stop_button.setEnabled(False)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
                        if msg:  # Only log if there's a message
                            self._append_proc_log(str(msg))
            except queue.Empty:
                pass
        
        # Check HuggingFace queue
        if hasattr(self, 'hf_queue') and self.hf_queue:
            try:
                while True:
                    msg_type, msg = self.hf_queue.get_nowait()
                    if msg_type == "log":
                        self._append_hf_log(str(msg))
                    elif msg_type == "success":
                        self.setWindowTitle(f"{APP_TITLE} - Done")
                        self.hf_dataset_start_button.setEnabled(True)
                        self.hf_dataset_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_hf_log(str(msg))
                        self._show_info(str(msg))
                    elif msg_type == "error":
                        self.setWindowTitle(f"{APP_TITLE} - Error")
                        self.hf_dataset_start_button.setEnabled(True)
                        self.hf_dataset_stop_button.setEnabled(False)
                        self.timer.stop()
                        self._append_hf_log(str(msg))
                        self._show_error(str(msg))
                    elif msg_type == "stopped":
                        self.hf_dataset_start_button.setEnabled(True)
                        self.hf_dataset_stop_button.setEnabled(False)
                        self.setWindowTitle(APP_TITLE)
                        self.timer.stop()
                        if msg:  # Only log if there's a message
                            self._append_hf_log(str(msg))
            except queue.Empty:
                pass
        
        # Now check main queue for generation tab
        queue_obj = self.queue
        if not isinstance(queue_obj, queue.Queue):
            return
        
        try:
            while True:
                msg_type, msg = queue_obj.get_nowait()  # type: ignore[misc]
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

    def toggle_proc_api_visibility(self, checked):
        if checked:
            self.proc_api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.proc_api_key_edit.setEchoMode(QLineEdit.Password)

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
    
    def toggle_hf_token_visibility(self, checked):
        if checked:
            self.hf_token_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.hf_token_edit.setEchoMode(QLineEdit.Password)
    
    def _append_hf_log(self, message):
        """Append message to HuggingFace log view"""
        if hasattr(self, 'hf_log_view'):
            self.hf_log_view.appendPlainText(message)
    
    def _append_civitai_log(self, text):
        """Append message to CivitAI log view"""
        if not text:
            return
        if hasattr(self, 'civitai_log_view'):
            cursor = self.civitai_log_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.civitai_log_view.setTextCursor(cursor)
            ansi_stripped = self.strip_ansi(text)
            lines = ansi_stripped.split('\n')
            for line in lines:
                if line.strip():
                    fmt = self._get_log_format(line)
                    cursor.insertText(line + '\n', fmt)
                else:
                    cursor.insertText('\n')
            self.civitai_log_view.ensureCursorVisible()

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
        """Fetch and update model dropdown based on selected API type for CaptionMaxxer tab"""
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
        """Fetch available models from the API for CaptionMaxxer tab"""
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
        """Handle models ready signal on main thread for CaptionMaxxer tab"""
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
            self.mm_image_dir_edit.setEnabled(False)  # type: ignore
            self.mm_hf_dataset_edit.setEnabled(True)  # type: ignore
        else:
            self.mm_image_dir_edit.setEnabled(True)  # type: ignore
            self.mm_hf_dataset_edit.setEnabled(False)  # type: ignore

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
            image_label.setAlignment(Qt.AlignCenter)  # type: ignore
            # Scale image to fit window while maintaining aspect ratio
            scaled_pix = pix.scaled(780, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # type: ignore
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


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

