"""
BehaviorMancer GUI Application

PyQt5-based GUI for model behavior pattern removal.
Detects and removes any behavior pattern you have training data for.
"""

import os
import sys
import json
import threading
import queue
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
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
    QDoubleSpinBox,
    QButtonGroup,
    QRadioButton,
    QTabWidget,
)

# Add parent directory to path for imports
_app_dir = Path(__file__).parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from App.BehaviorMancer.BehaviorMancer import BehaviorMancer, BehaviorMancerConfig
from App.Other.Theme import Theme
from App.Other.BG import GalaxyBackgroundWidget


APP_TITLE = "BehaviorMancer"
ICON_FILE = str(Path(__file__).parent.parent / "Assets" / "icon.ico")
CONFIG_FILE = str(Path(__file__).parent / "behaviormancer_config.json")


class WorkerSignals(QObject):
    """Signals for worker thread communication."""
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    model_info = pyqtSignal(dict)
    test_result = pyqtSignal(str)


class BehaviorMancerApp(QMainWindow):
    """Main application window."""
    
    def __init__(self, theme=None):
        super().__init__()
        
        self.theme = theme or Theme.DARK
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1280, 720)
        
        if os.path.exists(ICON_FILE):
            try:
                self.setWindowIcon(QIcon(ICON_FILE))
            except Exception:
                pass
        
        # Worker thread management
        self.worker_thread = None
        self.behavior_mancer = None
        self.message_queue = queue.Queue()
        
        # Signals
        self.signals = WorkerSignals()
        self.signals.log.connect(self._append_log)
        self.signals.finished.connect(self._on_finished)
        self.signals.model_info.connect(self._on_model_info)
        self.signals.test_result.connect(self._on_test_result)
        
        # Timer for checking message queue
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._check_queue)
        
        self._setup_style()
        self._build_ui()
        self._load_config()
    
    def _setup_style(self):
        """Apply dark theme styling."""
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
            QRadioButton {
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 1px solid #4B5563;
                background-color: #000000;
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                border: 1px solid #2563EB;
                background-color: #2563EB;
                border-radius: 7px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(5, 5, 15, 240);
                border: 1px solid rgba(31, 41, 55, 200);
                selection-background-color: #2563EB;
                selection-color: #F9FAFB;
            }
            QScrollArea {
                border: none;
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
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Galaxy background
        self.galaxy_bg = GalaxyBackgroundWidget(central)
        self.galaxy_bg.lower()
        self._central_widget = central
        
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        central.setLayout(root_layout)
        
        # Header
        header_row = QHBoxLayout()
        title_label = QLabel(APP_TITLE)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        subtitle_label = QLabel("Model Behavior Pattern Removal Tool")
        subtitle_label.setStyleSheet("color: #6B7280; font-size: 12pt;")
        
        title_container = QVBoxLayout()
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)
        
        header_row.addLayout(title_container)
        header_row.addStretch()
        root_layout.addLayout(header_row)
        
        # Tab widget
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, stretch=1)
        
        # Build tabs
        self._build_model_tab()
        self._build_dataset_tab()
        self._build_settings_tab()
        self._build_run_tab()
        
        # Initial resize for background
        QTimer.singleShot(100, lambda: self.galaxy_bg.resize(central.size()) if hasattr(self, 'galaxy_bg') else None)
    
    def _build_model_tab(self):
        """Build the Model Configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        tab.setLayout(layout)
        
        # Model Source Group
        model_group = QGroupBox("🤖 Model Source")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        model_group.setLayout(model_layout)
        
        # Source selection
        source_row = QHBoxLayout()
        self.model_source_group = QButtonGroup(self)
        
        self.model_local_radio = QRadioButton("Local Model")
        self.model_hf_radio = QRadioButton("HuggingFace Hub")
        self.model_local_radio.setChecked(True)
        
        self.model_source_group.addButton(self.model_local_radio, 0)
        self.model_source_group.addButton(self.model_hf_radio, 1)
        
        source_row.addWidget(self.model_local_radio)
        source_row.addWidget(self.model_hf_radio)
        source_row.addStretch()
        model_layout.addLayout(source_row)
        
        # Local path input
        self.model_local_widget = QWidget()
        local_layout = QFormLayout()
        local_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        local_layout.setHorizontalSpacing(10)
        self.model_local_widget.setLayout(local_layout)
        
        path_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to model folder...")
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.setFixedWidth(80)
        model_browse_btn.clicked.connect(self._browse_model)
        path_row.addWidget(self.model_path_edit)
        path_row.addWidget(model_browse_btn)
        local_layout.addRow("Model Path:", self._wrap_row(path_row))
        model_layout.addWidget(self.model_local_widget)
        
        # HuggingFace input
        self.model_hf_widget = QWidget()
        hf_layout = QFormLayout()
        hf_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hf_layout.setHorizontalSpacing(10)
        self.model_hf_widget.setLayout(hf_layout)
        
        self.model_hf_edit = QLineEdit()
        self.model_hf_edit.setPlaceholderText("e.g., meta-llama/Llama-2-7b-hf")
        hf_layout.addRow("Model ID:", self.model_hf_edit)
        
        token_row = QHBoxLayout()
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        self.hf_token_edit.setPlaceholderText("Optional - for gated models")
        self.hf_token_show_check = QCheckBox("Show")
        self.hf_token_show_check.toggled.connect(self._toggle_hf_token_visibility)
        token_row.addWidget(self.hf_token_edit)
        token_row.addWidget(self.hf_token_show_check)
        hf_layout.addRow("HF Token:", self._wrap_row(token_row))
        
        model_layout.addWidget(self.model_hf_widget)
        self.model_hf_widget.hide()
        
        # Model info display
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("color: #9CA3AF; font-size: 11pt; padding: 8px; background-color: rgba(0,0,0,0.3); border-radius: 4px;")
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)
        
        # Connect radio buttons
        self.model_local_radio.toggled.connect(self._on_model_source_changed)
        
        layout.addWidget(model_group)
        
        # Output Group
        output_group = QGroupBox("💾 Output")
        output_layout = QFormLayout()
        output_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_layout.setHorizontalSpacing(10)
        output_group.setLayout(output_layout)
        
        output_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Folder to save modified model...")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.setFixedWidth(80)
        output_browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_path_edit)
        output_row.addWidget(output_browse_btn)
        output_layout.addRow("Output Path:", self._wrap_row(output_row))
        
        layout.addWidget(output_group)
        layout.addStretch()
        
        self.tabs.addTab(tab, "🤖 Model")
    
    def _build_dataset_tab(self):
        """Build the Dataset Configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        tab.setLayout(layout)
        
        # Behavior Dataset Group
        behavior_group = QGroupBox("📊 Behavior Dataset")
        behavior_layout = QVBoxLayout()
        behavior_layout.setSpacing(10)
        behavior_group.setLayout(behavior_layout)
        
        # Info
        info_label = QLabel("Provide contrastive examples: target behavior (to exhibit) vs baseline behavior (to remove)")
        info_label.setStyleSheet("color: #9CA3AF; font-size: 11pt;")
        info_label.setWordWrap(True)
        behavior_layout.addWidget(info_label)
        
        # Source selection
        source_row = QHBoxLayout()
        self.dataset_source_group = QButtonGroup(self)
        
        self.dataset_local_radio = QRadioButton("Local File")
        self.dataset_hf_radio = QRadioButton("HuggingFace")
        self.dataset_local_radio.setChecked(True)
        
        self.dataset_source_group.addButton(self.dataset_local_radio, 0)
        self.dataset_source_group.addButton(self.dataset_hf_radio, 1)
        
        source_row.addWidget(self.dataset_local_radio)
        source_row.addWidget(self.dataset_hf_radio)
        source_row.addStretch()
        behavior_layout.addLayout(source_row)
        
        # Local file input
        self.dataset_local_widget = QWidget()
        local_layout = QFormLayout()
        local_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        local_layout.setHorizontalSpacing(10)
        self.dataset_local_widget.setLayout(local_layout)
        
        file_row = QHBoxLayout()
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("JSONL file with behavior examples...")
        dataset_browse_btn = QPushButton("Browse")
        dataset_browse_btn.setFixedWidth(80)
        dataset_browse_btn.clicked.connect(self._browse_dataset)
        file_row.addWidget(self.dataset_path_edit)
        file_row.addWidget(dataset_browse_btn)
        local_layout.addRow("File:", self._wrap_row(file_row))
        
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Target:"))
        self.local_target_col_edit = QLineEdit("target")
        self.local_target_col_edit.setFixedWidth(100)
        self.local_target_col_edit.setToolTip("Column containing behavior to exhibit")
        col_row.addWidget(self.local_target_col_edit)
        col_row.addSpacing(20)
        col_row.addWidget(QLabel("Baseline:"))
        self.local_baseline_col_edit = QLineEdit("baseline")
        self.local_baseline_col_edit.setFixedWidth(100)
        self.local_baseline_col_edit.setToolTip("Column containing behavior to remove")
        col_row.addWidget(self.local_baseline_col_edit)
        col_row.addStretch()
        local_layout.addRow("Columns:", self._wrap_row(col_row))
        
        behavior_layout.addWidget(self.dataset_local_widget)
        
        # HuggingFace input
        self.dataset_hf_widget = QWidget()
        hf_layout = QFormLayout()
        hf_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hf_layout.setHorizontalSpacing(10)
        self.dataset_hf_widget.setLayout(hf_layout)
        
        self.dataset_hf_edit = QLineEdit()
        self.dataset_hf_edit.setPlaceholderText("e.g., username/behavior-dataset")
        hf_layout.addRow("Dataset ID:", self.dataset_hf_edit)
        
        hf_col_row = QHBoxLayout()
        hf_col_row.addWidget(QLabel("Target:"))
        self.hf_target_col_edit = QLineEdit("target")
        self.hf_target_col_edit.setFixedWidth(100)
        self.hf_target_col_edit.setToolTip("Column containing behavior to exhibit")
        hf_col_row.addWidget(self.hf_target_col_edit)
        hf_col_row.addSpacing(20)
        hf_col_row.addWidget(QLabel("Baseline:"))
        self.hf_baseline_col_edit = QLineEdit("baseline")
        self.hf_baseline_col_edit.setFixedWidth(100)
        self.hf_baseline_col_edit.setToolTip("Column containing behavior to remove")
        hf_col_row.addWidget(self.hf_baseline_col_edit)
        hf_col_row.addStretch()
        hf_layout.addRow("Columns:", self._wrap_row(hf_col_row))
        
        behavior_layout.addWidget(self.dataset_hf_widget)
        self.dataset_hf_widget.hide()
        
        # Connect radio buttons
        self.dataset_local_radio.toggled.connect(self._on_dataset_source_changed)
        
        layout.addWidget(behavior_group)
        
        # Preservation Dataset Group
        preservation_group = QGroupBox("🛡️ Preservation Dataset (Optional)")
        preservation_layout = QVBoxLayout()
        preservation_layout.setSpacing(10)
        preservation_group.setLayout(preservation_layout)
        
        # Info
        pres_info = QLabel("Prompts testing capabilities to preserve (math, coding, reasoning). Required for null-space constraints.")
        pres_info.setStyleSheet("color: #9CA3AF; font-size: 11pt;")
        pres_info.setWordWrap(True)
        preservation_layout.addWidget(pres_info)
        
        # Source selection
        pres_source_row = QHBoxLayout()
        self.preservation_source_group = QButtonGroup(self)
        
        self.preservation_local_radio = QRadioButton("Local File")
        self.preservation_hf_radio = QRadioButton("HuggingFace")
        self.preservation_local_radio.setChecked(True)
        
        self.preservation_source_group.addButton(self.preservation_local_radio, 0)
        self.preservation_source_group.addButton(self.preservation_hf_radio, 1)
        
        pres_source_row.addWidget(self.preservation_local_radio)
        pres_source_row.addWidget(self.preservation_hf_radio)
        pres_source_row.addStretch()
        preservation_layout.addLayout(pres_source_row)
        
        # Local file input
        self.preservation_local_widget = QWidget()
        pres_local_layout = QFormLayout()
        pres_local_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pres_local_layout.setHorizontalSpacing(10)
        self.preservation_local_widget.setLayout(pres_local_layout)
        
        pres_file_row = QHBoxLayout()
        self.preservation_path_edit = QLineEdit()
        self.preservation_path_edit.setPlaceholderText("JSONL/TXT file with preservation prompts...")
        preservation_browse_btn = QPushButton("Browse")
        preservation_browse_btn.setFixedWidth(80)
        preservation_browse_btn.clicked.connect(self._browse_preservation)
        pres_file_row.addWidget(self.preservation_path_edit)
        pres_file_row.addWidget(preservation_browse_btn)
        pres_local_layout.addRow("File:", self._wrap_row(pres_file_row))
        
        pres_col_row = QHBoxLayout()
        pres_col_row.addWidget(QLabel("Prompt column:"))
        self.local_preservation_col_edit = QLineEdit("prompt")
        self.local_preservation_col_edit.setFixedWidth(100)
        self.local_preservation_col_edit.setToolTip("Column containing prompts (leave blank for plain text)")
        pres_col_row.addWidget(self.local_preservation_col_edit)
        pres_col_row.addStretch()
        pres_local_layout.addRow("Column:", self._wrap_row(pres_col_row))
        
        preservation_layout.addWidget(self.preservation_local_widget)
        
        # HuggingFace input
        self.preservation_hf_widget = QWidget()
        pres_hf_layout = QFormLayout()
        pres_hf_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pres_hf_layout.setHorizontalSpacing(10)
        self.preservation_hf_widget.setLayout(pres_hf_layout)
        
        self.preservation_hf_edit = QLineEdit()
        self.preservation_hf_edit.setPlaceholderText("e.g., username/preservation-prompts")
        pres_hf_layout.addRow("Dataset ID:", self.preservation_hf_edit)
        
        pres_hf_col_row = QHBoxLayout()
        pres_hf_col_row.addWidget(QLabel("Prompt column:"))
        self.hf_preservation_col_edit = QLineEdit("prompt")
        self.hf_preservation_col_edit.setFixedWidth(100)
        pres_hf_col_row.addWidget(self.hf_preservation_col_edit)
        pres_hf_col_row.addStretch()
        pres_hf_layout.addRow("Column:", self._wrap_row(pres_hf_col_row))
        
        preservation_layout.addWidget(self.preservation_hf_widget)
        self.preservation_hf_widget.hide()
        
        # Connect radio buttons
        self.preservation_local_radio.toggled.connect(self._on_preservation_source_changed)
        
        layout.addWidget(preservation_group)
        layout.addStretch()
        
        self.tabs.addTab(tab, "📊 Dataset")
    
    def _build_settings_tab(self):
        """Build the Settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        tab.setLayout(layout)
        
        # Split layout for two columns
        split = QHBoxLayout()
        split.setSpacing(16)
        layout.addLayout(split, stretch=1)
        
        # Left column
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        split.addLayout(left_col, stretch=1)
        
        # Basic Settings Group
        basic_group = QGroupBox("⚙️ Basic Settings")
        basic_layout = QFormLayout()
        basic_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        basic_layout.setHorizontalSpacing(12)
        basic_layout.setVerticalSpacing(10)
        basic_group.setLayout(basic_layout)
        
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(5, 500)
        self.n_samples_spin.setValue(30)
        self.n_samples_spin.setToolTip("Number of sample pairs for direction extraction")
        basic_layout.addRow("Sample Pairs:", self.n_samples_spin)
        
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 2.0)
        self.strength_spin.setSingleStep(0.1)
        self.strength_spin.setValue(1.0)
        self.strength_spin.setToolTip("Removal strength (1.0 = full, <1.0 = gentler)")
        basic_layout.addRow("Strength:", self.strength_spin)
        
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float16", "bfloat16", "float32"])
        self.precision_combo.setToolTip("Model precision (float16 is fastest)")
        basic_layout.addRow("Precision:", self.precision_combo)
        
        left_col.addWidget(basic_group)
        
        # Right column
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        split.addLayout(right_col, stretch=1)
        
        # Advanced Options Group
        advanced_group = QGroupBox("🔧 Advanced Options")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(10)
        advanced_group.setLayout(advanced_layout)
        
        self.norm_preservation_check = QCheckBox("Norm Preservation")
        self.norm_preservation_check.setChecked(True)
        self.norm_preservation_check.setToolTip("Keeps weight magnitudes stable (recommended)")
        advanced_layout.addWidget(self.norm_preservation_check)
        
        norm_info = QLabel("Recommended - prevents weight instability")
        norm_info.setStyleSheet("color: #6B7280; font-size: 10pt; margin-left: 20px;")
        advanced_layout.addWidget(norm_info)
        
        self.winsorization_check = QCheckBox("Winsorization")
        self.winsorization_check.setChecked(False)
        self.winsorization_check.setToolTip("Clips outlier activations (useful for Gemma models)")
        advanced_layout.addWidget(self.winsorization_check)
        
        win_info = QLabel("For Gemma models - clips outlier activations")
        win_info.setStyleSheet("color: #6B7280; font-size: 10pt; margin-left: 20px;")
        advanced_layout.addWidget(win_info)
        
        self.null_space_check = QCheckBox("Null-Space Constraints")
        self.null_space_check.setChecked(False)
        self.null_space_check.setToolTip("Preserves other model capabilities (requires preservation dataset)")
        advanced_layout.addWidget(self.null_space_check)
        
        null_info = QLabel("Requires preservation dataset - preserves capabilities")
        null_info.setStyleSheet("color: #6B7280; font-size: 10pt; margin-left: 20px;")
        advanced_layout.addWidget(null_info)
        
        self.adaptive_layers_check = QCheckBox("Adaptive Layer Weighting")
        self.adaptive_layers_check.setChecked(False)
        self.adaptive_layers_check.setToolTip("Focuses removal on middle-to-later layers")
        advanced_layout.addWidget(self.adaptive_layers_check)
        
        adapt_info = QLabel("Gaussian weighting on middle-to-later layers")
        adapt_info.setStyleSheet("color: #6B7280; font-size: 10pt; margin-left: 20px;")
        advanced_layout.addWidget(adapt_info)
        
        right_col.addWidget(advanced_group)
        
        left_col.addStretch()
        right_col.addStretch()
        
        self.tabs.addTab(tab, "⚙️ Settings")
    
    def _build_run_tab(self):
        """Build the Run & Test tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        tab.setLayout(layout)
        
        # Split layout
        split = QHBoxLayout()
        split.setSpacing(16)
        layout.addLayout(split, stretch=1)
        
        # Left - Log output
        log_group = QGroupBox("📋 Run Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Logs will appear here...")
        log_layout.addWidget(self.log_view, stretch=1)
        
        split.addWidget(log_group, stretch=3)
        
        # Right - Test section
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        split.addLayout(right_col, stretch=2)
        
        test_group = QGroupBox("🧪 Test Model")
        test_layout = QVBoxLayout()
        test_layout.setSpacing(10)
        test_group.setLayout(test_layout)
        
        self.test_prompt_edit = QLineEdit()
        self.test_prompt_edit.setPlaceholderText("Enter a prompt to test...")
        test_layout.addWidget(self.test_prompt_edit)
        
        self.test_button = QPushButton("Test Prompt")
        self.test_button.clicked.connect(self._test_prompt)
        self.test_button.setEnabled(False)
        test_layout.addWidget(self.test_button)
        
        self.test_output = QPlainTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setMaximumBlockCount(500)
        self.test_output.setPlaceholderText("Test results appear here...")
        test_layout.addWidget(self.test_output, stretch=1)
        
        right_col.addWidget(test_group)
        
        # Action buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        
        self.load_model_button = QPushButton("Load Model Only")
        self.load_model_button.setMinimumHeight(44)
        self.load_model_button.clicked.connect(self._load_model_only)
        
        self.start_button = QPushButton("🚀 Remove Behavior")
        self.start_button.setMinimumHeight(44)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(37, 99, 235, 200);
                font-weight: bold;
                font-size: 13pt;
            }
            QPushButton:hover {
                background-color: rgba(59, 130, 246, 220);
            }
            QPushButton:disabled {
                background-color: rgba(17, 24, 39, 200);
            }
        """)
        self.start_button.clicked.connect(self._start_removal)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(44)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(220, 38, 38, 200);
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(239, 68, 68, 220);
            }
            QPushButton:disabled {
                background-color: rgba(17, 24, 39, 200);
            }
        """)
        self.stop_button.clicked.connect(self._stop_removal)
        
        button_row.addWidget(self.load_model_button)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        
        layout.addLayout(button_row)
        
        self.tabs.addTab(tab, "🚀 Run")
    
    def _wrap_row(self, layout):
        """Wrap a layout in a widget."""
        container = QWidget()
        container.setLayout(layout)
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        container.setSizePolicy(sp)
        return container
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _on_model_source_changed(self, checked):
        """Handle model source radio button change."""
        if self.model_local_radio.isChecked():
            self.model_local_widget.show()
            self.model_hf_widget.hide()
        else:
            self.model_local_widget.hide()
            self.model_hf_widget.show()
    
    def _on_dataset_source_changed(self, checked):
        """Handle dataset source radio button change."""
        if self.dataset_local_radio.isChecked():
            self.dataset_local_widget.show()
            self.dataset_hf_widget.hide()
        else:
            self.dataset_local_widget.hide()
            self.dataset_hf_widget.show()
    
    def _on_preservation_source_changed(self, checked):
        """Handle preservation source radio button change."""
        if self.preservation_local_radio.isChecked():
            self.preservation_local_widget.show()
            self.preservation_hf_widget.hide()
        else:
            self.preservation_local_widget.hide()
            self.preservation_hf_widget.show()
    
    def _toggle_hf_token_visibility(self, checked):
        """Toggle HuggingFace token visibility."""
        self.hf_token_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
    
    def _browse_model(self):
        """Browse for model folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if folder:
            self.model_path_edit.setText(folder)
    
    def _browse_dataset(self):
        """Browse for dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset File", "", "JSONL files (*.jsonl);;JSON files (*.json);;All files (*)"
        )
        if file_path:
            self.dataset_path_edit.setText(file_path)
    
    def _browse_preservation(self):
        """Browse for preservation prompts file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Preservation File", "", "JSONL files (*.jsonl);;Text files (*.txt);;All files (*)"
        )
        if file_path:
            self.preservation_path_edit.setText(file_path)

    def _browse_output(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)
    
    def _get_config(self) -> BehaviorMancerConfig:
        """Build configuration from UI state."""
        config = BehaviorMancerConfig()
        
        # Model source
        if self.model_local_radio.isChecked():
            config.model_source = "local"
            config.model_path = self.model_path_edit.text().strip()
        else:
            config.model_source = "huggingface"
            config.model_path = self.model_hf_edit.text().strip()
            config.hf_token = self.hf_token_edit.text().strip() or None
        
        # Dataset source
        if self.dataset_local_radio.isChecked():
            config.dataset_source = "local"
            config.dataset_path = self.dataset_path_edit.text().strip()
            config.target_behavior_column = self.local_target_col_edit.text().strip()
            config.baseline_behavior_column = self.local_baseline_col_edit.text().strip()
        else:
            config.dataset_source = "huggingface"
            config.dataset_path = self.dataset_hf_edit.text().strip()
            config.target_behavior_column = self.hf_target_col_edit.text().strip()
            config.baseline_behavior_column = self.hf_baseline_col_edit.text().strip()
        
        # Preservation dataset source
        if self.preservation_local_radio.isChecked():
            config.preservation_source = "local"
            config.preservation_path = self.preservation_path_edit.text().strip()
            config.preservation_column = self.local_preservation_col_edit.text().strip()
        else:
            config.preservation_source = "huggingface"
            config.preservation_path = self.preservation_hf_edit.text().strip()
            config.preservation_column = self.hf_preservation_col_edit.text().strip()

        # Settings
        config.n_samples = self.n_samples_spin.value()
        config.direction_multiplier = self.strength_spin.value()
        config.precision = self.precision_combo.currentText()
        config.output_path = self.output_path_edit.text().strip()
        
        # Advanced options
        config.norm_preservation = self.norm_preservation_check.isChecked()
        config.winsorization = self.winsorization_check.isChecked()
        config.null_space_constraints = self.null_space_check.isChecked()
        config.adaptive_layer_weighting = self.adaptive_layers_check.isChecked()
        
        return config
    
    def _validate_config(self, config: BehaviorMancerConfig) -> tuple[bool, str]:
        """Validate configuration."""
        if not config.model_path:
            return False, "Please specify a model path or HuggingFace model ID"

        if config.model_source == "local" and not os.path.isdir(config.model_path):
            return False, f"Model folder does not exist: {config.model_path}"

        if not config.dataset_path:
            return False, "Please specify a behavior dataset"

        if config.dataset_source == "local" and not os.path.isfile(config.dataset_path):
            return False, f"Dataset file does not exist: {config.dataset_path}"

        if not config.target_behavior_column or not config.baseline_behavior_column:
            return False, "Please specify both target and baseline column names"
        
        # Validate preservation dataset if null-space constraints enabled
        if config.null_space_constraints:
            if not config.preservation_path:
                return False, "Please specify a preservation dataset for null-space constraints"
            
            if config.preservation_source == "local" and not os.path.isfile(config.preservation_path):
                return False, f"Preservation file does not exist: {config.preservation_path}"

        if not config.output_path:
            return False, "Please specify an output folder"

        return True, ""
    
    def _start_removal(self):
        """Start the behavior removal process."""
        config = self._get_config()
        
        valid, error = self._validate_config(config)
        if not valid:
            QMessageBox.warning(self, "Configuration Error", error)
            return
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.load_model_button.setEnabled(False)
        self.test_button.setEnabled(False)
        
        # Switch to Run tab
        self.tabs.setCurrentIndex(3)
        
        self.log_view.clear()
        self._append_log("Starting behavior removal process...")
        
        self.timer.start()
        
        # Run in background thread
        self.worker_thread = threading.Thread(
            target=self._run_removal,
            args=(config,),
            daemon=True
        )
        self.worker_thread.start()
    
    def _run_removal(self, config: BehaviorMancerConfig):
        """Run behavior removal in background thread."""
        try:
            self.behavior_mancer = BehaviorMancer(
                config=config,
                log_callback=lambda msg: self.signals.log.emit(msg)
            )
            
            success = self.behavior_mancer.run_full_pipeline()
            
            if success:
                self.signals.finished.emit(True, "Behavior removal completed successfully!")
            else:
                self.signals.finished.emit(False, "Behavior removal failed or was stopped")
                
        except Exception as e:
            import traceback
            self.signals.log.emit(f"Error: {str(e)}")
            self.signals.log.emit(traceback.format_exc())
            self.signals.finished.emit(False, f"Error: {str(e)}")
    
    def _stop_removal(self):
        """Stop the behavior removal process."""
        if self.behavior_mancer:
            self.behavior_mancer.request_stop()
            self._append_log("Stop requested...")
    
    def _load_model_only(self):
        """Load model without running full pipeline."""
        config = self._get_config()
        
        if not config.model_path:
            QMessageBox.warning(self, "Error", "Please specify a model path")
            return
        
        self.start_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        
        # Switch to Run tab
        self.tabs.setCurrentIndex(3)
        
        self._append_log("Loading model...")
        
        self.timer.start()
        
        thread = threading.Thread(
            target=self._run_load_model,
            args=(config,),
            daemon=True
        )
        thread.start()
    
    def _run_load_model(self, config: BehaviorMancerConfig):
        """Load model in background thread."""
        try:
            self.behavior_mancer = BehaviorMancer(
                config=config,
                log_callback=lambda msg: self.signals.log.emit(msg)
            )
            
            if self.behavior_mancer.load_model():
                info = self.behavior_mancer.get_model_info()
                self.signals.model_info.emit(info)
                self.signals.finished.emit(True, "Model loaded successfully!")
            else:
                self.signals.finished.emit(False, "Failed to load model")
                
        except Exception as e:
            self.signals.log.emit(f"Error: {str(e)}")
            self.signals.finished.emit(False, f"Error: {str(e)}")
    
    def _test_prompt(self):
        """Test the loaded model with a prompt."""
        if not self.behavior_mancer or not self.behavior_mancer.model:
            QMessageBox.warning(self, "Error", "No model loaded")
            return
        
        prompt = self.test_prompt_edit.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Please enter a test prompt")
            return
        
        self.test_button.setEnabled(False)
        self.test_output.clear()
        self.test_output.setPlainText("Generating...")
        
        thread = threading.Thread(
            target=self._run_test,
            args=(prompt,),
            daemon=True
        )
        thread.start()
    
    def _run_test(self, prompt: str):
        """Run test in background thread."""
        try:
            response = self.behavior_mancer.test_model(prompt)
            self.signals.test_result.emit(response)
        except Exception as e:
            self.signals.test_result.emit(f"Error: {str(e)}")
    
    def _on_finished(self, success: bool, message: str):
        """Handle worker thread completion."""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_model_button.setEnabled(True)
        
        if self.behavior_mancer and self.behavior_mancer.model:
            self.test_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            if "stopped" not in message.lower():
                QMessageBox.warning(self, "Error", message)
        
        self._save_config()
    
    def _on_model_info(self, info: dict):
        """Handle model info update."""
        if "error" in info:
            self.model_info_label.setText(info["error"])
        else:
            info_text = f"Architecture: {info.get('architecture', 'Unknown')} | "
            info_text += f"Params: {info.get('parameters', '?')} | "
            info_text += f"Layers: {info.get('num_layers', '?')}"
            self.model_info_label.setText(info_text)
    
    def _on_test_result(self, result: str):
        """Handle test result."""
        self.test_button.setEnabled(True)
        self.test_output.setPlainText(result)
    
    def _check_queue(self):
        """Check message queue for updates."""
        pass
    
    def _append_log(self, text: str):
        """Append text to log view."""
        if not text:
            return
        
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_view.setTextCursor(cursor)
        
        # Color coding
        fmt = QTextCharFormat()
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['error', 'failed', 'fatal']):
            fmt.setForeground(QColor(255, 100, 100))
        elif 'warning' in text_lower:
            fmt.setForeground(QColor(255, 255, 0))
        elif any(word in text_lower for word in ['success', 'complete', 'saved']):
            fmt.setForeground(QColor(100, 255, 100))
        elif any(word in text_lower for word in ['step', 'loading', 'processing', 'extracting']):
            fmt.setForeground(QColor(100, 150, 255))
        else:
            fmt.setForeground(QColor(200, 200, 200))
        
        cursor.insertText(text + '\n', fmt)
        self.log_view.ensureCursorVisible()
    
    # =========================================================================
    # Config Save/Load
    # =========================================================================
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            config = {
                "model_source": "local" if self.model_local_radio.isChecked() else "huggingface",
                "model_local_path": self.model_path_edit.text(),
                "model_hf_id": self.model_hf_edit.text(),
                "hf_token": self.hf_token_edit.text(),
                "dataset_source": "local" if self.dataset_local_radio.isChecked() else "huggingface",
                "dataset_local_path": self.dataset_path_edit.text(),
                "dataset_hf_id": self.dataset_hf_edit.text(),
                "local_target_col": self.local_target_col_edit.text(),
                "local_baseline_col": self.local_baseline_col_edit.text(),
                "hf_target_col": self.hf_target_col_edit.text(),
                "hf_baseline_col": self.hf_baseline_col_edit.text(),
                "preservation_source": "local" if self.preservation_local_radio.isChecked() else "huggingface",
                "preservation_local_path": self.preservation_path_edit.text(),
                "preservation_hf_id": self.preservation_hf_edit.text(),
                "local_preservation_col": self.local_preservation_col_edit.text(),
                "hf_preservation_col": self.hf_preservation_col_edit.text(),
                "n_samples": self.n_samples_spin.value(),
                "strength": self.strength_spin.value(),
                "precision": self.precision_combo.currentText(),
                "output_path": self.output_path_edit.text(),
                "norm_preservation": self.norm_preservation_check.isChecked(),
                "winsorization": self.winsorization_check.isChecked(),
                "null_space": self.null_space_check.isChecked(),
                "adaptive_layers": self.adaptive_layers_check.isChecked(),
            }

            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(CONFIG_FILE):
            # Set default output path
            default_output = str(Path(__file__).parent.parent.parent / "outputs" / "behavior_manced_models")
            self.output_path_edit.setText(default_output)
            return
        
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Model source
            if config.get("model_source") == "huggingface":
                self.model_hf_radio.setChecked(True)
            else:
                self.model_local_radio.setChecked(True)
            
            self.model_path_edit.setText(config.get("model_local_path", ""))
            self.model_hf_edit.setText(config.get("model_hf_id", ""))
            self.hf_token_edit.setText(config.get("hf_token", ""))
            
            # Dataset source
            if config.get("dataset_source") == "huggingface":
                self.dataset_hf_radio.setChecked(True)
            else:
                self.dataset_local_radio.setChecked(True)
            
            self.dataset_path_edit.setText(config.get("dataset_local_path", ""))
            self.dataset_hf_edit.setText(config.get("dataset_hf_id", ""))
            self.local_target_col_edit.setText(config.get("local_target_col", "target"))
            self.local_baseline_col_edit.setText(config.get("local_baseline_col", "baseline"))
            self.hf_target_col_edit.setText(config.get("hf_target_col", "target"))
            self.hf_baseline_col_edit.setText(config.get("hf_baseline_col", "baseline"))
            
            # Preservation source
            if config.get("preservation_source") == "huggingface":
                self.preservation_hf_radio.setChecked(True)
            else:
                self.preservation_local_radio.setChecked(True)
            
            self.preservation_path_edit.setText(config.get("preservation_local_path", ""))
            self.preservation_hf_edit.setText(config.get("preservation_hf_id", ""))
            self.local_preservation_col_edit.setText(config.get("local_preservation_col", "prompt"))
            self.hf_preservation_col_edit.setText(config.get("hf_preservation_col", "prompt"))

            # Settings
            self.n_samples_spin.setValue(config.get("n_samples", 30))
            self.strength_spin.setValue(config.get("strength", 1.0))
            
            precision_idx = self.precision_combo.findText(config.get("precision", "float16"))
            if precision_idx >= 0:
                self.precision_combo.setCurrentIndex(precision_idx)
            
            self.output_path_edit.setText(config.get("output_path", ""))
            
            # Advanced options
            self.norm_preservation_check.setChecked(config.get("norm_preservation", True))
            self.winsorization_check.setChecked(config.get("winsorization", False))
            self.null_space_check.setChecked(config.get("null_space", False))
            self.adaptive_layers_check.setChecked(config.get("adaptive_layers", False))
            
            # Update UI state
            self._on_model_source_changed(True)
            self._on_dataset_source_changed(True)
            self._on_preservation_source_changed(True)

        except Exception as e:
            print(f"Error loading config: {e}")
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'galaxy_bg') and hasattr(self, '_central_widget'):
            self.galaxy_bg.resize(self._central_widget.size())
    
    def closeEvent(self, event):
        """Handle window close."""
        self._save_config()
        
        if self.behavior_mancer:
            self.behavior_mancer.cleanup()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = BehaviorMancerApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
