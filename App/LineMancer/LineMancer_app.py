from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QSpinBox, QRadioButton, QButtonGroup, QGroupBox, QFrame, QFileDialog,
    QMessageBox, QStackedWidget, QFormLayout
)
from App.Other.Theme import Theme
from App.Other.BG import GalaxyBackgroundWidget
from App.LineMancer.LineMancer import LineMancerCore
import os


class LineMancerFrame(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core = LineMancerCore()
        self.theme = Theme.DARK

        self.mode = "split"
        self.input_paths = []
        self.input_path = ""
        self.lines_per_file = 1000

        # Background
        self.background = GalaxyBackgroundWidget(self)
        self.background.lower()

        self._setup_style()
        self._build_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'background'):
            self.background.resize(self.size())

    def _setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #e6e6fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            QLabel {
                background-color: transparent;
                color: #e6e6fa;
            }
            QLabel#titleLabel {
                font-family: 'Georgia', 'Times New Roman', serif;
                font-size: 26px;
                font-weight: bold;
                color: #e6e6fa;
                padding: 0px;
                margin: 0px;
            }
            QLabel#subtitleLabel {
                font-size: 12px;
                color: #a0a0c0;
                padding: 0px;
                margin: 0px;
            }
            QGroupBox {
                background-color: rgba(10, 10, 30, 0.7);
                border: 1px solid rgba(100, 100, 180, 0.3);
                border-radius: 12px;
                margin-top: 16px;
                padding: 16px;
                padding-top: 28px;
                font-weight: bold;
                font-size: 12pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 12px;
                background-color: rgba(30, 60, 120, 0.6);
                border-radius: 8px;
                color: #c0c0ff;
                left: 12px;
            }
            QLineEdit {
                background-color: rgba(5, 5, 20, 0.8);
                color: #e6e6fa;
                border: 1px solid rgba(100, 100, 180, 0.4);
                border-radius: 6px;
                padding: 10px 12px;
                font-size: 10pt;
                selection-background-color: rgba(70, 100, 180, 0.6);
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 150, 255, 0.6);
            }
            QSpinBox {
                background-color: rgba(5, 5, 20, 0.8);
                color: #e6e6fa;
                border: 1px solid rgba(100, 100, 180, 0.4);
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 10pt;
            }
            QSpinBox:focus {
                border: 1px solid rgba(100, 150, 255, 0.6);
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: rgba(30, 80, 160, 0.6);
                border: none;
                width: 20px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: rgba(50, 100, 200, 0.8);
            }
            QPushButton {
                background-color: rgba(30, 80, 160, 0.8);
                color: #ffffff;
                border: 1px solid rgba(100, 150, 255, 0.4);
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: rgba(50, 100, 200, 0.9);
                border: 1px solid rgba(150, 180, 255, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(20, 60, 140, 0.9);
            }
            QRadioButton {
                background-color: transparent;
                color: #e6e6fa;
                spacing: 10px;
                font-size: 11pt;
                padding: 6px 12px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid rgba(100, 150, 255, 0.5);
                background-color: rgba(10, 10, 30, 0.6);
            }
            QRadioButton::indicator:checked {
                background-color: rgba(70, 130, 220, 0.9);
                border: 2px solid rgba(150, 180, 255, 0.8);
            }
            QRadioButton::indicator:hover {
                border: 2px solid rgba(150, 180, 255, 0.7);
            }
            QStackedWidget {
                background-color: transparent;
            }
        """)

    def _action_button_style(self):
        return """
            QPushButton {
                background-color: rgba(30, 140, 80, 0.8);
                color: #ffffff;
                border: 1px solid rgba(80, 200, 120, 0.4);
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 11pt;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: rgba(40, 160, 100, 0.9);
                border: 1px solid rgba(100, 220, 140, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(20, 120, 60, 0.9);
            }
        """

    def _browse_button_style(self):
        return """
            QPushButton {
                background-color: rgba(80, 60, 140, 0.8);
                color: #ffffff;
                border: 1px solid rgba(140, 100, 200, 0.4);
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 10pt;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: rgba(100, 80, 160, 0.9);
                border: 1px solid rgba(160, 120, 220, 0.6);
            }
            QPushButton:pressed {
                background-color: rgba(60, 40, 120, 0.9);
            }
        """

    def _build_ui(self):
        self.setWindowTitle("LineMancer")
        self.setMinimumSize(700, 500)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        header_layout.setContentsMargins(0, 0, 0, 8)

        title_label = QLabel("✨ LineMancer")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("Split, merge, and shuffle JSONL files with ease")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: rgba(100, 100, 180, 0.3); max-height: 1px;")
        main_layout.addWidget(separator)

        # Mode Selection Group
        mode_group = QGroupBox("🎯 Operation Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setSpacing(20)
        mode_layout.setContentsMargins(16, 16, 16, 16)

        self.mode_group = QButtonGroup(self)
        modes = [("split", "🪓 Split"), ("merge", "🧵 Merge"), ("shuffle", "🎲 Shuffle")]

        mode_layout.addStretch()
        for mode_val, mode_text in modes:
            btn = QRadioButton(mode_text)
            btn.mode_val = mode_val
            btn.setCursor(Qt.PointingHandCursor)
            self.mode_group.addButton(btn)
            mode_layout.addWidget(btn)
            if mode_val == self.mode:
                btn.setChecked(True)
        mode_layout.addStretch()

        self.mode_group.buttonClicked.connect(self.on_mode_change)
        main_layout.addWidget(mode_group)

        # Stacked Widget for different modes
        self.stack = QStackedWidget()
        self._build_split_ui()
        self._build_merge_ui()
        self._build_shuffle_ui()
        main_layout.addWidget(self.stack, stretch=1)

        self.render_mode()

    def _build_split_ui(self):
        split_widget = QWidget()
        split_widget.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(split_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Settings group
        settings_group = QGroupBox("🪓 Split Settings")
        form_layout = QFormLayout(settings_group)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(16, 20, 16, 16)

        # Input file row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select a JSONL file to split...")
        browse_btn = QPushButton("📂 Browse")
        browse_btn.setStyleSheet(self._browse_button_style())
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_path_edit, stretch=1)
        input_layout.addWidget(browse_btn)

        input_label = QLabel("Input File:")
        input_label.setStyleSheet("font-weight: bold; color: #c0c0ff;")
        form_layout.addRow(input_label, input_layout)

        # Lines per file
        self.lines_per_file_spin = QSpinBox()
        self.lines_per_file_spin.setRange(1, 1000000)
        self.lines_per_file_spin.setValue(self.lines_per_file)
        self.lines_per_file_spin.setMinimumWidth(150)

        lines_label = QLabel("Lines per File:")
        lines_label.setStyleSheet("font-weight: bold; color: #c0c0ff;")
        form_layout.addRow(lines_label, self.lines_per_file_spin)

        layout.addWidget(settings_group)

        # Action button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        split_btn = QPushButton("🪓 Split JSONL")
        split_btn.setStyleSheet(self._action_button_style())
        split_btn.setCursor(Qt.PointingHandCursor)
        split_btn.clicked.connect(self.split_jsonl)
        btn_layout.addWidget(split_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.stack.addWidget(split_widget)

    def _build_merge_ui(self):
        merge_widget = QWidget()
        merge_widget.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(merge_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Settings group
        settings_group = QGroupBox("🧵 Merge Settings")
        form_layout = QFormLayout(settings_group)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(16, 20, 16, 16)

        # Input files row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        self.merge_path_edit = QLineEdit()
        self.merge_path_edit.setPlaceholderText("Select multiple JSONL files to merge...")
        browse_btn = QPushButton("📂 Browse")
        browse_btn.setStyleSheet(self._browse_button_style())
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self.browse_inputs)
        input_layout.addWidget(self.merge_path_edit, stretch=1)
        input_layout.addWidget(browse_btn)

        input_label = QLabel("Input Files:")
        input_label.setStyleSheet("font-weight: bold; color: #c0c0ff;")
        form_layout.addRow(input_label, input_layout)

        layout.addWidget(settings_group)

        # Action button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        merge_btn = QPushButton("🧵 Merge JSONL Files")
        merge_btn.setStyleSheet(self._action_button_style())
        merge_btn.setCursor(Qt.PointingHandCursor)
        merge_btn.clicked.connect(self.merge_jsonl)
        btn_layout.addWidget(merge_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.stack.addWidget(merge_widget)

    def _build_shuffle_ui(self):
        shuffle_widget = QWidget()
        shuffle_widget.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(shuffle_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Settings group
        settings_group = QGroupBox("🎲 Shuffle Settings")
        form_layout = QFormLayout(settings_group)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(16, 20, 16, 16)

        # Input file row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        self.input_path_edit_shuffle = QLineEdit()
        self.input_path_edit_shuffle.setPlaceholderText("Select a JSONL file to shuffle...")
        browse_btn = QPushButton("📂 Browse")
        browse_btn.setStyleSheet(self._browse_button_style())
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.clicked.connect(self.browse_input_shuffle)
        input_layout.addWidget(self.input_path_edit_shuffle, stretch=1)
        input_layout.addWidget(browse_btn)

        input_label = QLabel("Input File:")
        input_label.setStyleSheet("font-weight: bold; color: #c0c0ff;")
        form_layout.addRow(input_label, input_layout)

        layout.addWidget(settings_group)

        # Action button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        shuffle_btn = QPushButton("🎲 Shuffle JSONL Lines")
        shuffle_btn.setStyleSheet(self._action_button_style())
        shuffle_btn.setCursor(Qt.PointingHandCursor)
        shuffle_btn.clicked.connect(self.shuffle_jsonl)
        btn_layout.addWidget(shuffle_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.stack.addWidget(shuffle_widget)

    def on_mode_change(self, button):
        self.mode = button.mode_val
        self.render_mode()

    def render_mode(self):
        modes = {"split": 0, "merge": 1, "shuffle": 2}
        self.stack.setCurrentIndex(modes[self.mode])

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input JSONL File", "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if path:
            self.input_path_edit.setText(path)
            self.input_path = path

    def browse_inputs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input JSONL Files", "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if paths:
            self.input_paths = paths
            self.merge_path_edit.setText(", ".join(paths))

    def browse_input_shuffle(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input JSONL File", "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if path:
            self.input_path_edit_shuffle.setText(path)

    def split_jsonl(self):
        try:
            count = self.core.split_jsonl(
                self.input_path_edit.text(),
                lines_per_file=self.lines_per_file_spin.value()
            )
            QMessageBox.information(self, "Success", f"✅ Split complete!\n\n{count} parts saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ {str(e)}")

    def merge_jsonl(self):
        try:
            paths = self.merge_path_edit.text().split(', ')
            output_path, total = self.core.merge_jsonl(paths)
            QMessageBox.information(self, "Success", f"✅ Merge complete!\n\nMerged {total} lines into:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ {str(e)}")

    def shuffle_jsonl(self):
        try:
            output_path = self.core.shuffle_jsonl(self.input_path_edit_shuffle.text())
            QMessageBox.information(self, "Success", f"✅ Shuffle complete!\n\nShuffled lines saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"❌ {str(e)}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    app = QtWidgets.QApplication(sys.argv)
    window = LineMancerFrame()
    icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
    if icon_path.exists():
        window.setWindowIcon(QtGui.QIcon(str(icon_path)))
    window.show()
    sys.exit(app.exec_())
