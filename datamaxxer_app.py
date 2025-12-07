from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
import os
import json
from pathlib import Path
from datamaxxer import filter_dataset


class DataMaxxerApp(QtWidgets.QMainWindow):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("DataMaxxer")

        self.set_icon()

        # Main widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setSpacing(25)
        layout.setContentsMargins(30, 30, 30, 30)
        central_widget.setStyleSheet(f"background-color: {self.theme['bg']}; color: {self.theme['fg']};")

        font_header = "font-weight: bold; font-size: 16px;"
        font_checkbox = "font-size: 14px;"

        # --- Dataset selection section ---
        ds_label = QtWidgets.QLabel("📂 Dataset Selection")
        ds_label.setStyleSheet(font_header)
        layout.addWidget(ds_label)

        file_layout = QtWidgets.QHBoxLayout()
        self.dataset_entry = QtWidgets.QLineEdit()
        self.dataset_entry.setPlaceholderText("Choose your .jsonl dataset file...")
        self.dataset_entry.setStyleSheet(
            f"background-color: {self.theme['entry_bg']}; color: {self.theme['entry_fg']}; "
            f"padding: 8px; border-radius: 6px; font-size: 14px;"
        )
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.setMinimumWidth(100)
        browse_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['button_bg']};
                color: {self.theme['button_fg']};
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #7a7a7a;
            }}
            """
        )
        browse_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.dataset_entry)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)

        layout.addWidget(self._make_separator())

        # --- Filtering options section ---
        filter_label = QtWidgets.QLabel("🛠 Filtering Options")
        filter_label.setStyleSheet(font_header)
        layout.addWidget(filter_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)

        self.blank_turns_cb = QtWidgets.QCheckBox("Check Blank Turns")
        self.invalid_endings_cb = QtWidgets.QCheckBox("Check Invalid Endings")
        self.null_gpt_cb = QtWidgets.QCheckBox("Check Null GPT")
        self.duplicate_system_cb = QtWidgets.QCheckBox("Check Duplicate System")
        self.allow_empty_system_cb = QtWidgets.QCheckBox("Allow Empty System Role")
        # NEW: duplicate human→GPT turns
        self.duplicate_turns_cb = QtWidgets.QCheckBox("Check Duplicate Human → GPT Turns")

        checkboxes = [
            self.blank_turns_cb,
            self.invalid_endings_cb,
            self.null_gpt_cb,
            self.duplicate_system_cb,
            self.allow_empty_system_cb,
            self.duplicate_turns_cb,   # include new checkbox
        ]

        for cb in checkboxes:
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {self.theme['fg']}; spacing: 8px; {font_checkbox}")

        # arrange in 2 columns
        for idx, cb in enumerate(checkboxes):
            row, col = divmod(idx, 2)
            grid.addWidget(cb, row, col)

        layout.addLayout(grid)

        layout.addWidget(self._make_separator())

        # --- Process button ---
        process_button = QtWidgets.QPushButton("🚀 Process Dataset")
        process_button.setMinimumHeight(45)
        process_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.theme['button_bg']};
                color: {self.theme['button_fg']};
                padding: 10px 20px;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #888888;
            }}
            """
        )
        process_button.clicked.connect(self.process_dataset)
        layout.addWidget(process_button, alignment=Qt.AlignHCenter)  # type: ignore

        # Result label
        self.result_label = QtWidgets.QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.result_label.setStyleSheet("font-size: 13px; margin-top: 10px;")
        layout.addWidget(self.result_label)

    def _make_separator(self):
        """Returns a horizontal separator line"""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet(f"color: {self.theme['fg']};")
        return line

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

    def select_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Dataset", "", "JSON Lines files (*.jsonl)"
        )
        if file_path:
            self.dataset_entry.setText(file_path)

    def process_dataset(self):
        file_path = self.dataset_entry.text().strip()
        if not file_path:
            QtWidgets.QMessageBox.critical(self, "Input Error", "Please select a dataset file.")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                [json.loads(line) for line in f][:5]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "File Error", f"Could not read file: {str(e)}")
            return

        try:
            output_message = filter_dataset(
                file_path,
                Path(__file__).parent.absolute(),
                check_blank_turns=self.blank_turns_cb.isChecked(),
                check_invalid_endings=self.invalid_endings_cb.isChecked(),
                check_null_gpt=self.null_gpt_cb.isChecked(),
                check_duplicate_system=self.duplicate_system_cb.isChecked(),
                allow_empty_system_role=self.allow_empty_system_cb.isChecked(),
                check_duplicate_turns=self.duplicate_turns_cb.isChecked(),  # NEW
            )
            self.result_label.setText(output_message)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Processing Error", str(e))
