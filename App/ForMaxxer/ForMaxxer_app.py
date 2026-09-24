import os
import json
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QLineEdit, QMessageBox, QApplication, QCheckBox, QGroupBox,
    QGridLayout, QFrame, QSpinBox
)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from App.ForMaxxer.ForMaxxer import DatasetConverter, filter_dataset, format_bytes

# Logging will be configured only when needed, not at module import time


class ForMaxxerWorker(QThread):
    status = pyqtSignal(str)
    preview = pyqtSignal(object)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_paths, filter_options, parent=None):
        super().__init__(parent)
        self.input_paths = input_paths
        self.filter_options = filter_options

    def run(self):
        try:
            # Configure logging only when needed (not at module import time)
            if not logging.getLogger().handlers:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

            self.status.emit("⚙️ Conversion in progress...")
            logging.info(f"Starting conversion for {', '.join(self.input_paths)}")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(script_dir))
            output_dir = os.path.join(repo_root, "Outputs")
            os.makedirs(output_dir, exist_ok=True)

            format_info = []
            converted_files = []
            for input_path in self.input_paths:
                filename = os.path.basename(input_path)
                self.status.emit(f"Processing file: {filename}...")
                logging.info(f"Processing file: {input_path}")

                results = DatasetConverter.process_multiple_files([input_path], output_dir)
                result = results.get(filename, ([], "unknown"))

                if isinstance(result, tuple):
                    preview_entries, detected_format = result
                else:
                    preview_entries = result if isinstance(result, list) else []
                    detected_format = "unknown"

                format_info.append(f"{filename}: {detected_format}")
                self.preview.emit(preview_entries)

                converted_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jsonl")
                if os.path.exists(converted_file):
                    converted_files.append(converted_file)
                    input_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
                    converted_size = os.path.getsize(converted_file)
                    format_info[-1] = (
                        f"{filename}: {detected_format} "
                        f"({format_bytes(input_size)} → {format_bytes(converted_size)} converted)"
                    )

            format_summary = " | ".join(format_info)

            if self.filter_options["enabled"] and converted_files:
                self.status.emit("🛠 Applying filtering to converted files...")
                filter_results = []

                for converted_file in converted_files:
                    filename = os.path.basename(converted_file)
                    self.status.emit(f"Filtering: {filename}...")

                    try:
                        summary, output_path = filter_dataset(
                            converted_file,
                            output_dir,
                            check_blank_turns=self.filter_options["check_blank_turns"],
                            check_invalid_endings=self.filter_options["check_invalid_endings"],
                            check_null_gpt=self.filter_options["check_null_gpt"],
                            check_duplicate_system=self.filter_options["check_duplicate_system"],
                            allow_empty_system_role=self.filter_options["allow_empty_system_role"],
                            check_duplicate_turns=self.filter_options["check_duplicate_turns"],
                            duplicate_similarity_threshold=self.filter_options["duplicate_similarity_threshold"],
                            strip_think_tags=self.filter_options["strip_think_tags"],
                            normalize_gemma_reasoning=self.filter_options["normalize_gemma_reasoning"],
                        )
                        filter_results.append(
                            f"{filename}: filtered → {format_bytes(os.path.getsize(output_path))}\n{summary}"
                        )
                    except Exception as filter_error:
                        filter_results.append(f"{filename}: filter error - {str(filter_error)}")

                filter_summary = " | ".join(filter_results)
                self.completed.emit(f"✅ Conversion & filtering completed. Formats: {format_summary}\nFiltering: {filter_summary}")
            else:
                self.completed.emit(f"✅ Conversion completed. Detected formats: {format_summary}")
        except Exception as e:
            self.error.emit(str(e))


class DatasetConverterApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("🗂️ ForMaxxer")
        self.resize(700, 550)
        self.worker = None
        self.apply_theme()
        self.setup_ui()
        self.set_icon()

    def apply_theme(self):
        """Apply the given theme to the whole window."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.theme.get('bg', '#000000')))
        palette.setColor(QPalette.WindowText, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Base, QColor(self.theme.get('text_bg', '#000000')))
        palette.setColor(QPalette.Text, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Button, QColor(self.theme.get('button_bg', '#1e90ff')))
        palette.setColor(QPalette.ButtonText, QColor(self.theme.get('button_fg', '#ffffff')))
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QLabel {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: {self.theme.get('entry_bg', '#000000')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                padding: 4px;
                font-size: 14px;
            }}
            QTextEdit {{
                background-color: {self.theme.get('text_bg', '#000000')};
                color: {self.theme.get('text_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('fg', '#1e90ff')};
                color: {self.theme.get('bg', '#000000')};
            }}
        """)

    def set_icon(self):
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        # Note: UI_Manager also sets the icon, so this is a fallback

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Input file selection row
        file_row = QHBoxLayout()
        label = QLabel("📂 Input Files:")
        file_row.addWidget(label)

        self.entry_input_file = QLineEdit(self)
        self.entry_input_file.setPlaceholderText("Select input files...")
        file_row.addWidget(self.entry_input_file)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_input_files)
        file_row.addWidget(browse_button)
        layout.addLayout(file_row)

        # Separator
        layout.addWidget(self._make_separator())

        # Enable filtering checkbox
        self.enable_filtering_cb = QCheckBox("🛠 Apply filtering after conversion (formerly DataMaxxer)")
        self.enable_filtering_cb.setChecked(False)
        self.enable_filtering_cb.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')}; font-size: 14px; font-weight: bold;")
        self.enable_filtering_cb.stateChanged.connect(self.toggle_filtering_options)
        layout.addWidget(self.enable_filtering_cb)

        # Filtering options group (collapsible)
        self.filtering_group = QGroupBox("Filtering Options")
        self.filtering_group.setStyleSheet(f"""
            QGroupBox {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 13px;
                font-weight: bold;
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        filter_layout = QGridLayout(self.filtering_group)
        filter_layout.setSpacing(8)

        # Create filtering checkboxes
        self.blank_turns_cb = QCheckBox("Check Blank Turns")
        self.invalid_endings_cb = QCheckBox("Check Invalid Endings")
        self.null_gpt_cb = QCheckBox("Check Null GPT")
        self.duplicate_system_cb = QCheckBox("Check Duplicate System")
        self.allow_empty_system_cb = QCheckBox("Allow Empty System Role")
        self.duplicate_turns_cb = QCheckBox("Check Duplicate Human → GPT Turns")
        self.strip_think_tags_cb = QCheckBox("Collapse GPT reasoning blocks (single block per response)")
        self.normalize_gemma_reasoning_cb = QCheckBox("Normalize Gemma 4 reasoning tags")
        self.normalize_gemma_reasoning_cb.setToolTip(
            "Preserves valid <|channel>thought...<channel|> blocks, repairs common spacing variants, "
            "and prepends an empty thought block to untagged GPT responses."
        )
        self.strip_think_tags_cb.setToolTip(
            "Removes closed <think>...</think> and Gemma 4 <|channel>thought...<channel|> blocks from "
            "all but the last GPT response; the final GPT response keeps only its first reasoning block "
            "plus the remaining content. Drops conversations with an unclosed reasoning block."
        )

        filter_checkboxes = [
            self.blank_turns_cb,
            self.invalid_endings_cb,
            self.null_gpt_cb,
            self.duplicate_system_cb,
            self.allow_empty_system_cb,
            self.duplicate_turns_cb,
            self.strip_think_tags_cb,
            self.normalize_gemma_reasoning_cb,
        ]

        for cb in filter_checkboxes:
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')}; font-size: 12px;")

        # Arrange in 2 columns
        for idx, cb in enumerate(filter_checkboxes):
            row, col = divmod(idx, 2)
            filter_layout.addWidget(cb, row, col)

        threshold_row = (len(filter_checkboxes) + 1) // 2
        threshold_label = QLabel("Duplicate Similarity Threshold")
        threshold_label.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')}; font-size: 12px;")
        self.duplicate_similarity_spin = QSpinBox(self)
        self.duplicate_similarity_spin.setRange(0, 100)
        self.duplicate_similarity_spin.setValue(92)
        self.duplicate_similarity_spin.setSuffix("%")
        self.duplicate_similarity_spin.setToolTip("Higher = stricter duplicate filtering")
        self.duplicate_similarity_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {self.theme.get('entry_bg', '#000000')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 4px;
                padding: 2px 4px;
                selection-background-color: {self.theme.get('fg', '#1e90ff')};
                selection-color: {self.theme.get('bg', '#000000')};
            }}
        """)
        filter_layout.addWidget(threshold_label, threshold_row, 0)
        filter_layout.addWidget(self.duplicate_similarity_spin, threshold_row, 1)

        layout.addWidget(self.filtering_group)
        self.filtering_group.setVisible(False)  # Hidden by default

        # Separator
        layout.addWidget(self._make_separator())

        # Convert button
        self.convert_button = QPushButton("🚀 Convert")
        self.convert_button.clicked.connect(self.on_convert_button_click)
        layout.addWidget(self.convert_button)

        # Preview text box
        self.preview_text = QTextEdit(self)
        self.preview_text.setReadOnly(True)
        layout.addWidget(self.preview_text, stretch=1)

        # Status bar (as a read-only QTextEdit)
        self.status_bar = QTextEdit(self)
        self.status_bar.setReadOnly(True)
        self.status_bar.setFixedHeight(40)
        self.status_bar.setText("Status: Ready")
        layout.addWidget(self.status_bar)

    def _make_separator(self):
        """Returns a horizontal separator line"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"color: {self.theme.get('fg', '#1e90ff')};")
        return line

    def toggle_filtering_options(self, state):
        """Show/hide filtering options based on checkbox state."""
        self.filtering_group.setVisible(state == Qt.Checked)

    def select_input_files(self):
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Input Files",
                "",
                "Dataset Files (*.json *.jsonl *.parquet);;JSON Files (*.json *.jsonl);;Parquet Files (*.parquet);;All Files (*)"
            )
            if file_paths:
                self.entry_input_file.setText("; ".join(file_paths))
                self.update_status(f"Selected files: {', '.join(file_paths)}")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            QMessageBox.critical(self, "File Selection Error", f"An error occurred: {str(e)}")

    def on_convert_button_click(self):
        input_paths = self.entry_input_file.text().split("; ")
        if input_paths and input_paths[0].strip():
            self.convert_button.setEnabled(False)
            self.convert_multiple_datasets(input_paths)
        else:
            self.update_status("❌ No input files selected.")
            QMessageBox.critical(self, "Input Error", "Please select input files.")

    def convert_multiple_datasets(self, input_paths):
        filter_options = {
            "enabled": self.enable_filtering_cb.isChecked(),
            "check_blank_turns": self.blank_turns_cb.isChecked(),
            "check_invalid_endings": self.invalid_endings_cb.isChecked(),
            "check_null_gpt": self.null_gpt_cb.isChecked(),
            "check_duplicate_system": self.duplicate_system_cb.isChecked(),
            "allow_empty_system_role": self.allow_empty_system_cb.isChecked(),
            "check_duplicate_turns": self.duplicate_turns_cb.isChecked(),
            "duplicate_similarity_threshold": self.duplicate_similarity_spin.value(),
            "strip_think_tags": self.strip_think_tags_cb.isChecked(),
            "normalize_gemma_reasoning": self.normalize_gemma_reasoning_cb.isChecked(),
        }
        self.worker = ForMaxxerWorker(input_paths, filter_options, self)
        self.worker.status.connect(self.update_status)
        self.worker.preview.connect(self.update_preview)
        self.worker.completed.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_finished(self, message):
        self.update_status(message)
        self.convert_button.setEnabled(True)
        self.worker = None

    def on_worker_error(self, message):
        self.update_status(f"Error: {message}")
        self.convert_button.setEnabled(True)
        self.worker = None
        QMessageBox.critical(self, "Error", f"An error occurred: {message}")

    def update_preview(self, preview_entries):
        if not preview_entries:
            self.preview_text.setPlainText("No preview available.")
            return
        preview_data = preview_entries[:10] if isinstance(preview_entries, list) else [preview_entries]
        self.preview_text.setPlainText(json.dumps(preview_data, ensure_ascii=False, indent=2))

    def update_status(self, message):
        self.status_bar.setPlainText(message)


if __name__ == "__main__":
    import sys
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    theme = Theme.DARK
    window = DatasetConverterApp(theme)
    window.show()
    sys.exit(app.exec_())
