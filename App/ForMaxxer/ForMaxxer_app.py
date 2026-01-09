import os
import json
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QFileDialog, QLineEdit, QMessageBox, QApplication, QCheckBox, QGroupBox,
    QGridLayout, QFrame
)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt
from App.ForMaxxer.ForMaxxer import DatasetConverter, filter_dataset

# Logging will be configured only when needed, not at module import time


class DatasetConverterApp(QWidget):
    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self.theme = theme
        self.setWindowTitle("🗂️ ForMaxxer")
        self.resize(700, 550)
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

        filter_checkboxes = [
            self.blank_turns_cb,
            self.invalid_endings_cb,
            self.null_gpt_cb,
            self.duplicate_system_cb,
            self.allow_empty_system_cb,
            self.duplicate_turns_cb,
        ]

        for cb in filter_checkboxes:
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')}; font-size: 12px;")

        # Arrange in 2 columns
        for idx, cb in enumerate(filter_checkboxes):
            row, col = divmod(idx, 2)
            filter_layout.addWidget(cb, row, col)

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
                "JSON Files (*.json *.jsonl);;All Files (*)"
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
            self.convert_button.setEnabled(True)
        else:
            self.update_status("❌ No input files selected.")
            QMessageBox.critical(self, "Input Error", "Please select input files.")

    def convert_multiple_datasets(self, input_paths):
        try:
            # Configure logging only when needed (not at module import time)
            if not logging.getLogger().handlers:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            
            self.update_status("⚙️ Conversion in progress...")
            logging.info(f"Starting conversion for {', '.join(input_paths)}")

            # Output directory
            # Default to outputs folder in repo root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            output_dir = os.path.join(repo_root, "outputs", "formaxxer")
            os.makedirs(output_dir, exist_ok=True)

            # Process each file
            format_info = []
            converted_files = []
            for input_path in input_paths:
                filename = os.path.basename(input_path)
                self.update_status(f"Processing file: {filename}...")
                logging.info(f"Processing file: {input_path}")

                results = DatasetConverter.process_multiple_files([input_path], output_dir)
                result = results.get(filename, ([], "unknown"))
                
                if isinstance(result, tuple):
                    preview_entries, detected_format = result
                else:
                    # Backward compatibility
                    preview_entries = result if isinstance(result, list) else []
                    detected_format = "unknown"
                
                format_info.append(f"{filename}: {detected_format}")
                self.update_preview(preview_entries)
                
                # Track converted file path for optional filtering
                converted_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jsonl")
                if os.path.exists(converted_file):
                    converted_files.append(converted_file)

            format_summary = " | ".join(format_info)
            
            # Apply filtering if enabled
            if self.enable_filtering_cb.isChecked() and converted_files:
                self.update_status("🛠 Applying filtering to converted files...")
                filter_results = []
                
                for converted_file in converted_files:
                    filename = os.path.basename(converted_file)
                    self.update_status(f"Filtering: {filename}...")
                    
                    try:
                        summary, output_path = filter_dataset(
                            converted_file,
                            output_dir,
                            check_blank_turns=self.blank_turns_cb.isChecked(),
                            check_invalid_endings=self.invalid_endings_cb.isChecked(),
                            check_null_gpt=self.null_gpt_cb.isChecked(),
                            check_duplicate_system=self.duplicate_system_cb.isChecked(),
                            allow_empty_system_role=self.allow_empty_system_cb.isChecked(),
                            check_duplicate_turns=self.duplicate_turns_cb.isChecked(),
                        )
                        filter_results.append(f"{filename}: filtered")
                    except Exception as filter_error:
                        filter_results.append(f"{filename}: filter error - {str(filter_error)}")
                
                filter_summary = " | ".join(filter_results)
                self.update_status(f"✅ Conversion & filtering completed. Formats: {format_summary}\nFiltering: {filter_summary}")
            else:
                self.update_status(f"✅ Conversion completed. Detected formats: {format_summary}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

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
