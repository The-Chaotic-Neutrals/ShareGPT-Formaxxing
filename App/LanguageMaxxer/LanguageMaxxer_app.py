"""
LanguageMaxxer App - Combined English filtering and grammar correction UI.
"""

import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QCheckBox, QGroupBox, QGridLayout,
    QFrame, QMessageBox, QDoubleSpinBox
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread

from App.LanguageMaxxer.LanguageMaxxer import filter_english_jsonl, correct_grammar_jsonl, GrammarCorrector


class EnglishFilterWorker(QThread):
    """Worker thread for English filtering."""
    finished = pyqtSignal(dict, str)  # stats, output_path
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, input_path, threshold):
        super().__init__()
        self.input_path = input_path
        self.threshold = threshold

    def run(self):
        try:
            self.status_update.emit("Filtering for English conversations...")
            stats, output_path = filter_english_jsonl(
                input_path=self.input_path,
                output_path=None,
                rejected_path=None,
                threshold=self.threshold,
                batch_size=256,
                workers=None
            )
            self.finished.emit(stats, output_path)
        except Exception as e:
            self.error.emit(str(e))


class GrammarCorrectionWorker(QThread):
    """Worker thread for grammar correction."""
    finished = pyqtSignal(str)  # output_path
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    correction_update = pyqtSignal(str, str)  # original, corrected

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            self.status_update.emit("Correcting grammar...")
            output_path = correct_grammar_jsonl(
                self.input_path,
                output_path=None,
                update_callback=self.correction_update.emit
            )
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))


class LanguageMaxxerApp(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or {}
        self.setWindowTitle("🌐 LanguageMaxxer")
        self.resize(800, 650)
        self.apply_theme()
        self.setup_ui()
        self.set_icon()
        
        # Worker references
        self.english_worker = None
        self.grammar_worker = None
        self.current_output_path = None

    def apply_theme(self):
        """Apply the given theme to the whole window."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme.get('bg', '#000000')};
                color: {self.theme.get('text_fg', '#ffffff')};
            }}
            QLabel {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: {self.theme.get('entry_bg', '#000000')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                padding: 6px;
                font-size: 14px;
            }}
            QTextEdit {{
                background-color: {self.theme.get('text_bg', '#000000')};
                color: {self.theme.get('text_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('fg', '#1e90ff')};
                color: {self.theme.get('bg', '#000000')};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
            }}
            QGroupBox {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 14px;
                font-weight: bold;
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QCheckBox {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 13px;
                spacing: 8px;
            }}
            QDoubleSpinBox {{
                background-color: {self.theme.get('entry_bg', '#000000')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 4px;
                padding: 4px;
            }}
        """)

    def set_icon(self):
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("🌐 LanguageMaxxer")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # File selection row
        file_row = QHBoxLayout()
        file_label = QLabel("📂 Input File:")
        file_row.addWidget(file_label)

        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select a .jsonl file...")
        file_row.addWidget(self.input_file_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input_file)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Separator
        layout.addWidget(self._make_separator())

        # English Filtering Section
        self.english_group = QGroupBox("🇬🇧 English Filtering (FastText)")
        english_layout = QVBoxLayout(self.english_group)

        self.enable_english_cb = QCheckBox("Enable English filtering")
        self.enable_english_cb.setChecked(True)
        self.enable_english_cb.stateChanged.connect(self.toggle_english_options)
        english_layout.addWidget(self.enable_english_cb)

        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Confidence threshold:")
        threshold_row.addWidget(threshold_label)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.69)
        self.threshold_spin.setFixedWidth(80)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        english_layout.addLayout(threshold_row)

        layout.addWidget(self.english_group)

        # Grammar Correction Section
        self.grammar_group = QGroupBox("📝 Grammar Correction (LanguageTool)")
        grammar_layout = QVBoxLayout(self.grammar_group)

        self.enable_grammar_cb = QCheckBox("Enable grammar correction (applied to GPT responses)")
        self.enable_grammar_cb.setChecked(False)
        grammar_layout.addWidget(self.enable_grammar_cb)

        grammar_note = QLabel("Note: Grammar correction runs after English filtering if both are enabled.")
        grammar_note.setStyleSheet("color: #888888; font-size: 11px; font-style: italic;")
        grammar_layout.addWidget(grammar_note)

        layout.addWidget(self.grammar_group)

        # Separator
        layout.addWidget(self._make_separator())

        # Process button
        self.process_btn = QPushButton("🚀 Process")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Output/Log area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Processing output will appear here...")
        layout.addWidget(self.output_text, stretch=1)

        # Status bar
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("padding: 5px; font-size: 12px;")
        layout.addWidget(self.status_label)

    def _make_separator(self):
        """Returns a horizontal separator line"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"color: {self.theme.get('fg', '#1e90ff')};")
        return line

    def toggle_english_options(self, state):
        """Enable/disable threshold spinner based on checkbox state."""
        self.threshold_spin.setEnabled(state == Qt.Checked)

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSONL File", "", "JSONL files (*.jsonl)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)

    def start_processing(self):
        input_path = self.input_file_edit.text().strip()
        if not input_path:
            QMessageBox.critical(self, "Error", "Please select an input file.")
            return

        if not os.path.exists(input_path):
            QMessageBox.critical(self, "Error", "Input file does not exist.")
            return

        enable_english = self.enable_english_cb.isChecked()
        enable_grammar = self.enable_grammar_cb.isChecked()

        if not enable_english and not enable_grammar:
            QMessageBox.warning(self, "Warning", "Please enable at least one processing option.")
            return

        self.process_btn.setEnabled(False)
        self.output_text.clear()
        self.current_output_path = input_path

        if enable_english:
            self.run_english_filtering()
        elif enable_grammar:
            self.run_grammar_correction(input_path)

    def run_english_filtering(self):
        input_path = self.input_file_edit.text().strip()
        threshold = self.threshold_spin.value()

        self.english_worker = EnglishFilterWorker(input_path, threshold)
        self.english_worker.status_update.connect(self.update_status)
        self.english_worker.finished.connect(self.on_english_finished)
        self.english_worker.error.connect(self.on_error)
        self.english_worker.start()

    def on_english_finished(self, stats, output_path):
        self.current_output_path = output_path
        
        summary = (
            f"✅ English Filtering Complete!\n"
            f"   Total lines: {stats['total_lines']}\n"
            f"   English kept: {stats['english_total']}\n"
            f"   Non-English removed: {stats['non_english_total']}\n"
            f"   JSON errors: {stats['json_error_total']}\n"
            f"   Output: {output_path}\n\n"
        )
        self.output_text.append(summary)

        # If grammar correction is also enabled, run it on the filtered output
        if self.enable_grammar_cb.isChecked():
            self.run_grammar_correction(output_path)
        else:
            self.update_status("Done!")
            self.process_btn.setEnabled(True)

    def run_grammar_correction(self, input_path):
        self.grammar_worker = GrammarCorrectionWorker(input_path)
        self.grammar_worker.status_update.connect(self.update_status)
        self.grammar_worker.correction_update.connect(self.on_correction)
        self.grammar_worker.finished.connect(self.on_grammar_finished)
        self.grammar_worker.error.connect(self.on_error)
        self.grammar_worker.start()

    def on_correction(self, original, corrected):
        if original != corrected:
            # Only show if there was an actual change
            diff_preview = f"📝 Correction applied (showing first 100 chars):\n"
            diff_preview += f"   Before: {original[:100]}...\n" if len(original) > 100 else f"   Before: {original}\n"
            diff_preview += f"   After: {corrected[:100]}...\n\n" if len(corrected) > 100 else f"   After: {corrected}\n\n"
            self.output_text.append(diff_preview)
            # Auto-scroll
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def on_grammar_finished(self, output_path):
        summary = (
            f"✅ Grammar Correction Complete!\n"
            f"   Output: {output_path}\n\n"
        )
        self.output_text.append(summary)
        self.update_status("Done!")
        self.process_btn.setEnabled(True)

    def on_error(self, message):
        self.output_text.append(f"❌ Error: {message}\n")
        self.update_status(f"Error: {message}")
        self.process_btn.setEnabled(True)

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")


if __name__ == "__main__":
    from App.Other.Theme import Theme
    app = QApplication(sys.argv)
    window = LanguageMaxxerApp(Theme.DARK)
    window.show()
    sys.exit(app.exec_())
