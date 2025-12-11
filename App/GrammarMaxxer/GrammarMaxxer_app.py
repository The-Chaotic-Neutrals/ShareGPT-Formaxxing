import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QCheckBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread

from App.GrammarMaxxer.GrammarMaxxer import GrammarMaxxer


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    correction_update = pyqtSignal(str, str)

    def __init__(self, input_file, toggles):
        super().__init__()
        self.input_file = input_file
        self.toggles = toggles

    def run(self):
        grammar_maxxer = GrammarMaxxer(self.input_file, self.toggles)
        if not grammar_maxxer.validate_file():
            self.error.emit(f"Invalid input file: {self.input_file}")
            self.finished.emit()
            return

        output_file = grammar_maxxer.prepare_output_file()
        self.status_update.emit("Text correction started...")
        try:
            grammar_maxxer.process_file(output_file, self.correction_update.emit)
            self.status_update.emit(f"Text correction complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
            self.error.emit(f"Error processing file: {e}")
        self.finished.emit()


class GrammarMaxxerApp(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or {}
        self.toggles = {'grammar': True}
        self.init_ui()
        self.configure_logging()

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.StreamHandler()
                            ])

    def init_ui(self):
        self.setWindowTitle("📝 GrammarMaxxer")
        self.resize(750, 600)
        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title_label = QLabel("📝 GrammarMaxxer")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')};")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # File selection row
        file_layout = QHBoxLayout()
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Choose a .jsonl file to correct...")
        self.input_file_edit.setFont(QFont("Segoe UI", 12))
        self.input_file_edit.setStyleSheet(
            f"background-color: {self.theme.get('entry_bg', '#000000')}; "
            f"color: {self.theme.get('entry_fg', '#ffffff')}; padding: 6px; border-radius: 6px;"
        )

        self.browse_button = QPushButton("📂 Browse")
        self.browse_button.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.browse_button.clicked.connect(self.browse_input_file)
        self.browse_button.setStyleSheet(
            f"background-color: {self.theme.get('button_bg', '#1e90ff')}; "
            f"color: {self.theme.get('button_fg', '#ffffff')}; padding: 8px 16px; border-radius: 8px;"
        )

        file_layout.addWidget(self.input_file_edit)
        file_layout.addWidget(self.browse_button)
        main_layout.addLayout(file_layout)

        # Toggles row
        toggles_layout = QHBoxLayout()
        self.grammar_toggle = QCheckBox("Enable Grammar Correction ✨")
        self.grammar_toggle.setChecked(True)
        self.grammar_toggle.setFont(QFont("Segoe UI", 12))
        self.grammar_toggle.setStyleSheet(f"color: {self.theme.get('text_fg', '#ffffff')};")
        self.grammar_toggle.stateChanged.connect(self.toggle_grammar_correction)
        toggles_layout.addWidget(self.grammar_toggle)
        toggles_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(toggles_layout)

        # Correct button
        self.correct_button = QPushButton("🚀 Correct Text")
        self.correct_button.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.correct_button.clicked.connect(self.start_text_correction)
        self.correct_button.setStyleSheet(
            f"background-color: {self.theme.get('button_bg', '#1e90ff')}; "
            f"color: {self.theme.get('button_fg', '#ffffff')}; padding: 10px 20px; border-radius: 10px;"
        )
        main_layout.addWidget(self.correct_button)

        # Corrections text area
        self.corrections_text = QTextEdit()
        self.corrections_text.setReadOnly(True)
        self.corrections_text.setFont(QFont("Consolas", 11))
        self.corrections_text.setStyleSheet(
            f"background-color: {self.theme.get('text_bg', '#000000')}; "
            f"color: {self.theme.get('text_fg', '#ffffff')}; padding: 8px; border-radius: 6px;"
        )
        main_layout.addWidget(self.corrections_text, stretch=1)

        # Status bar
        self.status_bar = QLabel("Status: Ready")
        self.status_bar.setFont(QFont("Segoe UI", 11))
        self.status_bar.setStyleSheet(
            f"background-color: {self.theme.get('bg', '#000000')}; "
            f"color: {self.theme.get('text_fg', '#ffffff')}; padding: 4px 8px; border-radius: 4px;"
        )
        self.status_bar.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.status_bar)

        # Set main layout and background
        self.setLayout(main_layout)
        self.setStyleSheet(f"background-color: {self.theme.get('bg', '#000000')};")

    def toggle_grammar_correction(self, state):
        self.toggles['grammar'] = state == Qt.Checked

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSONL File", "", "JSONL files (*.jsonl)")
        if file_path:
            self.input_file_edit.setText(file_path)

    def start_text_correction(self):
        input_file = self.input_file_edit.text()
        if not input_file:
            self.update_status("Please select an input file.")
            return

        self.correct_button.setEnabled(False)
        self.corrections_text.clear()

        self.worker = Worker(input_file, self.toggles)
        self.qthread = QThread()
        self.worker.moveToThread(self.qthread)

        self.qthread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.correction_update.connect(self.update_corrections)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.qthread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.qthread.finished.connect(self.qthread.deleteLater)

        self.qthread.start()

    def update_status(self, message):
        self.status_bar.setText(f"Status: {message}")

    def update_corrections(self, original_text, corrected_text):
        correction_entry = f"Original: {original_text}\nCorrected: {corrected_text}\n\n"
        self.corrections_text.append(correction_entry)
        self.corrections_text.verticalScrollBar().setValue(self.corrections_text.verticalScrollBar().maximum())

    def handle_error(self, message):
        self.update_status(message)
        self.correct_button.setEnabled(True)

    def on_finished(self):
        self.correct_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    from App.Other.Theme import Theme
    window = GrammarMaxxerApp(Theme.DARK)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
