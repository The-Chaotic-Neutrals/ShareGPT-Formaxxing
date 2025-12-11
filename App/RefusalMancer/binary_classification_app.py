from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QRadioButton, QButtonGroup, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from App.RefusalMancer.binary_classification import (
    initialize_models,
    filter_conversations,
    set_filter_mode,
    FILTER_MODE_RP,
    FILTER_MODE_NORMAL
)


class FilterThread(QThread):
    status_update = pyqtSignal(str)
    counts_update = pyqtSignal(int, int)
    progress_update = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, input_files, threshold, batch_size):
        super().__init__()
        self.input_files = input_files
        self.threshold = threshold
        self.batch_size = batch_size

    def run(self):
        from App.RefusalMancer.binary_classification import filter_conversations as fc

        def status_callback(msg):
            # Extract numeric progress percentage if present
            if "%" in msg:
                try:
                    percent = int(float(msg.split("(")[-1].split("%")[0]))
                    self.progress_update.emit(percent)
                except ValueError:
                    pass
            self.status_update.emit(msg)

        def counts_callback(pos, neg):
            self.counts_update.emit(pos, neg)

        total_files = len(self.input_files)
        for idx, input_file in enumerate(self.input_files):
            class DummyEntry:
                def __init__(self, val):
                    self._val = val
                def get(self):
                    return self._val

            fc(
                input_file_entry=DummyEntry(input_file),
                threshold_entry=DummyEntry(str(self.threshold)),
                batch_size_entry=DummyEntry(str(self.batch_size)),
                status_update_callback=status_callback,
                counts_update_callback=counts_callback,
            )

        self.finished_signal.emit()


class BinaryClassificationApp(QWidget):
    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.input_files = []

        initialize_models()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("RefusalMancer")

        self.setStyleSheet(
            f"""
            background-color: {self.theme['bg']};
            color: {self.theme['fg']};
            font-family: Arial, sans-serif;
            font-size: 16px;
            """
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        self.setLayout(main_layout)

        # Input File Selection
        file_layout = QHBoxLayout()
        file_layout.setSpacing(10)
        main_layout.addLayout(file_layout)

        file_label = QLabel("📂 Select Input Files:")
        file_label.setStyleSheet(f"color: {self.theme['fg']}; font-weight: bold; font-size: 18px;")
        file_layout.addWidget(file_label)

        self.input_file_entry = QLineEdit()
        self.input_file_entry.setStyleSheet(
            f"""
            color: {self.theme['fg']};
            background-color: {self.theme['bg']};
            border: 1px solid {self.theme['button_bg']};
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 16px;
            """
        )
        file_layout.addWidget(self.input_file_entry)

        browse_btn = QPushButton("📁 Browse")
        browse_btn.setStyleSheet(
            f"""
            background-color: {self.theme['button_bg']};
            color: {self.theme['button_fg']};
            border: none;
            padding: 10px 18px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 12px;
            """
        )
        browse_btn.clicked.connect(self.browse_input_file)
        file_layout.addWidget(browse_btn)

        # Filter Button
        self.filter_button = QPushButton("⚡ Filter Conversations")
        self.filter_button.setStyleSheet(
            f"""
            background-color: {self.theme['button_bg']};
            color: {self.theme['button_fg']};
            border: none;
            padding: 12px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            """
        )
        self.filter_button.clicked.connect(self.start_filtering)
        main_layout.addWidget(self.filter_button)

        # Threshold Input
        threshold_label = QLabel("🎯 Threshold (0.0 - 1.0):")
        threshold_label.setStyleSheet(f"color: {self.theme['fg']}; font-size: 16px; font-weight: 600;")
        main_layout.addWidget(threshold_label)

        self.threshold_entry = QLineEdit("0.75")
        self.threshold_entry.setStyleSheet(
            f"""
            color: {self.theme['fg']};
            background-color: {self.theme['bg']};
            border: 1px solid {self.theme['button_bg']};
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 16px;
            """
        )
        main_layout.addWidget(self.threshold_entry)

        # Batch Size Input
        batch_size_label = QLabel("📦 Batch Size:")
        batch_size_label.setStyleSheet(f"color: {self.theme['fg']}; font-size: 16px; font-weight: 600;")
        main_layout.addWidget(batch_size_label)

        self.batch_size_entry = QLineEdit("64")
        self.batch_size_entry.setStyleSheet(
            f"""
            color: {self.theme['fg']};
            background-color: {self.theme['bg']};
            border: 1px solid {self.theme['button_bg']};
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 16px;
            """
        )
        main_layout.addWidget(self.batch_size_entry)

        # Status Bar
        self.status_bar = QLabel("Status: Ready")
        self.status_bar.setStyleSheet(
            f"""
            background-color: {self.theme['button_bg']};
            color: white;
            padding: 8px 12px;
            font-size: 16px;
            border-radius: 8px;
            """
        )
        self.status_bar.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(self.status_bar)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid {self.theme['button_bg']};
                border-radius: 8px;
                background-color: {self.theme['bg']};
                color: {self.theme['fg']};
                text-align: center;
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {self.theme['button_bg']};
                border-radius: 8px;
            }}
            """
        )
        main_layout.addWidget(self.progress_bar)

        # Counts display
        counts_layout = QHBoxLayout()
        counts_layout.setSpacing(20)
        main_layout.addLayout(counts_layout)

        self.positive_count_label = QLabel("✅ Positive Count: 0")
        self.positive_count_label.setStyleSheet(f"color: {self.theme['fg']}; font-size: 16px; font-weight: 600;")
        counts_layout.addWidget(self.positive_count_label)

        self.negative_count_label = QLabel("❌ Negative Count: 0")
        self.negative_count_label.setStyleSheet(f"color: {self.theme['fg']}; font-size: 16px; font-weight: 600;")
        counts_layout.addWidget(self.negative_count_label)

        # Filter Mode Radios
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(15)
        main_layout.addLayout(mode_layout)

        mode_label = QLabel("⚙️ Select Filter Mode:")
        mode_label.setStyleSheet(f"color: {self.theme['button_fg']}; font-weight: 700; font-size: 16px;")
        mode_layout.addWidget(mode_label)

        self.mode_group = QButtonGroup(self)

        self.rp_mode_radio = QRadioButton("🔍 RP Filter")
        self.rp_mode_radio.setChecked(True)
        self.rp_mode_radio.setStyleSheet(f"color: {self.theme['button_fg']}; font-size: 16px; font-weight: 600;")
        self.rp_mode_radio.toggled.connect(self.update_filter_mode)
        self.mode_group.addButton(self.rp_mode_radio)
        mode_layout.addWidget(self.rp_mode_radio)

        self.normal_mode_radio = QRadioButton("🛡️ Normal Filter")
        self.normal_mode_radio.setStyleSheet(f"color: {self.theme['button_fg']}; font-size: 16px; font-weight: 600;")
        self.normal_mode_radio.toggled.connect(self.update_filter_mode)
        self.mode_group.addButton(self.normal_mode_radio)
        mode_layout.addWidget(self.normal_mode_radio)

        # Classification logic label
        self.class_logic_label = QLabel(self.get_classification_logic_text())
        self.class_logic_label.setStyleSheet(
            f"color: {self.theme['button_fg']}; font-style: italic; font-size: 15px; margin-top: 12px;"
        )
        main_layout.addWidget(self.class_logic_label)

        self.resize(650, 520)

    def browse_input_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select JSONL Files", filter="JSONL Files (*.jsonl)")
        if files:
            self.input_files = files
            self.input_file_entry.setText(", ".join(files))

    def start_filtering(self):
        try:
            threshold = float(self.threshold_entry.text())
            batch_size = int(self.batch_size_entry.text())
        except ValueError:
            self.update_status("⚠️ Invalid threshold or batch size.")
            return

        if not self.input_files:
            self.update_status("⚠️ No input files selected.")
            return

        self.filter_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.thread = FilterThread(self.input_files, threshold, batch_size)
        self.thread.status_update.connect(self.update_status)
        self.thread.counts_update.connect(self.update_counts)
        self.thread.progress_update.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.filtering_finished)
        self.thread.start()

    def filtering_finished(self):
        self.update_status("✅ Filtering complete.")
        self.progress_bar.setValue(100)
        self.filter_button.setEnabled(True)

    def update_status(self, message):
        self.status_bar.setText(f"Status: {message}")

    def update_counts(self, positive_count, negative_count):
        self.positive_count_label.setText(f"✅ Positive Count: {positive_count}")
        self.negative_count_label.setText(f"❌ Negative Count: {negative_count}")

    def update_filter_mode(self):
        mode = FILTER_MODE_RP if self.rp_mode_radio.isChecked() else FILTER_MODE_NORMAL
        set_filter_mode(mode)
        self.positive_count_label.setText("✅ Positive Count: 0")
        self.negative_count_label.setText("❌ Negative Count: 0")
        self.class_logic_label.setText(self.get_classification_logic_text())
        self.update_status("⚙️ Filter mode switched.")

    def get_classification_logic_text(self):
        if self.rp_mode_radio.isChecked():
            return "🔍 RP Filter: Class 0 = Refusal (positive), Class 1 = Safe"
        else:
            return "🛡️ Normal Filter: Class 1 = Refusal (positive), Class 0 = Safe"


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    from App.Other.Theme import Theme

    window = BinaryClassificationApp(Theme.DARK)
    window.show()
    sys.exit(app.exec_())
