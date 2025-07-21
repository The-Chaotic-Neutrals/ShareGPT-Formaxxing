from PyQt5 import QtWidgets, QtGui, QtCore
from theme import Theme
from linemancer import LineMancerCore
import os

class LineMancerFrame(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core = LineMancerCore()
        self.theme = Theme.DARK

        self.setWindowTitle("✨ LineMancer — JSONL Tool ✨")
        self.setMinimumSize(640, 520)

        self.mode = "split"
        self.input_paths = []

        self.input_path = ""
        self.lines_per_file = 1000

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        title_label = QtWidgets.QLabel("✨ LineMancer — JSONL Tool ✨")
        title_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        main_layout.addWidget(title_label, alignment=QtCore.Qt.AlignLeft)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Mode:")
        mode_label.setFont(QtGui.QFont("Segoe UI", 11))
        mode_layout.addWidget(mode_label)

        self.mode_group = QtWidgets.QButtonGroup(self)
        modes = [("split", "🪓 Split"), ("merge", "🧵 Merge"), ("shuffle", "🎲 Shuffle")]
        for mode_val, mode_text in modes:
            btn = QtWidgets.QRadioButton(mode_text)
            btn.setFont(QtGui.QFont("Segoe UI", 10))
            btn.mode_val = mode_val
            self.mode_group.addButton(btn)
            mode_layout.addWidget(btn)
            if mode_val == self.mode:
                btn.setChecked(True)
        self.mode_group.buttonClicked.connect(self.on_mode_change)
        main_layout.addLayout(mode_layout)

        self.stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stack)

        self.build_split_ui()
        self.build_merge_ui()
        self.build_shuffle_ui()

        self.render_mode()

    def apply_theme(self):
        palette = self.palette()
        bg = QtGui.QColor(self.theme.get('bg', '#2e2e2e'))
        fg = QtGui.QColor(self.theme.get('fg', '#d4af37'))
        entry_bg = QtGui.QColor(self.theme.get('entry_bg', '#3b3b3b'))
        button_bg = QtGui.QColor(self.theme.get('button_bg', '#3b3b3b'))
        button_fg = QtGui.QColor(self.theme.get('button_fg', '#d4af37'))

        palette.setColor(QtGui.QPalette.Window, bg)
        palette.setColor(QtGui.QPalette.WindowText, fg)
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg.name()};
                color: {fg.name()};
            }}
            QLineEdit, QTextEdit {{
                background-color: {entry_bg.name()};
                color: {fg.name()};
                border: 1px solid {fg.name()};
                padding: 4px;
            }}
            QPushButton {{
                background-color: {button_bg.name()};
                color: {button_fg.name()};
                border: 1px solid {fg.name()};
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {fg.name()};
                color: {button_bg.name()};
            }}
            QRadioButton {{
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
        """)

    def on_mode_change(self, button):
        self.mode = button.mode_val
        self.render_mode()

    def render_mode(self):
        modes = {"split": 0, "merge": 1, "shuffle": 2}
        self.stack.setCurrentIndex(modes[self.mode])

    def build_split_ui(self):
        split_widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(split_widget)

        self.input_path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_input)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.input_path_edit)
        hbox.addWidget(browse_btn)
        layout.addRow("Input JSONL File:", hbox)

        self.lines_per_file_spin = QtWidgets.QSpinBox()
        self.lines_per_file_spin.setRange(1, 1000000)
        self.lines_per_file_spin.setValue(self.lines_per_file)
        layout.addRow("Lines per file:", self.lines_per_file_spin)

        split_btn = QtWidgets.QPushButton("Split JSONL")
        split_btn.clicked.connect(self.split_jsonl)
        layout.addRow(split_btn)

        self.stack.addWidget(split_widget)

    def build_merge_ui(self):
        merge_widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(merge_widget)

        self.merge_path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_inputs)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.merge_path_edit)
        hbox.addWidget(browse_btn)
        layout.addRow("Input JSONL Files:", hbox)

        merge_btn = QtWidgets.QPushButton("Merge JSONL Files")
        merge_btn.clicked.connect(self.merge_jsonl)
        layout.addRow(merge_btn)

        self.stack.addWidget(merge_widget)

    def build_shuffle_ui(self):
        shuffle_widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(shuffle_widget)

        self.input_path_edit_shuffle = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_input_shuffle)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.input_path_edit_shuffle)
        hbox.addWidget(browse_btn)
        layout.addRow("Input JSONL File:", hbox)

        shuffle_btn = QtWidgets.QPushButton("Shuffle JSONL Lines")
        shuffle_btn.clicked.connect(self.shuffle_jsonl)
        layout.addRow(shuffle_btn)

        self.stack.addWidget(shuffle_widget)

    def browse_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Input JSONL File", "", "JSONL Files (*.jsonl);;All Files (*)")
        if path:
            self.input_path_edit.setText(path)
            self.input_path = path

    def browse_inputs(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Input JSONL Files", "", "JSONL Files (*.jsonl);;All Files (*)")
        if paths:
            self.input_paths = paths
            self.merge_path_edit.setText(", ".join(paths))

    def browse_input_shuffle(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Input JSONL File", "", "JSONL Files (*.jsonl);;All Files (*)")
        if path:
            self.input_path_edit_shuffle.setText(path)

    def split_jsonl(self):
        try:
            count = self.core.split_jsonl(
                self.input_path_edit.text(),
                lines_per_file=self.lines_per_file_spin.value()
            )
            QtWidgets.QMessageBox.information(self, "Success", f"Split complete. {count} parts saved.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def merge_jsonl(self):
        try:
            paths = self.merge_path_edit.text().split(', ')
            output_path, total = self.core.merge_jsonl(paths)
            QtWidgets.QMessageBox.information(self, "Success", f"Merged {total} lines into:\n{output_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def shuffle_jsonl(self):
        try:
            output_path = self.core.shuffle_jsonl(self.input_path_edit_shuffle.text())
            QtWidgets.QMessageBox.information(self, "Success", f"Shuffled lines saved to:\n{output_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = LineMancerFrame()
    icon_path = "icon.ico"
    if os.path.exists(icon_path):
        window.setWindowIcon(QtGui.QIcon(icon_path))
    window.show()
    sys.exit(app.exec_())
