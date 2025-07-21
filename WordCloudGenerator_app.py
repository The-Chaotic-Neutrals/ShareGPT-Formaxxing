import sys
import os
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, QObject
from PIL import Image
from WordCloudGenerator import WordCloudGenerator
from theme import Theme  # <--- Import your exact theme here

class GenerateWordCloudApp(QObject):
    status_updated = pyqtSignal(str)
    image_ready = pyqtSignal(object)  # PIL Image

    def __init__(self, parent, theme):
        super().__init__()
        self.theme = theme
        self.icon_path = "icon.ico"
        self.generator = WordCloudGenerator(theme, self.update_status)
        self.window = QMainWindow(parent)
        self.window.setWindowTitle("Generate Word Cloud")
        self.status_updated.connect(self._update_status)
        self.image_ready.connect(self.show_image)
        self.setup_ui()
        self.set_icon()
        self.window.show()

    def setup_ui(self):
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 1)

        label = QLabel("Select JSONL File:")
        label.setStyleSheet(f"background-color: {self.theme['bg']}; color: {self.theme['fg']};")
        layout.addWidget(label, 0, 0)

        self.entry_file = QLineEdit()
        self.entry_file.setStyleSheet(f"background-color: {self.theme['entry_bg']}; color: {self.theme['entry_fg']};")
        layout.addWidget(self.entry_file, 0, 1)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_file)
        browse_button.setStyleSheet(f"background-color: {self.theme['button_bg']}; color: {self.theme['button_fg']};")
        layout.addWidget(browse_button, 0, 2)

        generate_button = QPushButton("Generate Word Cloud")
        generate_button.clicked.connect(self.start_wordcloud_generation)
        generate_button.setStyleSheet(f"background-color: {self.theme['button_bg']}; color: {self.theme['button_fg']};")
        layout.addWidget(generate_button, 1, 0, 1, 3)

        self.status_label = QLabel()
        self.status_label.setStyleSheet(f"background-color: {self.theme['bg']}; color: {self.theme['fg']};")
        layout.addWidget(self.status_label, 2, 0, 1, 3)

        central.setLayout(layout)
        central.setStyleSheet(f"background-color: {self.theme['bg']};")

    def set_icon(self):
        if os.path.exists(self.icon_path):
            self.window.setWindowIcon(QIcon(self.icon_path))
        else:
            print("Icon file not found.")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self.window, "Select JSONL File", "", "JSON Lines files (*.jsonl)")
        if file_path:
            self.entry_file.setText(file_path)

    def start_wordcloud_generation(self):
        file_path = self.entry_file.text()
        if os.path.isfile(file_path):
            threading.Thread(target=self.generate_wordcloud, args=(file_path,), daemon=True).start()
        else:
            self.update_status("Input Error: The file does not exist.")

    def generate_wordcloud(self, file_path):
        pil_image = self.generator.generate_wordcloud(file_path)
        if pil_image:
            self.image_ready.emit(pil_image)

    def show_image(self, pil_image):
        data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)

        top = QMainWindow(self.window)
        top.setWindowTitle("Word Cloud")
        top.resize(616, 308)
        top.setStyleSheet(f"background-color: {self.theme['bg']};")
        if os.path.exists(self.icon_path):
            top.setWindowIcon(QIcon(self.icon_path))

        label = QLabel()
        label.setPixmap(pix)
        top.setCentralWidget(label)
        top.show()

    def update_status(self, message):
        self.status_updated.emit(message)

    def _update_status(self, message):
        self.status_label.setText(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    generate_app = GenerateWordCloudApp(None, Theme.DARK)  # <--- Using your exact theme here
    sys.exit(app.exec_())
