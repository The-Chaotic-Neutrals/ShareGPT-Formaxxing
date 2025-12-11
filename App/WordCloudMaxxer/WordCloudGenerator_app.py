import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import pyqtSignal, QObject
from PIL import Image
from App.WordCloudMaxxer.wordCloudGenerator import WordCloudGenerator
from App.Other.Theme import Theme

class GenerateWordCloudApp(QObject):
    status_updated = pyqtSignal(str)
    image_ready = pyqtSignal(object) # PIL Image
    def __init__(self, parent, theme):
        super().__init__()
        self.theme = theme
        from pathlib import Path
        self.icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        self.generator = WordCloudGenerator(theme, self.update_status)
        self.window = QMainWindow(parent)
        self.window.setWindowTitle("☁️ Generate Word Cloud")
        self.status_updated.connect(self._update_status)
        self.image_ready.connect(self.show_image)
        self.setup_ui()
        self.set_icon()
        self.apply_theme()
        self.add_background_to_main()
        self.window.show()
    
    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.theme.get('bg', '#000000')))
        palette.setColor(QPalette.WindowText, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Base, QColor(self.theme.get('entry_bg', '#000000')))
        palette.setColor(QPalette.Text, QColor(self.theme.get('entry_fg', '#ffffff')))
        palette.setColor(QPalette.Button, QColor(self.theme.get('button_bg', '#1e90ff')))
        palette.setColor(QPalette.ButtonText, QColor(self.theme.get('button_fg', '#ffffff')))
        self.window.setPalette(palette)
        # Stylesheet with explicit button protection and hover effect
        stylesheet = f"""
            QMainWindow, QWidget#centralWidget {{
                background-color: rgba(0, 0, 0, 0.8);
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 14px;
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
            }}
            QPushButton:hover {{
                background-color: #4682b4;
                color: #ffffff;
                border: 1px solid #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #2f4f4f;
            }}
            QLineEdit {{
                background-color: {self.theme.get('entry_bg', '#000000')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                padding: 4px;
                font-size: 14px;
            }}
            QLabel {{
                background-color: transparent;
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 14px;
            }}
        """
        self.window.setStyleSheet(stylesheet)
    
    def setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.window.setCentralWidget(central)
        layout = QGridLayout()
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 1)
        # File selection
        self.label = QLabel("📂 Select JSONL File:")
        layout.addWidget(self.label, 0, 0)
        self.entry_file = QLineEdit()
        layout.addWidget(self.entry_file, 0, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.select_file)
        layout.addWidget(browse_button, 0, 2)
        # Generate button
        generate_button = QPushButton("🚀 Generate Word Cloud")
        generate_button.clicked.connect(self.start_wordcloud_generation)
        layout.addWidget(generate_button, 1, 0, 1, 3)
        # Status label
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label, 2, 0, 1, 3)
        central.setLayout(layout)
    
    def add_background_to_main(self):
        central = self.window.centralWidget()
        bg_class = self.theme.get('background_widget_class')
        if bg_class:
            bg = bg_class(central)
            bg.lower()
            original_resize = central.resizeEvent if hasattr(central, 'resizeEvent') else None
            def new_resize(event):
                bg.resize(central.size())
                if original_resize:
                    original_resize(event)
                else:
                    super(QWidget, central).resizeEvent(event)
            central.resizeEvent = new_resize
            bg.resize(central.size())
    
    def set_icon(self):
        if self.icon_path.exists():
            self.window.setWindowIcon(QIcon(str(self.icon_path)))
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self.window, "Select JSONL File", "", "JSON Lines files (*.jsonl)")
        if file_path:
            self.entry_file.setText(file_path)
    
    def start_wordcloud_generation(self):
        file_path = self.entry_file.text()
        if os.path.isfile(file_path):
            threading.Thread(target=self.generate_wordcloud, args=(file_path,), daemon=True).start()
        else:
            self.update_status("❌ Input Error: The file does not exist.")
    
    def generate_wordcloud(self, file_path):
        pil_image = self.generator.generate_wordcloud(file_path, width=1600, height=1200)
        if pil_image:
            self.image_ready.emit(pil_image)
    
    def show_image(self, pil_image):
        data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        qim = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)
        top = QMainWindow(self.window)
        top.setWindowTitle("☁️ Word Cloud")
        top.resize(800, 600)
        # Apply same stylesheet to popup for consistency
        stylesheet = f"""
            QMainWindow, QWidget#centralWidget {{
                background-color: rgba(0, 0, 0, 0.8);
            }}
            QLabel {{
                background-color: transparent;
            }}
        """
        top.setStyleSheet(stylesheet)
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel()
        label.setPixmap(pix)
        layout.addWidget(label)
        top.setCentralWidget(central_widget)
        bg_class = self.theme.get('background_widget_class')
        if bg_class:
            bg = bg_class(central_widget)
            bg.lower()
            original_resize = central_widget.resizeEvent if hasattr(central_widget, 'resizeEvent') else None
            def new_resize(event):
                bg.resize(central_widget.size())
                if original_resize:
                    original_resize(event)
                else:
                    super(QWidget, central_widget).resizeEvent(event)
            central_widget.resizeEvent = new_resize
            bg.resize(central_widget.size())
        top.show()
    
    def update_status(self, message):
        self.status_updated.emit(message)
    
    def _update_status(self, message):
        self.status_label.setText(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    generate_app = GenerateWordCloudApp(None, Theme.DARK)
    sys.exit(app.exec_())