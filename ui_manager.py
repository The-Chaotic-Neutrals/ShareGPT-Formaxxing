import os
import sys
import random
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QTabWidget
)
from PyQt5.QtGui import QIcon, QPalette, QColor, QPainter, QFont
from PyQt5.QtCore import Qt, QTimer, QPointF

# Import your tools
from theme import Theme
from dataset_converter_app import DatasetConverterApp
from datamaxxer_app import DataMaxxerApp
from DeslopTool_app import DeslopToolApp
from WordCloudGenerator_app import GenerateWordCloudApp
from binary_classification_app import BinaryClassificationApp
from deduplication_app import DeduplicationApp
from ngram_analyzer_app import NgramAnalyzerApp
from grammar_maxxer_app import GrammarMaxxerApp
from safetensormaxxer_app import SafetensorMaxxerApp
from linemancer_app import LineMancerFrame
from parquetmaxxer_app import ParquetMaxxer
from english_filter_app import EnglishFilterApp
from tokenmaxxerv3_app import TokenMaxxerV3App


class FloatingPixelsWidget(QWidget):
    def __init__(self, parent=None, pixel_count=100):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.pixels = []
        self.pixel_count = pixel_count
        self.init_pixels()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(30)  # ~33 FPS

    def init_pixels(self):
        self.pixels = []
        w = self.width() or 800
        h = self.height() or 480
        for _ in range(self.pixel_count):
            pos = QPointF(random.uniform(0, w), random.uniform(0, h))
            vel = QPointF(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
            size = random.uniform(1, 3)
            alpha = random.uniform(0.3, 1.0)
            self.pixels.append({'pos': pos, 'vel': vel, 'size': size, 'alpha': alpha})

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.init_pixels()

    def animate(self):
        w = self.width()
        h = self.height()
        for p in self.pixels:
            p['pos'] += p['vel']
            if p['pos'].x() < 0 or p['pos'].x() > w:
                p['vel'].setX(-p['vel'].x())
            if p['pos'].y() < 0 or p['pos'].y() > h:
                p['vel'].setY(-p['vel'].y())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        for p in self.pixels:
            color = QColor(255, 255, 255)
            color.setAlphaF(p['alpha'])
            painter.setBrush(color)
            size = p['size']
            pos = p['pos']
            painter.drawEllipse(pos, size, size)


class UIManager(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or Theme.DARK
        self.qt_windows = []
        self.setWindowTitle("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.resize(800, 480)

        self.background_widget = FloatingPixelsWidget(self)
        self.background_widget.lower()

        self.apply_theme()
        self.set_icon()
        self.setup_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.background_widget.resize(self.size())

    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.theme.get('bg', '#000000')))
        palette.setColor(QPalette.WindowText, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Button, QColor(self.theme.get('button_bg', '#1e90ff')))
        palette.setColor(QPalette.ButtonText, QColor(self.theme.get('button_fg', '#ffffff')))
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                color: {self.theme.get('text_fg', '#ffffff')};
            }}
            QLabel#titleLabel {{
                font-family: 'Times New Roman', serif;
                font-size: 28px;
                font-weight: bold;
                color: #e6e6fa;
                padding: 15px;
                border-bottom: 2px solid #1e90ff;
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border-radius: 12px;
                padding: 12px;
                font-family: 'Arial', sans-serif;
                font-size: 15px;
                font-weight: bold;
                border: 1px solid #4a4a4a;
                min-width: 150px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background-color: #4682b4;
                color: #ffffff;
                border: 1px solid #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #2f4f4f;
            }}
            QFrame {{
                background-color: transparent;
            }}
            QTabWidget::pane {{
                border: 2px solid #333333;
                background-color: rgba(0, 0, 0, 0.8);
            }}
            QTabBar::tab {{
                background-color: #1c2526;
                color: #ffffff;
                padding: 8px 20px;
                font-family: 'Arial', sans-serif;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: #ffffff;
            }}
            QTabBar::tab:hover {{
                background-color: #4682b4;
            }}
        """)

    def set_icon(self):
        icon_path = Path(__file__).parent / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Main title
        title_label = QLabel(self.windowTitle())
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Tab widget
        tab_widget = QTabWidget()

        # Maxxer Tools tab
        maxxer_buttons = [
            ("DataMaxxer", self.open_filter_app),
            ("WordCloudMaxxer", self.open_wordcloud_generator_app),
            ("SafetensorMaxxer", self.open_safetensormaxxer_app),
            ("ParquetMaxxer", self.open_parquetmaxxer_app),
            ("EnglishMaxxer", self.open_englishfilter_app),
            ("TokenMaxxer", self.open_tokenmaxxer_app),
            ("ForMaxxer", self.open_dataset_converter_app),
            ("GrammarMaxxer", self.open_text_correction_app),
        ]
        maxxer_widget = QWidget()
        maxxer_layout = QHBoxLayout()
        maxxer_layout.addStretch()
        maxxer_layout.addWidget(self.create_button_column(maxxer_buttons))
        maxxer_layout.addStretch()
        maxxer_widget.setLayout(maxxer_layout)
        tab_widget.addTab(maxxer_widget, "🛠 Maxxer Tools")

        # Mancer Tools tab
        mancer_buttons = [
            ("DeslopMancer", self.open_deslop_tool),
            ("RefusalMancer", self.open_binary_classification_app),
            ("DedupMancer", self.open_deduplication_app),
            ("LineMancer", self.open_linemancer_app),
            ("N-GraMancer", self.open_ngram_analyzer_app),
        ]
        mancer_widget = QWidget()
        mancer_layout = QHBoxLayout()
        mancer_layout.addStretch()
        mancer_layout.addWidget(self.create_button_column(mancer_buttons))
        mancer_layout.addStretch()
        mancer_widget.setLayout(mancer_layout)
        tab_widget.addTab(mancer_widget, "⚒️ Mancer Tools")

        layout.addWidget(tab_widget)

    def create_button_column(self, buttons):
        column = QVBoxLayout()
        column.setSpacing(8)
        for text, command in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(command)
            column.addWidget(btn)
        frame = QFrame()
        frame.setLayout(column)
        return frame

    # ---- Window Launchers ----
    def open_dataset_converter_app(self):
        win = DatasetConverterApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_filter_app(self):
        win = DataMaxxerApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_deslop_tool(self):
        win = DeslopToolApp()
        win.setStyleSheet(f"background-color: {self.theme.get('bg')}; color: {self.theme.get('text_fg')};")
        win.show()
        self.qt_windows.append(win)

    def open_binary_classification_app(self):
        win = BinaryClassificationApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_deduplication_app(self):
        win = DeduplicationApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_ngram_analyzer_app(self):
        win = NgramAnalyzerApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_text_correction_app(self):
        win = GrammarMaxxerApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_safetensormaxxer_app(self):
        win = SafetensorMaxxerApp(self.theme)
        win.show()
        self.qt_windows.append(win)

    def open_linemancer_app(self):
        win = LineMancerFrame()
        win.resize(640, 520)
        win.show()
        self.qt_windows.append(win)

    def open_parquetmaxxer_app(self):
        win = ParquetMaxxer()
        win.show()
        self.qt_windows.append(win)

    def open_englishfilter_app(self):
        win = EnglishFilterApp()
        win.show()
        self.qt_windows.append(win)

    def open_tokenmaxxer_app(self):
        win = TokenMaxxerV3App()
        win.show()
        self.qt_windows.append(win)

    def open_wordcloud_generator_app(self):
        app = GenerateWordCloudApp(None, self.theme)
        if hasattr(app, 'window'):
            app.window.show()
            self.qt_windows.append(app.window)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIManager(Theme.DARK)
    window.show()
    sys.exit(app.exec_())