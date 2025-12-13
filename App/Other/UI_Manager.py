import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QTabWidget, QMainWindow
)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QTimer

# Import your tools
from App.Other.Theme import Theme
from App.ForMaxxer.ForMaxxer_app import DatasetConverterApp
from App.DataMaxxer.DataMaxxer_app import DataMaxxerApp
from App.DeslopMancer.DeslopTool_app import DeslopToolApp
from App.WordCloudMaxxer.WordCloudGenerator_app import GenerateWordCloudApp
from App.RefusalMancer.binary_classification_app import BinaryClassificationApp
from App.DedupeMancer.DedupeMancer_app import DeduplicationApp
# Import N-GraMancer using workaround for hyphen in directory name
import sys
import importlib.util
from pathlib import Path
_ngramancer_app_path = Path(__file__).parent.parent / "N-GraMancer" / "N-GraMancer_app.py"
spec = importlib.util.spec_from_file_location("N_GraMancer_app", _ngramancer_app_path)
N_GraMancer_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N_GraMancer_app)
NgramAnalyzerApp = N_GraMancer_app.NgramAnalyzerApp
from App.GrammarMaxxer.GrammarMaxxer_app import GrammarMaxxerApp
from App.SafetensorMaxxer.safetensormaxxer_app import SafetensorMaxxerApp
from App.LineMancer.LineMancer_app import LineMancerFrame
from App.ParquetMaxxer.parquetmaxxer_app import ParquetMaxxer
from App.EnglishMaxxer.EnglishMaxxer_app import EnglishFilterApp
from App.TokenMaxxer.tokenmaxxerv3_app import TokenMaxxerV3App
from App.SynthMaxxer.synthmaxxer_app import MainWindow as SynthMaxxerApp
from App.MergeMaxxer.MergeMaxxer import MergeKitGUI


class UIManager(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or Theme.DARK
        self.qt_windows = []
        self.setWindowTitle("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.resize(800, 480)
        self.icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"

        # Use background widget from theme
        bg_class = self.theme.get('background_widget_class', None)
        if bg_class:
            self.background_widget = bg_class(self)
            self.background_widget.lower()
        else:
            self.background_widget = None

        self.apply_theme()
        self.set_icon()
        self.setup_ui()

    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if self.background_widget:
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
                padding: 14px 14px;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                min-width: 120px;
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
        if self.icon_path.exists():
            self.setWindowIcon(QIcon(str(self.icon_path)))

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Main title
        title_label = QLabel(self.windowTitle())
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
            ("SynthMaxxer", self.open_synthmaxxer_app),
            ("MergeMaxxer", self.open_mergemaxxer_app),
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

    def add_background_to_window(self, win):
        bg_class = self.theme.get('background_widget_class')
        if bg_class:
            if isinstance(win, QMainWindow):
                central = win.centralWidget()
                if central:
                    bg = bg_class(central)
                    bg.lower()
                    bg.setGeometry(0, 0, central.width() or 800, central.height() or 600)
                    # Force transparent background on central to show animated bg
                    current_ss = central.styleSheet()
                    if current_ss:
                        # Replace various background color patterns
                        current_ss = current_ss.replace("background-color: #000000;", "background-color: transparent;")
                        current_ss = current_ss.replace("background-color:#000000;", "background-color: transparent;")
                        current_ss = current_ss.replace("background-color: #05050F;", "background-color: transparent;")
                        current_ss = current_ss.replace("background-color:#05050F;", "background-color: transparent;")
                    else:
                        current_ss = "background-color: transparent;"
                    central.setStyleSheet(current_ss)
                    # Patch resizeEvent on central - use a flag to prevent recursion
                    original_resize = central.resizeEvent if hasattr(central, 'resizeEvent') else None
                    _resizing = False
                    def new_resize(a0):
                        nonlocal _resizing
                        if _resizing:
                            if original_resize:
                                original_resize(a0)
                            else:
                                QWidget.resizeEvent(central, a0)
                            return
                        _resizing = True
                        try:
                            if bg and central:
                                # Use setGeometry instead of resize to avoid triggering events
                                bg.setGeometry(0, 0, central.width(), central.height())
                        finally:
                            _resizing = False
                        if original_resize:
                            original_resize(a0)
                        else:
                            QWidget.resizeEvent(central, a0)
                    central.resizeEvent = new_resize
                    # Initial resize with delay to ensure layout is complete
                    QTimer.singleShot(50, lambda: bg.setGeometry(0, 0, central.width() or 800, central.height() or 600) if bg and central else None)
                else:
                    # If no central, fallback to win
                    bg = bg_class(win)
                    bg.lower()
                    bg.setGeometry(0, 0, win.width() or 800, win.height() or 600)
                    win.setStyleSheet("background-color: transparent;")
                    original_resize = win.resizeEvent if hasattr(win, 'resizeEvent') else None
                    _resizing = False
                    def new_resize(a0):
                        nonlocal _resizing
                        if _resizing:
                            if original_resize:
                                original_resize(a0)
                            else:
                                QMainWindow.resizeEvent(win, a0)
                            return
                        _resizing = True
                        try:
                            if bg:
                                # Use setGeometry instead of resize to avoid triggering events
                                bg.setGeometry(0, 0, win.width(), win.height())
                        finally:
                            _resizing = False
                        if original_resize:
                            original_resize(a0)
                        else:
                            QMainWindow.resizeEvent(win, a0)
                    win.resizeEvent = new_resize
                    QTimer.singleShot(50, lambda: bg.setGeometry(0, 0, win.width() or 800, win.height() or 600) if bg else None)
            else:
                bg = bg_class(win)
                bg.lower()
                bg.setGeometry(0, 0, win.width() or 800, win.height() or 600)
                # Force transparent background on win
                current_ss = win.styleSheet()
                if current_ss:
                    current_ss = current_ss.replace("background-color: #000000;", "background-color: transparent;")
                    current_ss = current_ss.replace("background-color:#000000;", "background-color: transparent;")
                    current_ss = current_ss.replace("background-color: #05050F;", "background-color: transparent;")
                    current_ss = current_ss.replace("background-color:#05050F;", "background-color: transparent;")
                else:
                    current_ss = "background-color: transparent;"
                win.setStyleSheet(current_ss)
                # Handle resize for QWidget - use a flag to prevent recursion
                original_resize = win.resizeEvent if hasattr(win, 'resizeEvent') else None
                _resizing = False
                def new_resize(a0):
                    nonlocal _resizing
                    if _resizing:
                        if original_resize:
                            original_resize(a0)
                        else:
                            QWidget.resizeEvent(win, a0)
                        return
                    _resizing = True
                    try:
                        if bg:
                            # Use setGeometry instead of resize to avoid triggering events
                            bg.setGeometry(0, 0, win.width(), win.height())
                    finally:
                        _resizing = False
                    if original_resize:
                        original_resize(a0)
                    else:
                        QWidget.resizeEvent(win, a0)
                win.resizeEvent = new_resize
                # Initial resize with delay to ensure layout is complete
                QTimer.singleShot(50, lambda: bg.setGeometry(0, 0, win.width() or 800, win.height() or 600) if bg else None)

    # ---- Window Launchers ----
    def open_dataset_converter_app(self):
        win = DatasetConverterApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_filter_app(self):
        win = DataMaxxerApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_deslop_tool(self):
        win = DeslopToolApp()
        win.setStyleSheet(f"background-color: transparent; color: {self.theme.get('text_fg')};")
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_binary_classification_app(self):
        win = BinaryClassificationApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_deduplication_app(self):
        win = DeduplicationApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_ngram_analyzer_app(self):
        win = NgramAnalyzerApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_text_correction_app(self):
        win = GrammarMaxxerApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_safetensormaxxer_app(self):
        win = SafetensorMaxxerApp(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_linemancer_app(self):
        win = LineMancerFrame()
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        win.setStyleSheet(f"background-color: transparent; color: {self.theme.get('text_fg')};")
        win.resize(640, 520)
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_parquetmaxxer_app(self):
        win = ParquetMaxxer()
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        win.setStyleSheet(f"background-color: transparent; color: {self.theme.get('text_fg')};")
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_englishfilter_app(self):
        win = EnglishFilterApp()
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        win.setStyleSheet(f"background-color: transparent; color: {self.theme.get('text_fg')};")
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_tokenmaxxer_app(self):
        win = TokenMaxxerV3App()
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        win.setStyleSheet(f"background-color: transparent; color: {self.theme.get('text_fg')};")
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_wordcloud_generator_app(self):
        app = GenerateWordCloudApp(None, self.theme)
        if hasattr(app, 'window'):
            if self.icon_path.exists():
                app.window.setWindowIcon(QIcon(str(self.icon_path)))
            self.add_background_to_window(app.window)
            app.window.show()
            self.qt_windows.append(app.window)

    def open_synthmaxxer_app(self):
        win = SynthMaxxerApp()
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)

    def open_mergemaxxer_app(self):
        win = MergeKitGUI(self.theme)
        if self.icon_path.exists():
            win.setWindowIcon(QIcon(str(self.icon_path)))
        self.add_background_to_window(win)
        win.show()
        self.qt_windows.append(win)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIManager(Theme.DARK)
    window.show()
    sys.exit(app.exec_())