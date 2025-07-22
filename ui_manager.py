import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import logging
logging.getLogger("faiss").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message=".*load_model does not return WordVectorModel.*")

import tkinter as tk
from theme import Theme
from dataset_converter_app import DatasetConverterApp
from datamaxxer_app import DataMaxxerApp
from DeslopTool_app import DeslopToolApp  # PyQt5 version of DeslopToolApp
from WordCloudGenerator_app import GenerateWordCloudApp
from music_player_app import MusicPlayerApp
from binary_classification_app import BinaryClassificationApp
from deduplication_app import DeduplicationApp
from ngram_analyzer_app import NgramAnalyzerApp
from grammar_maxxer_app import GrammarMaxxerApp
from safetensormaxxer_app import SafetensorMaxxerApp
from linemancer_app import LineMancerFrame
from parquetmaxxer_app import ParquetMaxxer
from english_filter_app import EnglishFilterApp
from tokenmaxxerv3_app import TokenMaxxerV3App

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui


class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK
        self.qt_windows = []  # keep refs to PyQt windows alive
        self.setup_ui()

        # Ensure QApplication instance is created once at startup
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)

        self.start_qt_event_loop()

    def setup_ui(self):
        self.root.configure(bg=self.theme.get('bg', '#000000'))
        self.set_icon()
        self.create_options_ui()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Could not set main icon: {e}")

    def open_music_player_app(self):
        MusicPlayerApp(self.root, self.theme)

    def open_dataset_converter_app(self):
        DatasetConverterApp(self.root, self.theme)

    def open_filter_app(self):
        win = DataMaxxerApp(self.theme)
        win.setWindowTitle("DataMaxxer")
        win.show()
        self.qt_windows.append(win)

    def open_deslop_tool(self):
        from DeslopTool_app import DeslopToolApp  # PyQt5 version

        win = DeslopToolApp()  # Create the PyQt5 window
        win.setWindowTitle("DeslopMancer")

        # Apply dark theme background and text colors
        bg = self.theme.get('bg', '#222222')
        fg = self.theme.get('fg', '#ffffff')
        win.setStyleSheet(f"background-color: {bg}; color: {fg};")

        win.show()
        self.qt_windows.append(win)

    def open_binary_classification_app(self):
        BinaryClassificationApp(self.root, self.theme)

    def open_deduplication_app(self):
        DeduplicationApp(self.root, self.theme)

    def open_ngram_analyzer_app(self):
        win = NgramAnalyzerApp(self.theme)
        win.setWindowTitle("N-GraMancer")
        win.show()
        self.qt_windows.append(win)

    def open_text_correction_app(self):
        GrammarMaxxerApp(self.root, self.theme)

    def open_safetensormaxxer_app(self):
        win = tk.Toplevel(self.root)
        win.title("SafetensorMaxxer")
        SafetensorMaxxerApp(win, self.theme)

    def open_linemancer_app(self):
        win = LineMancerFrame()
        win.setWindowTitle("LineMancer")
        win.resize(640, 520)
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            win.setWindowIcon(QtGui.QIcon(icon_path))
        win.show()
        self.qt_windows.append(win)

    def open_parquetmaxxer_app(self):
        win = ParquetMaxxer()
        win.setWindowTitle("ParquetMaxxer")
        win.show()
        self.qt_windows.append(win)

    def open_englishfilter_app(self):
        win = EnglishFilterApp()
        win.setWindowTitle("EnglishMaxxer")
        win.show()
        self.qt_windows.append(win)

    def open_tokenmaxxer_app(self):
        win = TokenMaxxerV3App()
        win.setWindowTitle("TokenMaxxer")
        win.show()
        self.qt_windows.append(win)

    def open_wordcloud_generator_app(self):
        app = GenerateWordCloudApp(None, self.theme)
        if hasattr(app, 'window'):
            app.window.setWindowTitle("WordCloudMaxxer")
            app.window.show()
            self.qt_windows.append(app.window)

    def _add_section(self, parent, title, buttons):
        """Create a vertical list section with header and rounded buttons"""
        section_frame = tk.Frame(parent, bg=self.theme['bg'])
        section_frame.pack(side='left', fill='both', expand=True, padx=20)

        header = tk.Label(section_frame, text=title,
                          bg=self.theme['bg'], fg=self.theme['fg'],
                          font=("Helvetica", 16, "bold"), pady=10)
        header.pack(fill='x', pady=(0, 10))

        for text, command in buttons:
            btn = tk.Button(
                section_frame, text=text, command=command,
                bg=self.theme.get('button_bg', '#1e90ff'),
                fg=self.theme.get('button_fg', '#ffffff'),
                relief='flat', padx=10, pady=8,
                font=("Helvetica", 12, "bold"),
                activebackground="#666666",
                activeforeground=self.theme.get('button_fg', '#ffffff'),
                bd=0,
                highlightthickness=0
            )
            btn.pack(fill='x', pady=5)

    def create_options_ui(self):
        container = tk.Frame(self.root, bg=self.theme['bg'])
        container.pack(fill='both', expand=True, padx=30, pady=30)

        self._add_section(container, "🛠 Maxxer Tools", [
            ("MusicMaxxer", self.open_music_player_app),
            ("DataMaxxer", self.open_filter_app),
            ("WordCloudMaxxer", self.open_wordcloud_generator_app),
            ("SafetensorMaxxer", self.open_safetensormaxxer_app),
            ("ParquetMaxxer", self.open_parquetmaxxer_app),
            ("EnglishMaxxer", self.open_englishfilter_app),
            ("TokenMaxxer", self.open_tokenmaxxer_app),
            ("ForMaxxer", self.open_dataset_converter_app),
            ("GrammarMaxxer", self.open_text_correction_app),
        ])

        self._add_section(container, "⚒️‍ Mancer Tools", [
            ("DeslopMancer", self.open_deslop_tool),
            ("RefusalMancer", self.open_binary_classification_app),
            ("DedupMancer", self.open_deduplication_app),
            ("LineMancer", self.open_linemancer_app),
            ("N-GraMancer", self.open_ngram_analyzer_app),
        ])

    def start_qt_event_loop(self):
        def process_qt_events():
            self.qt_app.processEvents()
            self.root.after(10, process_qt_events)

        self.root.after(10, process_qt_events)


if __name__ == "__main__":
    root = tk.Tk()
    app = UIManager(root)
    root.mainloop()
