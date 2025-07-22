import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import logging
logging.getLogger("faiss").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message=".*load_model does not return WordVectorModel.*")

import tkinter as tk
from tkinter import ttk
from theme import Theme
from dataset_converter_app import DatasetConverterApp
from datamaxxer_app import DataMaxxerApp
from DeslopTool_app import DeslopToolApp
from WordCloudGenerator_app import GenerateWordCloudApp
from ui_elements import UIElements
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
        self.style = ttk.Style()
        self.qt_windows = []  # keep refs to PyQt windows alive
        self.setup_ui()

        # Ensure QApplication instance is created once at startup
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)

        self.start_qt_event_loop()

    def setup_ui(self):
        self.root.configure(bg=self.theme.get('bg', '#000000'))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.set_icon()
        self.create_options_ui()
        self.update_ui_styles()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception as e:
                print(f"Could not set main icon: {e}")
        else:
            print("Icon file not found.")

    # All these pass the main root window
    def open_music_player_app(self):
        MusicPlayerApp(self.root, self.theme)

    def open_dataset_converter_app(self):
        DatasetConverterApp(self.root, self.theme)

    def open_filter_app(self):
        DataMaxxerApp(self.root, self.theme)

    def open_deslop_tool(self):
        DeslopToolApp(self.root, self.theme)

    def open_binary_classification_app(self):
        BinaryClassificationApp(self.root, self.theme)

    def open_deduplication_app(self):
        DeduplicationApp(self.root, self.theme)

    def open_ngram_analyzer_app(self):
        win = NgramAnalyzerApp(self.theme)
        win.setWindowTitle("N-gramlyzer")
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

    def create_options_ui(self):
        options_frame = tk.Frame(self.root, bg=self.theme.get('bg', '#000000'))
        options_frame.grid(row=0, column=0, columnspan=14, pady=20, sticky='ew')

        for i in range(14):
            options_frame.columnconfigure(i, weight=1)

        buttons = [
            ("MusicMaxxer", self.open_music_player_app),
            ("DataMaxxer", self.open_filter_app),
            ("DeslopMancer", self.open_deslop_tool),
            ("Formaxxer", self.open_dataset_converter_app),
            ("WordCloudMaxxer", self.open_wordcloud_generator_app),
            ("RefusalMancer", self.open_binary_classification_app),
            ("DedupMancer", self.open_deduplication_app),
            ("N-gramlyzer", self.open_ngram_analyzer_app),
            ("GrammarMaxxer", self.open_text_correction_app),
            ("SafetensorMaxxer", self.open_safetensormaxxer_app),
            ("LineMancer", self.open_linemancer_app),
            ("ParquetMaxxer", self.open_parquetmaxxer_app),
            ("EnglishMaxxer", self.open_englishfilter_app),
            ("TokenMaxxer", self.open_tokenmaxxer_app),
        ]

        for index, (text, command) in enumerate(buttons):
            button = tk.Button(
                options_frame,
                text=text,
                command=command,
                bg=self.theme.get('button_bg', '#1e90ff'),
                fg=self.theme.get('button_fg', '#ffffff')
            )
            button.grid(row=0, column=index, pady=10, padx=5, sticky='ew')

    def update_ui_styles(self):
        self.style.configure('TButton',
                             background=self.theme.get('button_bg', '#1e90ff'),
                             foreground=self.theme.get('button_fg', '#ffffff'))
        self.style.configure('TLabel',
                             background=self.theme.get('bg', '#000000'),
                             foreground=self.theme.get('fg', '#1e90ff'))
        self.style.configure('TEntry',
                             fieldbackground=self.theme.get('entry_bg', '#000000'),
                             foreground=self.theme.get('entry_fg', '#ffffff'))
        for widget in self.root.winfo_children():
            if widget.winfo_class() == 'Frame':
                widget.configure(bg=self.theme.get('bg', '#000000'))
            elif widget.winfo_class() in ['Label', 'Button']:
                widget.configure(bg=self.theme.get('bg', '#000000'),
                                 fg=self.theme.get('fg', '#1e90ff'))
            elif widget.winfo_class() == 'Entry':
                widget.configure(bg=self.theme.get('entry_bg', '#000000'),
                                 fg=self.theme.get('entry_fg', '#ffffff'))

    def start_qt_event_loop(self):
        def process_qt_events():
            self.qt_app.processEvents()
            self.root.after(10, process_qt_events)

        self.root.after(10, process_qt_events)


if __name__ == "__main__":
    root = tk.Tk()
    app = UIManager(root)
    root.mainloop()
