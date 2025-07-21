import tkinter as tk
from tkinter import ttk
from theme import Theme
from dataset_converter_app import DatasetConverterApp
from dataset_filter_app import DatasetFilterApp
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
from english_filter_app import EnglishFilterApp        # 💫 NEW IMPORT
from tokenmaxxerv3_app import TokenMaxxerV3App         # PyQt TokenMaxxer

import os
import sys
from PyQt5.QtWidgets import QApplication
import threading

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK
        self.style = ttk.Style()
        self.setup_ui()

    def setup_ui(self):
        self.root.configure(bg=self.theme.get('bg', 'white'))
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

    # Existing open methods
    def open_music_player_app(self): MusicPlayerApp(self.root, self.theme)
    def open_dataset_converter_app(self): DatasetConverterApp(self.root, self.theme)
    def open_filter_app(self): DatasetFilterApp(self.root, self.theme)
    def open_deslop_tool(self): DeslopToolApp(self.root, self.theme)
    def open_binary_classification_app(self): BinaryClassificationApp(self.root, self.theme)
    def open_deduplication_app(self): DeduplicationApp(self.root, self.theme)

    def open_ngram_analyzer_app(self):
        ngram_window = tk.Toplevel(self.root)
        ngram_window.title("N-gram Analyzer App")
        if os.path.exists("icon.ico"):
            try:
                ngram_window.iconbitmap("icon.ico")
            except Exception as e:
                print(f"Could not set icon: {e}")
        NgramAnalyzerApp(ngram_window, self.theme)

    def open_text_correction_app(self): GrammarMaxxerApp(self.root, self.theme)

    def open_safetensormaxxer_app(self):
        safetensor_window = tk.Toplevel(self.root)
        safetensor_window.title("SafetensorMaxxer App")
        if os.path.exists("icon.ico"):
            try:
                safetensor_window.iconbitmap("icon.ico")
            except Exception as e:
                print(f"Could not set icon: {e}")
        SafetensorMaxxerApp(safetensor_window, self.theme)

    # ✨ LineMancer
    def open_linemancer_app(self):
        linemancer_window = tk.Toplevel(self.root)
        linemancer_window.title("LineMancer — JSONL Split/Merge/Shuffle")
        linemancer_window.geometry("620x500")
        linemancer_window.configure(bg="#2e2e2e")
        if os.path.exists("icon.ico"):
            try:
                linemancer_window.iconbitmap("icon.ico")
            except Exception as e:
                print(f"Could not set icon: {e}")
        frame = LineMancerFrame(linemancer_window)
        frame.pack(fill="both", expand=True)

    # ✨ ParquetMaxxer
    def open_parquetmaxxer_app(self):
        ParquetMaxxer(self.root)

    # ✨ EnglishFilter (English Maxxer)
    def open_englishfilter_app(self):
        EnglishFilterApp(self.root)

    # ✨ TokenMaxxer (PyQt5)
    def open_tokenmaxxer_app(self):
        def launch():
            qt_app = QApplication.instance()
            if qt_app is None:
                qt_app = QApplication(sys.argv)
            win = TokenMaxxerV3App()
            win.show()
            qt_app.exec_()
        threading.Thread(target=launch, daemon=True).start()

    # Updated: WordCloud Generator (PyQt5)
    def open_wordcloud_generator_app(self):
        def launch():
            qt_app = QApplication.instance()
            if qt_app is None:
                qt_app = QApplication(sys.argv)
            app = GenerateWordCloudApp(None, self.theme)
            qt_app.exec_()
        threading.Thread(target=launch, daemon=True).start()

    def create_options_ui(self):
        # increased to 14 columns
        options_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        options_frame.grid(row=0, column=0, columnspan=14, pady=20, sticky='ew')

        for i in range(14):
            options_frame.columnconfigure(i, weight=1)

        buttons = [
            ("Music Player", self.open_music_player_app),
            ("DataMaxxer", self.open_filter_app),
            ("Deslop", self.open_deslop_tool),
            ("Dataset Converter", self.open_dataset_converter_app),
            ("Generate Word Cloud", self.open_wordcloud_generator_app),
            ("Binary Classification", self.open_binary_classification_app),
            ("Deduplication", self.open_deduplication_app),
            ("N-gram Analyzer", self.open_ngram_analyzer_app),
            ("GrammarMaxxer", self.open_text_correction_app),
            ("SafetensorMaxxer", self.open_safetensormaxxer_app),
            ("LineMancer", self.open_linemancer_app),
            ("ParquetMaxxer", self.open_parquetmaxxer_app),
            ("EnglishFilter", self.open_englishfilter_app),  # 💖 NEW BUTTON
            ("TokenMaxxer", self.open_tokenmaxxer_app),
        ]

        for index, (text, command) in enumerate(buttons):
            button = tk.Button(
                options_frame,
                text=text,
                command=command,
                bg=self.theme.get('button_bg', 'lightgrey'),
                fg=self.theme.get('button_fg', 'black')
            )
            button.grid(row=0, column=index, pady=10, padx=5, sticky='ew')

    def update_ui_styles(self):
        self.style.configure('TButton',
                             background=self.theme.get('button_bg', 'lightgrey'),
                             foreground=self.theme.get('button_fg', 'black'))
        self.style.configure('TLabel',
                             background=self.theme.get('bg', 'white'),
                             foreground=self.theme.get('fg', 'black'))
        self.style.configure('TEntry',
                             fieldbackground=self.theme.get('entry_bg', 'white'),
                             foreground=self.theme.get('entry_fg', 'black'))
        for widget in self.root.winfo_children():
            if widget.winfo_class() == 'Frame':
                widget.configure(bg=self.theme.get('bg', 'white'))
            elif widget.winfo_class() in ['Label', 'Button']:
                widget.configure(bg=self.theme.get('bg', 'white'),
                                 fg=self.theme.get('fg', 'black'))
            elif widget.winfo_class() == 'Entry':
                widget.configure(bg=self.theme.get('entry_bg', 'white'),
                                 fg=self.theme.get('entry_fg', 'black'))