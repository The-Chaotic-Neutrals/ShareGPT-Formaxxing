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
import os

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK
        self.style = ttk.Style()
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface with initial configurations."""
        self.root.configure(bg=self.theme.get('bg', 'white'))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.set_icon()
        self.create_options_ui()
        self.update_ui_styles()

    def set_icon(self):
        """Set the icon for the window."""
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def open_music_player_app(self):
        """Open the Music Player application."""
        MusicPlayerApp(self.root, self.theme)

    def open_dataset_converter_app(self):
        """Open the Dataset Converter application."""
        DatasetConverterApp(self.root, self.theme)

    def open_filter_app(self):
        """Open the Dataset Filter application.""" 
        DatasetFilterApp(self.root, self.theme)

    def open_deslop_tool(self):
        """Open the Deslop Tool application."""
        DeslopToolApp(self.root, self.theme)

    def open_wordcloud_generator_app(self):
        """Open the Word Cloud Generator application."""        
        GenerateWordCloudApp(self.root, self.theme)
       
    def open_binary_classification_app(self):
        """Open the Binary Classification Tool."""
        BinaryClassificationApp(self.root, self.theme)
    
    def open_deduplication_app(self):
        """Open the Deduplication Tool."""
        DeduplicationApp(self.root, self.theme)
    
    def open_ngram_analyzer_app(self):
        """Open the N-gram Analyzer application in a separate window."""        
        ngram_window = tk.Toplevel(self.root)
        ngram_window.title("N-gram Analyzer App")
        NgramAnalyzerApp(ngram_window, self.theme)

    def open_text_correction_app(self):
        """Open the GrammarMaxxer application."""
        GrammarMaxxerApp(self.root, self.theme)

    def open_safetensormaxxer_app(self):
        """Open the SafetensorMaxxer application.""" 
        safetensor_window = tk.Toplevel(self.root)
        safetensor_window.title("SafetensorMaxxer App")
        SafetensorMaxxerApp(safetensor_window, self.theme)

    def create_options_ui(self):
        """Create and place the UI elements for options."""
        options_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        options_frame.grid(row=0, column=0, columnspan=10, pady=20, sticky='ew')

        for i in range(10):  # Adjusted to 10 columns
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
            ("SafetensorMaxxer", self.open_safetensormaxxer_app)
        ]

        for index, (text, command) in enumerate(buttons):
            button = tk.Button(options_frame, text=text, command=command, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
            button.grid(row=0, column=index, pady=10, padx=5, sticky='ew')

    def update_ui_styles(self):
        """Update the UI styles based on the selected theme."""
        self.style.configure('TButton', background=self.theme.get('button_bg', 'lightgrey'), foreground=self.theme.get('button_fg', 'black'))
        self.style.configure('TLabel', background=self.theme.get('bg', 'white'), foreground=self.theme.get('fg', 'black'))
        self.style.configure('TEntry', fieldbackground=self.theme.get('entry_bg', 'white'), foreground=self.theme.get('entry_fg', 'black'))

        for widget in self.root.winfo_children():
            if widget.winfo_class() == 'Frame':
                widget.configure(bg=self.theme.get('bg', 'white'))
            elif widget.winfo_class() in ['Label', 'Button']:
                widget.configure(bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
            elif widget.winfo_class() == 'Entry':
                widget.configure(bg=self.theme.get('entry_bg', 'white'), fg=self.theme.get('entry_fg', 'black'))
