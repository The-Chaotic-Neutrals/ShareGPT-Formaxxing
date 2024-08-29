import tkinter as tk
from tkinter import messagebox, ttk
from theme import Theme
from music_player import MusicPlayer
from dataset_converter_app import DatasetConverterApp
from dataset_filter_app import DatasetFilterApp
from deslop_tool_app import DeslopToolApp
from generate_wordcloud import GenerateWordCloudApp
from ui_elements import UIElements
import os

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK
        self.style = ttk.Style()
        self.music_player = MusicPlayer()
        self.music_button = None  # Initialize as None
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
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def play_music(self):
        if self.music_button is None:
            print("Error: music_button is not initialized.")
            return
        
        try:
            status = self.music_player.play_music()
            self.music_button.config(text=status)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play music: {e}")
            self.music_button.config(text="Play Music")

    def set_volume(self, value):
        self.music_player.set_volume(value)

    def create_options_ui(self):
        options_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        options_frame.grid(row=0, column=0, columnspan=5, pady=20, sticky='ew')

        for i in range(5):
            options_frame.columnconfigure(i, weight=1)

        buttons = [
            ("Play Music", self.play_music),
            ("DataMaxxer", self.open_filter_app),
            ("Deslop Tool", self.open_deslop_tool),
            ("Dataset Converter", self.open_dataset_converter_app),  # Button text updated here
            ("Generate Word Cloud", self.open_wordcloud_generator),
        ]

        for index, (text, command) in enumerate(buttons):
            button = tk.Button(options_frame, text=text, command=command, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
            button.grid(row=0, column=index, pady=10, padx=5, sticky='ew')

        # Initialize music_button here
        self.music_button = tk.Button(options_frame, text="Play Music", command=self.play_music, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
        self.music_button.grid(row=0, column=0, pady=10, padx=5, sticky='ew')

        volume_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        volume_frame.grid(row=1, column=0, columnspan=5, pady=10, sticky='ew')
        volume_frame.columnconfigure(0, weight=1)

        self.volume_slider = UIElements.create_volume_slider(volume_frame, self.theme, self.set_volume)
        self.volume_slider.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

    def open_dataset_converter_app(self):
        DatasetConverterApp(self.root, self.theme)

    def open_filter_app(self):
        DatasetFilterApp(self.root, self.theme)

    def open_deslop_tool(self):
        DeslopToolApp(self.root, self.theme)

    def open_wordcloud_generator(self):
        GenerateWordCloudApp(self.root, self.theme)

    def update_ui_styles(self):
        self.style.configure('TButton', background=self.theme.get('button_bg', 'lightgrey'), foreground=self.theme.get('button_fg', 'black'))
        self.style.configure('TLabel', background=self.theme.get('bg', 'white'), foreground=self.theme.get('fg', 'black'))
        self.style.configure('TEntry', fieldbackground=self.theme.get('entry_bg', 'white'), foreground=self.theme.get('entry_fg', 'black'))
        self.style.configure('TScale', background=self.theme.get('volume_slider_bg', 'lightgrey'), foreground=self.theme.get('volume_slider_fg', 'black'))

        for widget in self.root.winfo_children():
            if widget.winfo_class() == 'Frame':
                widget.configure(bg=self.theme.get('bg', 'white'))
            elif widget.winfo_class() in ['Label', 'Button']:
                widget.configure(bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
            elif widget.winfo_class() == 'Entry':
                widget.configure(bg=self.theme.get('entry_bg', 'white'), fg=self.theme.get('entry_fg', 'black'))
            elif widget.winfo_class() == 'Scale':
                widget.configure(bg=self.theme.get('volume_slider_bg', 'lightgrey'), fg=self.theme.get('volume_slider_fg', 'black'))
