import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from theme import Theme
from music_player import MusicPlayer
from dataset_converter import DatasetConverter
from dataset_filter_app import DatasetFilterApp
from ui_elements import UIElements
import json
import os

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK
        self.style = ttk.Style()
        self.music_player = MusicPlayer()  # Initialize MusicPlayer
        self.setup_ui()

    def setup_ui(self):
        # Configure the root window
        self.root.configure(bg=self.theme['bg'])
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(6, weight=1)
        
        # Set the icon for the main window
        self.set_icon()

        self.create_file_selection_ui()
        self.create_options_ui()
        self.create_preview_ui()
        self.create_status_bar()
        self.update_ui_styles()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def play_music(self):
        status = self.music_player.play_music()
        self.music_button.config(text=status)

    def set_volume(self, value):
        self.music_player.set_volume(value)

    def create_file_selection_ui(self):
        self.entry_input_file = UIElements.create_labeled_entry(self.root, "Input File:", 0, self.select_input_file, self.theme)
        self.entry_output_file = UIElements.create_labeled_entry(self.root, "Output File:", 1, self.select_output_file, self.theme)

    def create_options_ui(self):
        self.convert_button = tk.Button(self.root, text="Convert", command=self.on_convert_button_click, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.convert_button.grid(row=2, column=0, columnspan=3, pady=20)

        self.music_button = tk.Button(self.root, text="Play Music", command=self.play_music, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.music_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.volume_label = tk.Label(self.root, text="Volume:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.volume_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")

        self.volume_slider = UIElements.create_volume_slider(self.root, self.theme, self.set_volume)
        self.volume_slider.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        self.filter_button = tk.Button(self.root, text="DataMaxxer", command=self.open_filter_app, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.filter_button.grid(row=9, column=0, columnspan=3, pady=20)

    def create_preview_ui(self):
        self.preview_label = tk.Label(self.root, text="Preview Output:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.preview_label.grid(row=6, column=0, padx=10, pady=10, sticky="nw")
        self.preview_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=('Consolas', 10), bg=self.theme['text_bg'], fg=self.theme['text_fg'])
        self.preview_text.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN, bg=self.theme['bg'], fg=self.theme['fg'])
        self.status_bar.grid(row=8, column=0, columnspan=3, sticky='ew')

    def select_input_file(self):
        self.select_file(self.entry_input_file)

    def select_output_file(self):
        self.select_file(self.entry_output_file, save=True)

    def select_file(self, entry, save=False):
        filetypes = [
            ("All Supported Files", "*.json;*.jsonl;*.parquet;*.txt;*.csv;*.sql"),
            ("JSON files", "*.json"),
            ("JSON Lines files", "*.jsonl"),
            ("Parquet files", "*.parquet"),
            ("Plaintext files", "*.txt"),
            ("CSV files", "*.csv"),
            ("SQL files", "*.sql")
        ]
        try:
            if save:
                file_path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=filetypes)
            else:
                file_path = filedialog.askopenfilename(filetypes=filetypes)
            if file_path:
                entry.delete(0, tk.END)
                entry.insert(0, file_path)
        except Exception as e:
            messagebox.showerror("File Selection Error", f"An error occurred: {str(e)}")

    def on_convert_button_click(self):
        input_path = self.entry_input_file.get()
        output_path = self.entry_output_file.get()
        if input_path and output_path:
            self.update_status_bar("Starting conversion...")
            self.convert_dataset(input_path, output_path)
        else:
            self.update_status_bar("Input Error: Please select both input and output files.")

    def convert_dataset(self, input_path, output_path):
        try:
            data = DatasetConverter.load_data(input_path)
            processed_data = DatasetConverter.process_data(data, output_path)
            if processed_data:
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, json.dumps(processed_data, indent=2))
                self.update_status_bar(f"Conversion completed: {len(processed_data)} records processed.")
            else:
                self.update_status_bar("Conversion Error: No data processed.")
        except Exception as e:
            self.update_status_bar(f"Conversion Error: {str(e)}")

    def update_status_bar(self, message):
        self.status_var.set(message)

    def open_filter_app(self):
        # Pass the necessary arguments to DatasetFilterApp
        DatasetFilterApp(self.root, self.theme)

    def update_ui_styles(self):
        # Update ttk styles first
        self.style.configure('TButton', background=self.theme['button_bg'], foreground=self.theme['button_fg'])
        self.style.configure('TLabel', background=self.theme['bg'], foreground=self.theme['fg'])
        self.style.configure('TEntry', fieldbackground=self.theme['entry_bg'], foreground=self.theme['entry_fg'])
        self.style.configure('TScale', background=self.theme['volume_slider_bg'], foreground=self.theme['volume_slider_fg'])

        # Update standard Tk widgets
        for widget in self.root.winfo_children():
            widget_type = widget.winfo_class()
            
            if widget_type == 'Label' or widget_type == 'Button':
                widget.configure(bg=self.theme['bg'], fg=self.theme['fg'])
            elif widget_type == 'Entry':
                widget.configure(bg=self.theme['entry_bg'], fg=self.theme['entry_fg'])
            elif widget_type == 'Scale':
                widget.configure(bg=self.theme['volume_slider_bg'], fg=self.theme['volume_slider_fg'])
            elif widget_type == 'Frame':
                widget.configure(bg=self.theme['bg'])
