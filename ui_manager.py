import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pygame
import os
from dataset_converter import DatasetConverter

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Converter")
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.setup_ui()
        self.init_pygame()

    def setup_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(6, weight=1)

        self.create_file_selection_ui()
        self.create_options_ui()
        self.create_preview_ui()
        self.create_status_bar()

        self.style = ttk.Style()
        self.apply_light_mode()

    def create_file_selection_ui(self):
        self.entry_input_file = self.create_labeled_entry("Input File:", 0, self.select_input_file)
        self.entry_output_file = self.create_labeled_entry("Output File:", 1, self.select_output_file)

    def create_labeled_entry(self, label, row, command):
        tk.Label(self.root, text=label).grid(row=row, column=0, padx=10, pady=10, sticky="e")
        entry = tk.Entry(self.root, width=50)
        entry.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        tk.Button(self.root, text="Browse...", command=command).grid(row=row, column=2, padx=10, pady=10)
        return entry

    def create_options_ui(self):
        self.debug_var = tk.BooleanVar()
        tk.Checkbutton(self.root, text="Enable Debugging", variable=self.debug_var).grid(row=2, column=0, columnspan=3, pady=10)

        tk.Button(self.root, text="Convert", command=self.on_convert_button_click).grid(row=3, column=0, columnspan=3, pady=20)

        self.music_button = tk.Button(self.root, text="Play Music", command=self.play_music)
        self.music_button.grid(row=4, column=0, columnspan=3, pady=10)

        tk.Label(self.root, text="Volume:").grid(row=5, column=0, padx=10, pady=10, sticky="e")
        self.volume_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_volume)
        self.volume_slider.set(100)
        self.volume_slider.grid(row=5, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        tk.Checkbutton(self.root, text="Dark Mode", variable=self.dark_mode_var, command=self.toggle_dark_mode).grid(row=6, column=0, columnspan=3, pady=10)

    def create_preview_ui(self):
        tk.Label(self.root, text="Preview Output:").grid(row=7, column=0, padx=10, pady=10, sticky="nw")
        self.preview_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=('Consolas', 10))
        self.preview_text.grid(row=8, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN)
        self.status_bar.grid(row=9, column=0, columnspan=3, sticky='ew')

    def init_pygame(self):
        pygame.init()
        pygame.mixer.init()
        self.mp3_file_path = "bgm.mp3"

    def select_input_file(self):
        self.select_file(self.entry_input_file, [
            ("JSON files", "*.json"),
            ("JSON Lines files", "*.jsonl"),
            ("Parquet files", "*.parquet"),
            ("Plaintext files", "*.txt"),
            ("CSV files", "*.csv"),
            ("SQL files", "*.sql")
        ])

    def select_output_file(self):
        self.select_file(self.entry_output_file, [("JSON Lines files", "*.jsonl")], save=True)

    def select_file(self, entry, filetypes, save=False):
        if save:
            file_path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=filetypes)
        else:
            file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def on_convert_button_click(self):
        input_path = self.entry_input_file.get()
        output_path = self.entry_output_file.get()
        if input_path and output_path:
            self.convert_dataset(input_path, output_path)
        else:
            messagebox.showwarning("Input Error", "Please select both input and output files.")

    def convert_dataset(self, input_path, output_path):
        try:
            data = DatasetConverter.load_data(input_path)
            preview_entries = DatasetConverter.process_data(data, output_path)
            self.update_preview(preview_entries)
            self.status_var.set("Conversion completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_preview(self, preview_entries):
        self.preview_text.delete(1.0, tk.END)
        if preview_entries:
            self.preview_text.insert(tk.END, json.dumps(preview_entries, indent=2))
        else:
            self.preview_text.insert(tk.END, "No conversations found for this dataset.")

    def play_music(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.music_button.config(text="Play Music")
        else:
            if os.path.exists(self.mp3_file_path):
                pygame.mixer.music.load(self.mp3_file_path)
                pygame.mixer.music.play(-1)
                self.music_button.config(text="Pause Music")
            else:
                messagebox.showwarning("Music Error", "MP3 file not found.")

    def set_volume(self, val):
        volume = float(val) / 100
        pygame.mixer.music.set_volume(volume)

    def toggle_dark_mode(self):
        if self.dark_mode_var.get():
            self.apply_dark_mode()
        else:
            self.apply_light_mode()

    def apply_dark_mode(self):
        self.root.tk_setPalette(background='#2e2e2e', foreground='#ffffff')
        self.style.configure('TButton', background='#3e3e3e', foreground='#ffffff', padding=6)
        self.style.configure('TLabel', background='#2e2e2e', foreground='#ffffff')
        self.style.configure('TCheckbutton', background='#2e2e2e', foreground='#ffffff')
        self.style.configure('TEntry', background='#3e3e3e', foreground='#ffffff')
        self.style.configure('TScrolledText', background='#3e3e3e', foreground='#ffffff')
        self.status_bar.config(bg='#2e2e2e', fg='#ffffff')

    def apply_light_mode(self):
        self.root.tk_setPalette(background='#ffffff', foreground='#000000')
        self.style.configure('TButton', background='#f0f0f0', foreground='#000000', padding=6)
        self.style.configure('TLabel', background='#ffffff', foreground='#000000')
        self.style.configure('TCheckbutton', background='#ffffff', foreground='#000000')
        self.style.configure('TEntry', background='#ffffff', foreground='#000000')
        self.style.configure('TScrolledText', background='#ffffff', foreground='#000000')
        self.status_bar.config(bg='#ffffff', fg='#000000')

    def on_closing(self):
        pygame.quit()
        self.root.quit()
        messagebox.showinfo("Info", "Press OK to exit.")