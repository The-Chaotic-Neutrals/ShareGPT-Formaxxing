import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pygame
import os
import json
from pathlib import Path
import polars as pl
from dataset_converter import DatasetConverter

class Theme:
    DARK = {
        'bg': '#2e2e2e',
        'fg': 'gold',
        'text_bg': '#3e3e3e',
        'text_fg': 'gold',
        'entry_bg': '#3e3e3e',
        'entry_fg': 'gold',
        'button_bg': '#3e3e3e',
        'button_fg': 'gold',
        'volume_slider_bg': '#3e3e3e',
        'volume_slider_fg': 'gold',
        'volume_slider_trough': '#2e2e2e'
    }

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Chaotic Neutral's ShareGPT Formaxxing-Tool")
        self.theme = Theme.DARK  # Use dark mode theme by default
        self.setup_ui()
        self.init_pygame()
        self.load_icon()

    def setup_ui(self):
        # Configure the root window
        self.root.configure(bg=self.theme['bg'])
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(6, weight=1)

        self.create_file_selection_ui()
        self.create_options_ui()
        self.create_preview_ui()
        self.create_status_bar()

        self.style = ttk.Style()
        self.update_ui_styles()

    def load_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def create_file_selection_ui(self):
        self.entry_input_file = self.create_labeled_entry("Input File:", 0, self.select_input_file)
        self.entry_output_file = self.create_labeled_entry("Output File:", 1, self.select_output_file)

    def create_labeled_entry(self, label, row, command):
        self.label = tk.Label(self.root, text=label, bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.grid(row=row, column=0, padx=10, pady=10, sticky="e")
        entry = tk.Entry(self.root, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'])
        entry.grid(row=row, column=1, padx=10, pady=10, sticky="ew")
        browse_button = tk.Button(self.root, text="Browse...", command=command, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        browse_button.grid(row=row, column=2, padx=10, pady=10)
        return entry

    def create_options_ui(self):
        self.convert_button = tk.Button(self.root, text="Convert", command=self.on_convert_button_click, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.convert_button.grid(row=2, column=0, columnspan=3, pady=20)

        self.music_button = tk.Button(self.root, text="Play Music", command=self.play_music, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.music_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.volume_label = tk.Label(self.root, text="Volume:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.volume_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")

        # Create a frame for the volume slider to customize its appearance
        self.volume_frame = tk.Frame(self.root, bg=self.theme['bg'])
        self.volume_frame.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        # Adding a custom style to the volume slider
        self.volume_slider = tk.Scale(self.volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.set_volume,
                                      bg=self.theme['volume_slider_bg'], fg=self.theme['volume_slider_fg'],
                                      troughcolor=self.theme['volume_slider_trough'], sliderrelief=tk.FLAT)
        self.volume_slider.set(100)
        self.volume_slider.pack(fill=tk.X, padx=5, pady=5)

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

    def init_pygame(self):
        try:
            pygame.init()
            pygame.mixer.init()
            self.mp3_file_path = "kitchen.mp3"
        except Exception as e:
            messagebox.showerror("Pygame Initialization Error", f"An error occurred while initializing Pygame: {str(e)}")

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
            self.update_status_bar("Loading data from the input file...")
            data = DatasetConverter.load_data(input_path)
            if not data:
                raise ValueError("Loaded data is empty or invalid.")

            self.update_status_bar("Processing data...")
            preview_entries = DatasetConverter.process_data(data, output_path)
            
            if not preview_entries:
                self.update_status_bar("No conversations found for this dataset.")
                self.update_preview("No conversations found for this dataset.")
            else:
                self.update_status_bar("Conversion completed successfully!")
                self.update_preview(preview_entries)

        except FileNotFoundError:
            self.update_status_bar("File not found.")
            self.update_preview("The specified file could not be found.")
        except json.JSONDecodeError:
            self.update_status_bar("Invalid JSON format.")
            self.update_preview("The data could not be decoded as JSON.")
        except ValueError as ve:
            self.update_status_bar(f"Value Error: {str(ve)}")
            self.update_preview("The data contained invalid values.")
        except Exception as e:
            self.update_status_bar(f"Error: {str(e)}")
            self.update_preview("No conversations available due to error.")

    def update_preview(self, preview_entries):
        self.preview_text.delete(1.0, tk.END)
        if isinstance(preview_entries, list):
            self.preview_text.insert(tk.END, json.dumps(preview_entries, indent=2))
        else:
            self.preview_text.insert(tk.END, preview_entries)
        self.root.update_idletasks()  # Force the UI to update

    def play_music(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                self.music_button.config(text="Play Music")
            else:
                if os.path.exists(self.mp3_file_path):
                    pygame.mixer.music.load(self.mp3_file_path)
                    pygame.mixer.music.play()
                    self.music_button.config(text="Pause Music")
                else:
                    messagebox.showerror("File Not Found", "Music file not found.")
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def set_volume(self, value):
        volume = int(value) / 100
        pygame.mixer.music.set_volume(volume)

    def update_ui_styles(self):
        self.style.configure('TLabel', background=self.theme['bg'], foreground=self.theme['fg'])
        self.style.configure('TButton', background=self.theme['button_bg'], foreground=self.theme['button_fg'])
        self.style.configure('TEntry', background=self.theme['entry_bg'], foreground=self.theme['entry_fg'])
        self.style.configure('TScrolledText', background=self.theme['text_bg'], foreground=self.theme['text_fg'])

        # Update widget styles
        for widget in [self.label, self.convert_button, self.music_button, self.volume_label, self.preview_label, self.filter_button]:
            widget.config(bg=self.theme['bg'], fg=self.theme['fg'], activebackground=self.theme['button_bg'], activeforeground=self.theme['button_fg'])
        
        self.entry_input_file.config(bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.entry_output_file.config(bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.preview_text.config(bg=self.theme['text_bg'], fg=self.theme['text_fg'], insertbackground=self.theme['text_fg'])

        # Update volume slider style
        self.volume_slider.config(bg=self.theme['volume_slider_bg'], fg=self.theme['volume_slider_fg'],
                                  troughcolor=self.theme['volume_slider_trough'], sliderrelief=tk.FLAT)

    def update_status_bar(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()  # Force the UI to update

    def open_filter_app(self):
        filter_root = tk.Toplevel(self.root)
        filter_app = DatasetFilterApp(filter_root, self.theme)

class DatasetFilterApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.root.title("DataMaxxer")
        
        # Set the icon for the DataMaxxer window
        self.set_icon()

        # Configure root window
        self.root.configure(bg=self.theme['bg'])

        # Create and place widgets
        self.label = tk.Label(root, text="Dataset File:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.pack(pady=10)

        self.dataset_entry = tk.Entry(root, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.dataset_entry.pack(pady=5)

        self.browse_button = tk.Button(root, text="Browse...", command=self.select_file, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.browse_button.pack(pady=5)

        self.process_button = tk.Button(root, text="Process Dataset", command=self.process_dataset, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.process_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.result_label.pack(pady=10)

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Lines files", "*.jsonl")]
        )
        if file_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, file_path)

    def process_dataset(self):
        file_path = self.dataset_entry.get().strip()
        if not file_path:
            messagebox.showerror("Input Error", "Please select a dataset file.")
            return

        try:
            # Attempt to read the file and check for valid JSON Lines
            data = self.read_json_lines(file_path)
            
            # Define a function to check if a conversation contains all required roles
            def has_required_roles(conversation):
                roles = set(msg['from'] for msg in conversation)
                return 'system' in roles and 'human' in roles and 'gpt' in roles

            # Convert the data to a Polars DataFrame
            df = pl.DataFrame(data)

            # Create a boolean column based on the presence of required roles
            df = df.with_columns(
                pl.col('conversations').map_elements(has_required_roles, return_dtype=pl.Boolean).alias('has_required_roles')
            )

            # Filter the data based on the new boolean column
            filtered_data = df.filter(pl.col('has_required_roles'))

            # Create the Filtered directory if it doesn't exist
            filtered_dir = Path(__file__).parent.absolute() / "Filtered"
            filtered_dir.mkdir(exist_ok=True)

            # Save the filtered data as JSONL
            output_file = filtered_dir / f"{Path(file_path).stem}_filtered.jsonl"

            with output_file.open('w') as f:
                for row in filtered_data.drop('has_required_roles').to_dicts():
                    json.dump(row, f)
                    f.write('\n')

            self.result_label.config(text=f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(filtered_data)}")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def read_json_lines(self, file_path):
        """Reads a JSON Lines file and returns a list of dictionaries."""
        try:
            with open(file_path, 'r') as file:
                data = [json.loads(line) for line in file]
            return data
        except json.JSONDecodeError as e:
            messagebox.showerror("File Error", f"Error decoding JSON: {str(e)}")
            return []
        except Exception as e:
            messagebox.showerror("File Error", f"Error reading file: {str(e)}")
            return []