import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from threading import Thread
import os
import time
from deduplication import Deduplication
import logging

class DeduplicationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.deduplication = Deduplication()
        self.use_min_hash = tk.BooleanVar()  # Track dedup method choice
        self.start_time = None  # For speed calc
        self.setup_ui()

    def setup_ui(self):
        self.window = tk.Toplevel(self.root)
        self.window.title("Deduplication App")
        self.window.configure(bg=self.theme.get('bg', 'white'))

        icon_path = 'icon.ico'
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        else:
            logging.error(f"Icon file {icon_path} not found. Using default.")

        main_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # File selection frame
        self.file_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.file_frame.pack(pady=5)

        self.input_file_label = tk.Label(
            self.file_frame, text="Select Input File:",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black')
        )
        self.input_file_label.pack(side=tk.LEFT, padx=5)

        self.input_file_entry = tk.Entry(self.file_frame, width=50)
        self.input_file_entry.pack(side=tk.LEFT, padx=5)

        self.input_file_button = tk.Button(
            self.file_frame, text="Browse", command=self.browse_input_file,
            bg=self.theme.get('button_bg', 'lightgrey'), fg='gold'
        )
        self.input_file_button.pack(side=tk.LEFT, padx=5)

        # Buttons for deduplication method
        button_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        button_frame.pack(pady=10, fill=tk.X)

        self.min_hash_radiobutton = tk.Radiobutton(
            button_frame, text="Min-Hash/Semantic", variable=self.use_min_hash, value=True,
            command=self.update_button_styles, bg='black', fg='gold',
            font=("Arial", 12, "bold"), indicatoron=0, width=15, relief="solid"
        )
        self.min_hash_radiobutton.pack(side=tk.LEFT, padx=10, pady=5)

        self.sha256_radiobutton = tk.Radiobutton(
            button_frame, text="String-Match", variable=self.use_min_hash, value=False,
            command=self.update_button_styles, bg='black', fg='gold',
            font=("Arial", 12, "bold"), indicatoron=0, width=15, relief="solid"
        )
        self.sha256_radiobutton.pack(side=tk.LEFT, padx=10, pady=5)

        self.dedup_button = tk.Button(
            button_frame, text="Remove Duplicates", command=self.start_deduplication,
            bg=self.theme.get('button_bg', 'lightgrey'), fg='gold'
        )
        self.dedup_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Status frame: status text on left, percentage next, speed on right
        status_frame = tk.Frame(self.window, bg=self.theme.get('status_bg', 'lightgrey'))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_bar = tk.Label(
            status_frame, text="Status: Ready",
            bg=self.theme.get('status_bg', 'lightgrey'), fg='black', anchor='w'
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_percent_label = tk.Label(
            status_frame, text="0%", 
            bg=self.theme.get('status_bg', 'lightgrey'), fg='black', font=("Arial", 10, "bold")
        )
        self.progress_percent_label.pack(side=tk.LEFT, padx=10)

        self.speed_label = tk.Label(
            status_frame, text="Speed: 0 it/s",
            bg=self.theme.get('status_bg', 'lightgrey'), fg='black'
        )
        self.speed_label.pack(side=tk.RIGHT, padx=10)

        # Progress bar below status frame, clean and no text overlay
        progress_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress = Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X)

        # Initialize button styles
        self.update_button_styles()

    def update_button_styles(self):
        if self.use_min_hash.get():
            self.min_hash_radiobutton.config(bg='white', fg='black')
            self.sha256_radiobutton.config(bg='#333333', fg='gold')
        else:
            self.min_hash_radiobutton.config(bg='#333333', fg='gold')
            self.sha256_radiobutton.config(bg='white', fg='black')

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)
            self.progress['value'] = 0
            self.progress_percent_label.config(text="0%")
            self.speed_label.config(text="Speed: 0 it/s")
            self.update_status("Ready")

    def start_deduplication(self):
        self.start_time = time.time()
        thread = Thread(target=self.deduplicate_file)
        thread.start()

    def deduplicate_file(self):
        input_file = self.input_file_entry.get()
        if not input_file.endswith('.jsonl'):
            self.update_status("Invalid file type. Please select a .jsonl file.")
            return

        output_dir = "deduplicated"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-deduplicated.jsonl")

        self.update_status("Deduplication started...")

        deduplication_method = "Min-Hash" if self.use_min_hash.get() else "String-Match"
        self.update_status(f"Deduplication method: {deduplication_method} - Processing...")

        self.deduplication.duplicate_count = 0

        if self.use_min_hash.get():
            self.deduplication.perform_min_hash_deduplication(
                input_file, output_file, self.update_status, self.update_progress)
        else:
            self.deduplication.perform_sha256_deduplication(
                input_file, output_file, self.update_status, self.update_progress)

        # Ensure progress bar hits 100% when done
        self.update_progress(total=1, current=1)  # this forces 100% display

    def update_status(self, message):
        self.status_bar.config(text=f"Status: {message}")

    def update_progress(self, current, total):
        if total == 0:
            percent = 0
        elif current >= total:
            percent = 100.0
        else:
            percent = (current / total) * 100

        self.progress['value'] = percent
        self.progress_percent_label.config(text=f"{percent:.2f}%")
        self.window.update_idletasks()

        elapsed = time.time() - self.start_time if self.start_time else 1
        speed = current / elapsed if elapsed > 0 else 0
        self.speed_label.config(text=f"Speed: {speed:.2f} it/s")
