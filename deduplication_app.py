import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from threading import Thread
import os
from deduplication import Deduplication

class DeduplicationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.deduplication = Deduplication()
        self.use_min_hash = tk.BooleanVar()  # Variable to track deduplication method choice

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for the Deduplication app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Deduplication App")
        self.window.configure(bg=self.theme.get('bg', 'white'))

        # Set the icon for the window (ensure the icon exists in the same folder)
        icon_path = 'icon.ico'  # Ensure this file exists in the same directory
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        else:
            logging.error(f"Icon file {icon_path} not found. Using default.")

        # Main frame
        main_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # File handling UI
        self.file_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.file_frame.pack(pady=5)

        self.input_file_label = tk.Label(self.file_frame, text="Select Input File:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.input_file_label.pack(side=tk.LEFT, padx=5)

        self.input_file_entry = tk.Entry(self.file_frame, width=50)
        self.input_file_entry.pack(side=tk.LEFT, padx=5)

        self.input_file_button = tk.Button(self.file_frame, text="Browse", command=self.browse_input_file, bg=self.theme.get('button_bg', 'lightgrey'))
        self.input_file_button.pack(side=tk.LEFT, padx=5)

        # Deduplication method toggle
        self.method_toggle = tk.Checkbutton(self.window, 
                                            text="Use Min-Hash Deduplication [Otherwise defaults to string matching]", 
                                            variable=self.use_min_hash, 
                                            bg=self.theme.get('bg', 'white'),
                                            fg='gold')  # Set text color to gold
        self.method_toggle.pack(pady=5)

        # Deduplication button
        self.dedup_button = tk.Button(self.window, text="Remove Duplicates", command=self.start_deduplication, bg=self.theme.get('button_bg', 'lightgrey'))
        self.dedup_button.pack(pady=10)

        # Status bar
        self.status_bar = tk.Label(self.window, text="Status: Ready", bg=self.theme.get('status_bg', 'lightgrey'), fg='black', anchor='w')  # Explicitly set the text color to black
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress bar
        self.progress = Progressbar(self.window, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def start_deduplication(self):
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
        self.deduplication.duplicate_count = 0

        if self.use_min_hash.get():
            self.deduplication.perform_min_hash_deduplication(
                input_file, output_file, self.update_status, self.update_progress)
        else:
            self.deduplication.perform_sha256_deduplication(
                input_file, output_file, self.update_status, self.update_progress)

    def update_status(self, message):
        """Update the status message on the UI."""
        self.status_bar.config(text=f"Status: {message}")

    def update_progress(self, current, total):
        """Update the progress bar."""
        self.progress['value'] = (current / total) * 100
        self.window.update_idletasks()
