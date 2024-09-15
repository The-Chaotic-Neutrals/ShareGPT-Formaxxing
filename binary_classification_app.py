import tkinter as tk
from tkinter import filedialog
from threading import Thread
from binary_classification import initialize_models, update_device_preference, filter_conversations

class BinaryClassificationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.device = 'cuda'  # Default to GPU
        self.total_positive_count = 0
        self.total_negative_count = 0

        # Initialize the Spacy and Transformer models
        initialize_models()

        # Set up the user interface
        self.setup_ui()

        # Configure logging
        self.configure_logging()

        # Update the device preference after initializing models
        self.update_device_preference()

    def configure_logging(self):
        """Configure logging to reduce output to the command line."""
        import logging
        logging.basicConfig(level=logging.WARNING)  # Only log warnings and errors

    def setup_ui(self):
        """Set up the user interface for the Binary Classification app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Binary Classification App")
        self.window.configure(bg=self.theme.get('bg', 'white'))
        self.window.iconbitmap('icon.ico')  # Ensure 'icon.ico' is in the same directory as your script

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

        self.filter_button = tk.Button(main_frame, text="Filter Conversations", command=self.start_filtering, bg=self.theme.get('button_bg', 'lightgrey'))
        self.filter_button.pack(pady=10)

        # Threshold input
        self.threshold_label = tk.Label(main_frame, text="Threshold (0.0 - 1.0):", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.threshold_label.pack(pady=5)

        self.threshold_entry = tk.Entry(main_frame, width=10)
        self.threshold_entry.insert(0, '0.9')
        self.threshold_entry.pack(pady=5)

        # Batch size input
        self.batch_size_label = tk.Label(main_frame, text="Batch Size:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.batch_size_label.pack(pady=5)

        self.batch_size_entry = tk.Entry(main_frame, width=10)
        self.batch_size_entry.insert(0, '16')  # Default batch size
        self.batch_size_entry.pack(pady=5)

        # Status bar
        self.status_bar = tk.Label(main_frame, text="Status: Ready", bg='lightgrey', fg='black', anchor='w')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Counts display
        self.count_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.count_frame.pack(pady=5)

        self.positive_count_label = tk.Label(self.count_frame, text="Positive Count: 0", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.positive_count_label.pack(side=tk.LEFT, padx=5)

        self.negative_count_label = tk.Label(self.count_frame, text="Negative Count: 0", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.negative_count_label.pack(side=tk.LEFT, padx=5)

        # Device preference checkbox
        self.gpu_var = tk.BooleanVar()
        self.gpu_checkbox = tk.Checkbutton(main_frame, text="Prefer GPU", variable=self.gpu_var, command=self.update_device_preference, bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.gpu_checkbox.pack(pady=10)

    def update_device_preference(self):
        """Update the device preference based on user selection."""
        update_device_preference(self.gpu_var, self.status_bar)

    def browse_input_file(self):
        """Open a file dialog to select the input JSONL file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def start_filtering(self):
        """Start filtering in a separate thread to keep the UI responsive."""
        thread = Thread(target=self.filter_conversations)
        thread.start()

    def filter_conversations(self):
        """Filter conversations based on the specified criteria."""
        filter_conversations(
            input_file_entry=self.input_file_entry,
            threshold_entry=self.threshold_entry,
            batch_size_entry=self.batch_size_entry,
            status_bar=self.status_bar,
            positive_count_label=self.positive_count_label,
            negative_count_label=self.negative_count_label
        )

    def update_status(self, message):
        """Update the status bar with a message."""
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def update_counts(self, positive_count, negative_count):
        """Update the positive and negative counts in the UI."""
        if self.root:
            self.root.after(0, lambda: self.positive_count_label.config(text=f"Positive Count: {positive_count}"))
            self.root.after(0, lambda: self.negative_count_label.config(text=f"Negative Count: {negative_count}"))

# Sample usage
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    app = BinaryClassificationApp(root, theme={
        'bg': 'white',
        'fg': 'black',
        'button_bg': 'lightgrey'
    })
    root.mainloop()
