import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import jsonlines
import os
import logging
from threading import Thread
import hashlib
from datasketch import MinHash, MinHashLSH

class DeduplicationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.duplicate_count = 0
        self.unique_conversations = set()  # For SHA-256 deduplication
        self.threshold = 0.8  # Similarity threshold for MinHash deduplication
        self.use_min_hash = tk.BooleanVar()  # Variable to track deduplication method choice

        # Set up the user interface
        self.setup_ui()
        # Configure logging
        self.configure_logging()

    def configure_logging(self):
        """Configure logging to a file and console."""
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("app.log"),
                                logging.StreamHandler()
                            ])

    def setup_ui(self):
        """Set up the user interface for the Deduplication app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Deduplication App")
        self.window.configure(bg=self.theme.get('bg', 'white'))
        self.window.iconbitmap('icon.ico')  # Ensure 'icon.ico' is in the same directory

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

        # Toggle for deduplication method
        self.method_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.method_frame.pack(pady=5)
        
        self.dedup_method_label = tk.Label(self.method_frame, text="Choose Deduplication Method:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.dedup_method_label.pack(side=tk.LEFT, padx=5)

        self.method_toggle = tk.Checkbutton(self.method_frame, text="Use Min-Hash Deduplication", variable=self.use_min_hash, bg=self.theme.get('bg', 'white'))
        self.method_toggle.pack(side=tk.LEFT, padx=5)

        self.dedup_button = tk.Button(main_frame, text="Remove Duplicates", command=self.start_deduplication, bg=self.theme.get('button_bg', 'lightgrey'))
        self.dedup_button.pack(pady=10)

        # Status bar
        self.status_bar = tk.Label(main_frame, text="Status: Ready", bg='lightgrey', fg='black', anchor='w')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Duplicate count display
        self.count_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.count_frame.pack(pady=5)

        self.duplicate_count_label = tk.Label(self.count_frame, text="Duplicate Count: 0", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.duplicate_count_label.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

    def update_status(self, message):
        """Update the status bar with a message."""
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def update_duplicate_count(self, duplicate_count):
        """Update the duplicate count in the UI."""
        if self.root:
            logging.info(f"Updating UI with duplicate count: {duplicate_count}")
            self.root.after(0, lambda: self.duplicate_count_label.config(text=f"Duplicate Count: {duplicate_count}"))

    def browse_input_file(self):
        """Open a file dialog to select the input JSONL file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def start_deduplication(self):
        """Start deduplication in a separate thread to keep the UI responsive."""
        thread = Thread(target=self.deduplicate_file)
        thread.start()

    def deduplicate_file(self):
        """Deduplicate using the selected method (SHA-256 or Min-Hash)."""
        input_file = self.input_file_entry.get()
        if not input_file.endswith('.jsonl'):
            self.update_status("Invalid file type. Please select a .jsonl file.")
            return

        # Prepare output file
        output_dir = "deduplicated"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-deduplicated.jsonl")

        self.update_status("Deduplication started...")
        self.duplicate_count = 0
        self.unique_conversations.clear()  # Clear previous unique identifiers

        if self.use_min_hash.get():
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=128)
            self.perform_min_hash_deduplication(input_file, output_file)
        else:
            self.perform_sha256_deduplication(input_file, output_file)

    def perform_sha256_deduplication(self, input_file, output_file):
        """Exact deduplication using SHA-256 hashing."""
        try:
            with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
                total_lines = sum(1 for _ in open(input_file, 'r'))
                for i, conversation in enumerate(reader):
                    # Update progress
                    self.progress['value'] = (i / total_lines) * 100
                    self.root.update_idletasks()

                    conversation_id = self.generate_sha256_hash(conversation)
                    if conversation_id in self.unique_conversations:
                        self.duplicate_count += 1
                        self.update_duplicate_count(self.duplicate_count)
                        continue

                    self.unique_conversations.add(conversation_id)
                    writer.write(conversation)

            self.update_status(f"Deduplication complete. Output file: {output_file}")
            messagebox.showinfo("Deduplication Complete", f"Output saved to {output_file}")
        except Exception as e:
            logging.error(f"Error during SHA-256 deduplication: {e}", exc_info=True)

    def perform_min_hash_deduplication(self, input_file, output_file):
        """Near-duplicate detection using MinHash deduplication."""
        try:
            with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
                total_lines = sum(1 for _ in open(input_file, 'r'))
                for i, conversation in enumerate(reader):
                    # Update progress
                    self.progress['value'] = (i / total_lines) * 100
                    self.root.update_idletasks()

                    conversation_text = ''.join(turn['value'] for turn in conversation.get('conversations', []) if turn.get('value') is not None)
                    shingles = self.shingle_text(conversation_text)
                    m = self.generate_min_hash(shingles)

                    # Check if similar signature already exists
                    if any(self.lsh.query(m)):
                        self.duplicate_count += 1
                        self.update_duplicate_count(self.duplicate_count)
                        continue

                    # Add to LSH and write unique conversation to output
                    self.lsh.insert(conversation_text, m)
                    writer.write(conversation)

            self.update_status(f"Deduplication complete. Output file: {output_file}")
            messagebox.showinfo("Deduplication Complete", f"Output saved to {output_file}")
        except Exception as e:
            logging.error(f"Error during MinHash deduplication: {e}", exc_info=True)

    def shingle_text(self, text, k=5):
        """Generate k-shingles for text."""
        return set(text[i:i+k] for i in range(len(text) - k + 1))

    def generate_sha256_hash(self, conversation):
        """Generate a SHA-256 hash for a conversation."""
        conversation_text = ''.join(turn['value'] for turn in conversation.get('conversations', []) if turn.get('value') is not None)
        return hashlib.sha256(conversation_text.encode('utf-8')).hexdigest()

    def generate_min_hash(self, shingles):
        """Generate a MinHash signature from shingles."""
        m = MinHash(num_perm=128)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        return m

# Sample usage
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    app = DeduplicationApp(root, theme={
        'bg': 'white',
        'fg': 'black',
        'button_bg': 'lightgrey'
    })
    root.mainloop()
