import tkinter as tk
from tkinter import filedialog
import jsonlines
import os
import logging
from threading import Thread

class DeduplicationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.duplicate_count = 0

        # Set to store unique conversation identifiers (e.g., hashes)
        self.unique_conversations = set()

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
        """Remove duplicates from the input file and write to an output file."""
        input_file = self.input_file_entry.get()
        if not input_file.endswith('.jsonl'):
            self.update_status("Invalid file type. Please select a .jsonl file.")
            return

        # Create output directory if it doesn't exist
        output_dir = "deduplicated"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-deduplicated.jsonl")

        self.update_status("Deduplication started...")
        try:
            self.duplicate_count = 0
            self.unique_conversations.clear()  # Clear previous unique identifiers

            with jsonlines.open(input_file) as reader:
                with jsonlines.open(output_file, mode='w') as writer:
                    for conversation in reader:
                        # Generate a unique identifier for deduplication
                        conversation_id = self.generate_conversation_id(conversation)
                        if conversation_id in self.unique_conversations:
                            self.duplicate_count += 1
                            logging.debug(f"Duplicate detected. Total duplicates: {self.duplicate_count}")
                            self.update_duplicate_count(self.duplicate_count)
                            continue
                        self.unique_conversations.add(conversation_id)
                        writer.write(conversation)

            self.update_status(f"Deduplication complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
            self.update_status(f"Error processing file: {e}")

    def generate_conversation_id(self, conversation):
        """Generate a unique identifier for a conversation based on its content."""
        conversation_text = ''.join(turn.get('value', '') for turn in conversation.get('conversations', []))
        return hash(conversation_text)

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
