import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import spacy
import jsonlines
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from threading import Thread

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

class BinaryClassificationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.device = 'cuda'  # Default to GPU
        self.total_positive_count = 0
        self.total_negative_count = 0

        # Initialize the Spacy and Transformer models
        self.initialize_models()

        # Set up the user interface
        self.setup_ui()

        # Configure logging
        self.configure_logging()

        # Update the device preference after initializing models
        self.update_device_preference()

    def configure_logging(self):
        """Configure logging to reduce output to the command line."""
        logging.basicConfig(level=logging.WARNING)  # Only log warnings and errors

    def initialize_models(self):
        """Initialize Spacy and Transformer models."""
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.nlp.add_pipe("sentencizer")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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
        if hasattr(self, 'model'):
            if self.gpu_var.get() and torch.cuda.is_available():
                self.device = 'cuda'
                self.update_status("Using GPU")
            else:
                self.device = 'cpu'
                self.update_status("Using CPU")
            self.model.to(self.device)

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
        input_file = self.input_file_entry.get()
        if not input_file.endswith('.jsonl'):
            self.update_status("Invalid file type. Please select a .jsonl file.")
            return

        try:
            threshold = float(self.threshold_entry.get())
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0.")
        except ValueError as e:
            self.update_status(f"Error: {e}")
            return

        try:
            batch_size = int(self.batch_size_entry.get())
            if batch_size <= 0:
                raise ValueError("Batch size must be a positive integer.")
        except ValueError as e:
            self.update_status(f"Error: {e}")
            return

        # Create output directory if it doesn't exist
        output_dir = "classified"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

        # Perform filtering in a separate thread to keep the UI responsive
        self.update_status("Filtering started...")
        try:
            self.total_positive_count = 0
            self.total_negative_count = 0
            self.run_filter(input_file, output_file, threshold, batch_size)
            self.update_status(f"Filtering complete. Output file: {output_file}")
        except Exception as e:
            self.update_status(f"Error: {e}")

    def run_filter(self, input_file, output_file, threshold, batch_size):
        """Run the filtering process on the input file."""
        try:
            with jsonlines.open(input_file) as reader:
                total_lines = sum(1 for _ in jsonlines.open(input_file))
                self.update_status(f"Total lines: {total_lines}")

                # Use a thread pool for parallel processing of conversations
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []

                    with open(output_file, mode='w') as writer:
                        batch = []
                        for i, conversation in enumerate(reader):
                            batch.append(conversation)
                            if len(batch) >= batch_size:
                                futures.append(executor.submit(self.process_batch, batch, threshold, writer))
                                batch = []

                        # Process remaining batch
                        if batch:
                            futures.append(executor.submit(self.process_batch, batch, threshold, writer))

                        for future in futures:
                            future.result()  # Ensure all futures are completed

            self.update_status(f"Filtering complete. Output file: {output_file}")

        except Exception as e:
            self.update_status(f"Error processing JSONL file: {e}")

    def process_batch(self, batch, threshold, writer):
        """Process a batch of conversations and write the results to the output file."""
        try:
            for conversation in batch:
                result = self.process_conversation(conversation, threshold)
                if result:
                    # Convert result to JSON string and write to file
                    writer.write(json.dumps(result) + '\n')
        except Exception as e:
            self.update_status(f"Error processing batch: {e}")

    def process_conversation(self, conversation, threshold):
        """Process a single conversation and classify its sentences."""
        keep_conversation = True

        sentences = []
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                doc = self.nlp(turn.get('value', ''))
                sentences.extend(self.extract_sentences(doc))

        # Classify sentences and check against threshold
        classifications = self.predict(sentences)

        # Update counts
        positive_count = sum(1 for classification in classifications if classification['positive'] > threshold)
        negative_count = sum(1 for classification in classifications if classification['negative'] > threshold)

        self.total_positive_count += positive_count
        self.total_negative_count += negative_count
        self.update_counts(self.total_positive_count, self.total_negative_count)

        if positive_count > 0:
            keep_conversation = False

        return conversation if keep_conversation else None

    def extract_sentences(self, doc):
        """Extract sentences from a Spacy Doc object."""
        return [sent.text.strip() for sent in doc.sents]

    def update_status(self, message):
        """Update the status bar with a message."""
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def update_counts(self, positive_count, negative_count):
        """Update the positive and negative counts in the UI."""
        if self.root:
            self.root.after(0, lambda: self.positive_count_label.config(text=f"Positive Count: {positive_count}"))
            self.root.after(0, lambda: self.negative_count_label.config(text=f"Negative Count: {negative_count}"))

    def predict(self, texts):
        """Predict the classifications for a batch of texts."""
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()
        results = [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]
        return results

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
