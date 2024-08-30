import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import spacy
import jsonlines
import os
import requests
from tqdm import tqdm
from threading import Thread
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import aiofiles

class BinaryClassificationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        
        # Require GPU; raise an exception if not available
        spacy.require_gpu()
        
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Load Spacy model without parser and NER
        self.nlp.add_pipe("sentencizer")  # Add sentencizer component to detect sentence boundaries
        
        self.setup_ui()
        self.configure_logging()

    def configure_logging(self):
        """Configure logging to reduce output to the command line."""
        logging.basicConfig(level=logging.WARNING)  # Only log warnings and errors

    def setup_ui(self):
        """Set up the UI elements for the Binary Classification Tool."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Binary Classification")
        self.window.configure(bg=self.theme.get('bg', 'white'))

        # Set the window icon
        icon_path = "icon.ico"
        if os.path.isfile(icon_path):
            self.window.iconbitmap(icon_path)
        else:
            print(f"Icon file not found: {icon_path}")

        self.input_file_label = tk.Label(self.window, text="Select Input File:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.input_file_label.pack(pady=5)
        
        self.input_file_entry = tk.Entry(self.window, width=50)
        self.input_file_entry.pack(pady=5)
        
        self.input_file_button = tk.Button(self.window, text="Browse", command=self.browse_input_file, bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.input_file_button.pack(pady=5)

        self.filter_button = tk.Button(self.window, text="Filter Conversations", command=self.start_filtering, bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.filter_button.pack(pady=10)

        # Threshold input
        self.threshold_label = tk.Label(self.window, text="Threshold (0.0 - 1.0):", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.threshold_label.pack(pady=5)

        self.threshold_entry = tk.Entry(self.window, width=10)
        self.threshold_entry.insert(0, '0.9')
        self.threshold_entry.pack(pady=5)

        # URL input
        self.url_label = tk.Label(self.window, text="Classification URL:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.url_label.pack(pady=5)

        self.url_entry = tk.Entry(self.window, width=50)
        self.url_entry.insert(0, "http://localhost:8120/predict")
        self.url_entry.pack(pady=5)

        # Progress bar
        self.progress_label = tk.Label(self.window, text="Progress:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.progress_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self.window, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        # Status bar
        self.status_bar = tk.Label(self.window, text="Status: Ready", bg='lightgrey', fg='black', anchor='w')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

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

    def test_connection(self):
        """Test the connection to the classification URL."""
        url = self.url_entry.get().strip()
        if not url:
            self.update_status("Please enter a classification URL.")
            return

        # Use a thread to handle the network request
        thread = Thread(target=self._test_connection_thread, args=(url,))
        thread.start()

    def _test_connection_thread(self, url):
        """Thread worker for testing the connection."""
        try:
            # Send a test request to the server
            test_sentence = "This is a test sentence."
            response = requests.post(url, json={"text": test_sentence})
            
            if response.status_code == 200:
                result = response.json()
                self.update_status(f"Connection successful! Response: {result}")
            else:
                self.update_status(f"Connection failed with status code: {response.status_code}")
        except requests.RequestException as e:
            self.update_status(f"Failed to connect: {e}")

    async def classify_sentences_async(self, sentences, url):
        """Classify multiple sentences asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            # Batch sentences in groups of 10 to reduce the number of HTTP requests
            for i in range(0, len(sentences), 22):
                batch = sentences[i:i+22]
                tasks.append(self.classify_sentence_async(session, batch, url))
            results = await asyncio.gather(*tasks)
            return [result for batch_results in results for result in batch_results]  # Flatten the results list

    async def classify_sentence_async(self, session, sentences_batch, url):
        """Classify a batch of sentences asynchronously."""
        try:
            async with session.post(url, json={"texts": sentences_batch}) as response:  # Assume the server can handle batch requests
                response.raise_for_status()
                result = await response.json()
                logging.debug(f"Server response: {result}")  # Debug logging can be removed if needed
                return result.get("results", [])
        except aiohttp.ClientError as e:
            logging.error(f"Failed to classify batch: {sentences_batch}, Error: {e}")
            self.update_status(f"Failed to classify sentences.")
            return [None] * len(sentences_batch)

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

        # Create output directory if it doesn't exist
        output_dir = "classified"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

        # Perform filtering in a separate thread to keep the UI responsive
        self.update_status("Filtering started...")
        try:
            asyncio.run(self.run_filter(input_file, output_file, threshold))
            self.update_status(f"Filtering complete. Output file: {output_file}")
            # Asynchronously save the file
            asyncio.run(self.async_save_file(output_file))
        except Exception as e:
            self.update_status(f"Error: {e}")

    async def run_filter(self, input_file, output_file, threshold):
        """Run the filtering process on the input file."""
        url = self.url_entry.get().strip()

        try:
            with jsonlines.open(input_file) as reader:
                total_lines = sum(1 for _ in jsonlines.open(input_file))
                self.update_progress(0, total_lines)  # Initialize progress bar

                # Use a thread pool for parallel processing of conversations
                with ThreadPoolExecutor(max_workers=4) as executor:
                    loop = asyncio.get_event_loop()
                    futures = []
                    
                    async with aiofiles.open(output_file, mode='w') as writer:
                        for i, conversation in enumerate(tqdm(reader, total=total_lines, desc="Filtering")):
                            futures.append(loop.run_in_executor(executor, self.process_conversation, conversation, url, threshold))
                        
                        for future in asyncio.as_completed(futures):
                            result = await future
                            if result:
                                await writer.write(jsonlines.dumps(result))

                        self.update_progress(total_lines, total_lines)  # Finalize progress bar

        except Exception as e:
            self.update_status(f"Error processing JSONL file: {e}")

    async def async_save_file(self, file_path):
        """Asynchronously save the file."""
        async with aiofiles.open(file_path, mode='w') as file:
            await file.write("File saved successfully.")  # Placeholder for actual file saving

    def process_conversation(self, conversation, url, threshold):
        """Process a single conversation and classify its sentences."""
        keep_conversation = True

        sentences = []
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                doc = self.nlp(turn.get('value', ''))
                sentences.extend(self.extract_sentences(doc))

        # Classify sentences in batches
        classifications = asyncio.run(self.classify_sentences_async(sentences, url))

        for classification in classifications:
            if classification and classification.get('positive', 0) > threshold:
                keep_conversation = False
                break

        if keep_conversation:
            return conversation
        return None

    def extract_sentences(self, doc):
        """Extract sentences from a Spacy Doc object."""
        return [sent.text.strip() for sent in doc.sents]

    def update_status(self, message):
        """Update the status bar with a message."""
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def update_progress(self, current, total):
        """Update the progress bar with the current progress."""
        if self.root:
            progress = (current / total) * 100
            self.root.after(0, lambda: self.progress_bar.config(value=progress))
            if current % 10 == 0:  # Update UI less frequently
                self.root.after(0, lambda: self.progress_bar.update_idletasks())
