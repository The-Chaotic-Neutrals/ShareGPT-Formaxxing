import tkinter as tk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import urllib.parse

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

class BinaryClassificationServerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.server_thread = None
        self.server_running = False
        self.total_positive = 0
        self.total_negative = 0
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface for the Binary Classification Server app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Binary Classification Server")
        self.window.configure(bg=self.theme.get('bg', 'white'))
        self.window.iconbitmap('icon.ico')  # Ensure 'icon.ico' is in the same directory as your script

        # Main frame
        main_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Port entry frame
        port_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        port_frame.pack(pady=5)

        self.port_label = tk.Label(port_frame, text="Port:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.port_label.pack(side=tk.LEFT, padx=5)
        
        self.port_entry = tk.Entry(port_frame)
        self.port_entry.pack(side=tk.LEFT, padx=5)

        # Start and Stop buttons
        button_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Server", command=self.start_server, bg=self.theme.get('button_bg', 'lightgrey'))
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop Server", command=self.stop_server, bg=self.theme.get('button_bg', 'lightgrey'))
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_bar = tk.Label(main_frame, text="Status: Idle", bg=self.theme.get('bg', 'lightgrey'), fg=self.theme.get('fg', 'black'))
        self.status_bar.pack(pady=5, fill=tk.X)

        # Device indicator
        self.device_indicator = tk.Label(main_frame, text=f"Running on: {self.device.upper()}", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.device_indicator.pack(pady=5)

        # Classification counts
        counts_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        counts_frame.pack(pady=10, fill=tk.X)

        self.counts_label = tk.Label(counts_frame, text="Classification Counts:", bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.counts_label.pack(anchor=tk.W)

        self.positive_count_label = tk.Label(counts_frame, text=f"Positive: {self.total_positive}", bg=self.theme.get('bg', 'lightgrey'), fg=self.theme.get('fg', 'black'))
        self.positive_count_label.pack(pady=5, anchor=tk.W)

        self.negative_count_label = tk.Label(counts_frame, text=f"Negative: {self.total_negative}", bg=self.theme.get('bg', 'lightgrey'), fg=self.theme.get('fg', 'black'))
        self.negative_count_label.pack(pady=5, anchor=tk.W)

    def update_status(self, message):
        """Update the status bar with a message."""
        self.status_bar.config(text=f"Status: {message}")

    def update_counts(self, positive, negative):
        """Update the counts for positive and negative classifications."""
        self.total_positive += positive
        self.total_negative += negative
        self.positive_count_label.config(text=f"Positive: {self.total_positive}")
        self.negative_count_label.config(text=f"Negative: {self.total_negative}")

    def start_server(self):
        """Start the server in a separate thread."""
        if not self.server_running:
            port = self.port_entry.get()
            if not port:
                port = 8120  # Default port
            try:
                port = int(port)
                self.server_thread = threading.Thread(target=self.run_server, args=(port,))
                self.server_thread.start()
                self.server_running = True
                self.update_status(f"Server started on port {port}.")
            except ValueError:
                self.update_status("Invalid port number.")
        else:
            self.update_status("Server is already running.")

    def stop_server(self):
        """Stop the server."""
        if self.server_running:
            self.server_running = False
            if self.server_thread:
                self.server_thread.join(timeout=5)
            self.update_status("Server stopped.")
        else:
            self.update_status("Server is not running.")

    def run_server(self, port):
        """Run an HTTP server to listen for incoming requests."""
        self.update_status("Server is running and waiting for connections.")
        server_address = ('0.0.0.0', port)
        httpd = HTTPServer(server_address, self.RequestHandler)
        httpd.request_handler = self
        while self.server_running:
            try:
                httpd.handle_request()
            except Exception as e:
                self.update_status(f"Server error: {e}")
                break
        httpd.server_close()
        self.update_status("Server has stopped or encountered an error.")

    class RequestHandler(BaseHTTPRequestHandler):
        """Handle HTTP requests for the server."""
        
        def do_GET(self):
            """Handle GET requests."""
            if self.path.startswith('/predict'):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                texts = params.get('texts', [None])[0]

                if texts:
                    texts = json.loads(texts)  # Assuming texts is a JSON-encoded list
                    response = self.server.request_handler.predict(texts)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"results": response}).encode('utf-8'))
                    # Update counts
                    positive_count = sum(1 for res in response if res['positive'] > 0.5)
                    negative_count = len(response) - positive_count
                    self.server.request_handler.update_counts(positive_count, negative_count)
                else:
                    self.send_error(400, "Bad Request: Missing 'texts' parameter.")
            else:
                self.send_error(404, "Not Found: Endpoint does not exist.")
        
        def do_POST(self):
            """Handle POST requests."""
            if self.path.startswith('/predict'):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data)
                    texts = data.get('texts', None)
                    
                    if texts:
                        response = self.server.request_handler.predict(texts)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"results": response}).encode('utf-8'))
                        # Update counts
                        positive_count = sum(1 for res in response if res['positive'] > 0.5)
                        negative_count = len(response) - positive_count
                        self.server.request_handler.update_counts(positive_count, negative_count)
                    else:
                        self.send_error(400, "Bad Request: Missing 'texts' field in JSON.")
                except json.JSONDecodeError:
                    self.send_error(400, "Bad Request: Invalid JSON format.")
            else:
                self.send_error(404, "Not Found: Endpoint does not exist.")

    def predict(self, texts):
        """Predict the classifications for a batch of texts."""
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()
        results = [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]
        return results
