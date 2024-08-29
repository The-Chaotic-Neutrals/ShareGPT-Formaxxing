import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import pandas as pd

class DatasetConverterApp:
    def __init__(self, parent, theme):
        self.top = tk.Toplevel(parent)
        self.top.title("Dataset Converter")
        self.theme = theme
        self.top.configure(bg=self.theme['bg'])
        self.setup_ui()
        self.set_icon()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.top.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def setup_ui(self):
        self.top.geometry("600x350")

        # Input file selection
        self.entry_input_file = self.create_labeled_entry("Input File:", self.select_input_file, row=0)

        # Convert button
        self.convert_button = tk.Button(
            self.top, text="Convert", command=self.on_convert_button_click,
            bg=self.theme['button_bg'], fg=self.theme['button_fg']
        )
        self.convert_button.grid(row=1, column=0, pady=20, padx=10, sticky="ew")

        # Preview text box for output
        self.preview_text = tk.Text(
            self.top, wrap=tk.WORD, font=('Consolas', 10),
            bg=self.theme['text_bg'], fg=self.theme['text_fg'], height=10
        )
        self.preview_text.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Status bar using a Text widget for copyable text
        self.status_bar = tk.Text(self.top, height=2, bg=self.theme['bg'], fg=self.theme['fg'], wrap=tk.WORD)
        self.status_bar.grid(row=3, column=0, sticky="ew")
        self.status_bar.insert(tk.END, "Ready")
        self.status_bar.config(state=tk.DISABLED)  # Make the text read-only

        # Ensure the rows and columns expand properly
        self.top.rowconfigure(2, weight=1)
        self.top.columnconfigure(0, weight=1)

    def create_labeled_entry(self, label_text, command, row):
        frame = tk.Frame(self.top, bg=self.theme['bg'])
        frame.grid(row=row, column=0, padx=10, pady=5, sticky="ew")

        label = tk.Label(frame, text=label_text, bg=self.theme['bg'], fg=self.theme['fg'])
        label.pack(side=tk.LEFT)

        entry = tk.Entry(frame, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_button = tk.Button(frame, text="Browse...", command=command, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        browse_button.pack(side=tk.RIGHT, padx=10)

        return entry

    def select_input_file(self):
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
            file_path = filedialog.askopenfilename(filetypes=filetypes)
            if file_path:
                self.entry_input_file.delete(0, tk.END)
                self.entry_input_file.insert(0, file_path)
                self.update_status(f"Selected file: {file_path}")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("File Selection Error", f"An error occurred: {str(e)}")

    def on_convert_button_click(self):
        input_path = self.entry_input_file.get()
        if input_path:
            self.convert_dataset(input_path)
        else:
            self.update_status("No input file selected.")
            messagebox.showerror("Input Error", "Please select an input file.")

    def convert_dataset(self, input_path):
        try:
            self.update_status("Conversion in progress...")
            # Automatically generate output path
            output_path = self.generate_output_path(input_path)
            
            data = self.load_data(input_path)
            if data is not None:
                with open(output_path, 'w') as outfile:
                    if isinstance(data, list):
                        for record in data:
                            json.dump(record, outfile)
                            outfile.write('\n')
                    else:
                        json.dump(data, outfile, indent=2)

                # Preview first 10 records if available
                preview_data = data[:10] if isinstance(data, list) else [data]
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(tk.END, json.dumps(preview_data, indent=2))
                
                # Update status bar with success message
                self.update_status(f"Conversion completed: {len(data)} records saved to {output_path}.")
            else:
                self.update_status("No data processed.")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def generate_output_path(self, input_path):
        # Define the output directory relative to the script's current directory
        output_dir = os.path.join(os.getcwd(), 'converted')
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = base_name + '_converted.jsonl'
        
        return os.path.join(output_dir, output_file)

    def load_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        self.update_status(f"Loading data from {file_path}")
        if ext == '.json':
            with open(file_path, 'r') as infile:
                return json.load(infile)
        elif ext == '.csv':
            return pd.read_csv(file_path).to_dict(orient='records')
        elif ext == '.txt':
            with open(file_path, 'r') as infile:
                return infile.readlines()
        elif ext == '.jsonl':
            with open(file_path, 'r') as infile:
                return [json.loads(line) for line in infile]
        else:
            raise ValueError("Unsupported file format.")

    def update_status(self, message):
        self.status_bar.config(state=tk.NORMAL)  # Make the text editable to update it
        self.status_bar.delete(1.0, tk.END)  # Clear the current text
        self.status_bar.insert(tk.END, message)  # Insert the new message
        self.status_bar.config(state=tk.DISABLED)  # Make the text read-only again
