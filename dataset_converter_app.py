import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from dataset_converter import DatasetConverter  # Adjust the import based on your file structure
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.top.geometry("600x400")

        # Input file selection
        self.entry_input_file = self.create_labeled_entry("Input Files:", self.select_input_files, row=0)

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

    def select_input_files(self):
        filetypes = [
            ("All Supported Files", "*.json;*.jsonl"),
            ("JSON files", "*.json"),
            ("JSON Lines files", "*.jsonl")
        ]
        try:
            file_paths = filedialog.askopenfilenames(filetypes=filetypes)
            if file_paths:
                self.entry_input_file.delete(0, tk.END)
                self.entry_input_file.insert(0, "; ".join(file_paths))
                self.update_status(f"Selected files: {', '.join(file_paths)}")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("File Selection Error", f"An error occurred: {str(e)}")

    def on_convert_button_click(self):
        input_paths = self.entry_input_file.get().split("; ")
        if input_paths:
            self.convert_button.config(state=tk.DISABLED)  # Disable button
            self.convert_multiple_datasets(input_paths)
            self.convert_button.config(state=tk.NORMAL)  # Re-enable button
        else:
            self.update_status("No input files selected.")
            messagebox.showerror("Input Error", "Please select input files.")

    def convert_multiple_datasets(self, input_paths):
        try:
            self.update_status("Conversion in progress...")
            logging.info(f"Starting conversion for {', '.join(input_paths)}")

            # Generate output directory
            output_dir = os.path.join(os.getcwd(), 'converted')
            os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Process each file
            for input_path in input_paths:
                self.update_status(f"Processing file: {input_path}")
                logging.info(f"Processing file: {input_path}")

                # Process each file
                preview_entries = DatasetConverter.process_multiple_files([input_path], output_dir)
                self.update_preview(preview_entries.get(os.path.basename(input_path), []))

            self.update_status(f"Conversion completed for all selected files.")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_preview(self, preview_entries):
        preview_data = preview_entries[:10] if isinstance(preview_entries, list) else [preview_entries]
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, json.dumps(preview_data, ensure_ascii=False, indent=2))

    def update_status(self, message):
        self.status_bar.config(state=tk.NORMAL)  # Make the text editable to update it
        self.status_bar.delete(1.0, tk.END)  # Clear the current text
        self.status_bar.insert(tk.END, message)  # Insert the new message
        self.status_bar.config(state=tk.DISABLED)  # Make the text read-only again
