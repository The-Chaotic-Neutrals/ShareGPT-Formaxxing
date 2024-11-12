import tkinter as tk 
from tkinter import filedialog, messagebox
import os
import json
from pathlib import Path
from dataset_filter import filter_dataset

class DatasetFilterApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme
        
        # Create a new top-level window
        self.root = tk.Toplevel(self.master)
        self.root.title("DataMaxxer")
        
        # Set the icon for the DataMaxxer window
        self.set_icon()

        # Configure the top-level window
        self.root.configure(bg=self.theme['bg'])

        # Initialize filtering method toggle variables
        self.init_filtering_variables()

        # Create and place widgets
        self.create_widgets()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def init_filtering_variables(self):
        # Initialize boolean variables for each filtering method
        self.check_blank_turns = tk.BooleanVar(value=True)
        self.check_invalid_endings = tk.BooleanVar(value=True)
        self.check_null_gpt = tk.BooleanVar(value=True)
        self.check_duplicate_system = tk.BooleanVar(value=True)

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Dataset File:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.dataset_entry = tk.Entry(self.root, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.dataset_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self.root, text="Browse...", command=self.select_file, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        # Add checkboxes for each filtering method
        self.add_checkboxes()

        self.process_button = tk.Button(self.root, text="Process Dataset", command=self.process_dataset, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.process_button.grid(row=7, column=0, columnspan=3, pady=10)

        self.result_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.result_label.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

    def add_checkboxes(self):
        # Create checkboxes for each filtering method
        self.blank_turns_cb = tk.Checkbutton(self.root, text="Check Blank Turns", variable=self.check_blank_turns, bg=self.theme['bg'], fg=self.theme['fg'])
        self.blank_turns_cb.grid(row=1, column=0, columnspan=2, sticky='w', padx=10, pady=2)

        self.invalid_endings_cb = tk.Checkbutton(self.root, text="Check Invalid Endings", variable=self.check_invalid_endings, bg=self.theme['bg'], fg=self.theme['fg'])
        self.invalid_endings_cb.grid(row=2, column=0, columnspan=2, sticky='w', padx=10, pady=2)

        self.null_gpt_cb = tk.Checkbutton(self.root, text="Check Null GPT", variable=self.check_null_gpt, bg=self.theme['bg'], fg=self.theme['fg'])
        self.null_gpt_cb.grid(row=3, column=0, columnspan=2, sticky='w', padx=10, pady=2)

        self.duplicate_system_cb = tk.Checkbutton(self.root, text="Check Duplicate System", variable=self.check_duplicate_system, bg=self.theme['bg'], fg=self.theme['fg'])
        self.duplicate_system_cb.grid(row=4, column=0, columnspan=2, sticky='w', padx=10, pady=2)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Lines files", "*.jsonl")]
        )
        if file_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, file_path)

    def process_dataset(self):
        file_path = self.dataset_entry.get().strip()
        if not file_path:
            messagebox.showerror("Input Error", "Please select a dataset file.")
            return

        print(f"Selected file path: {file_path}")  # Debug statement

        # Check if the file is readable
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first 5 lines as a sample with encoding and error handling
                sample_data = [json.loads(line) for line in f][:5]
            print(f"Sample data: {sample_data}")  # Debug statement
        except UnicodeDecodeError:
            messagebox.showerror("File Error", "Unicode decoding error. Please make sure the file is UTF-8 encoded.")
            return
        except json.JSONDecodeError:
            messagebox.showerror("File Error", "JSON parsing error. The file may contain invalid JSON.")
            return
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file: {str(e)}")
            return

        try:
            # Get the selected options from the checkboxes
            output_message = filter_dataset(
                file_path,
                Path(__file__).parent.absolute(),
                check_blank_turns=self.check_blank_turns.get(),
                check_invalid_endings=self.check_invalid_endings.get(),
                check_null_gpt=self.check_null_gpt.get(),
                check_duplicate_system=self.check_duplicate_system.get()
            )
            self.result_label.config(text=output_message)
        except ValueError as e:
            messagebox.showerror("Processing Error", str(e))
        except Exception as e:
            # Catch any unexpected exceptions
            messagebox.showerror("Processing Error", f"An unexpected error occurred: {str(e)}")
