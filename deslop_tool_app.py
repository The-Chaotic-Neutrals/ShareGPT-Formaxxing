import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from pathlib import Path
import yaml
import polars as pl

class DeslopToolApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme
        
        # Create a new top-level window
        self.root = tk.Toplevel(self.master)
        self.root.title("Deslop Tool")
        
        # Set the icon for the Deslop Tool window
        self.set_icon()

        # Configure the top-level window
        self.root.configure(bg=self.theme['bg'])

        # Create and place widgets
        self.create_widgets()

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Dataset File:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.dataset_entry = tk.Entry(self.root, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.dataset_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self.root, text="Browse...", command=self.select_file, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.process_button = tk.Button(self.root, text="Process Dataset", command=self.process_dataset, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.process_button.grid(row=1, column=0, columnspan=3, pady=10)

        self.result_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

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

        try:
            output_message = self.filter_dataset(file_path, Path(__file__).parent.absolute())
            self.result_label.config(text=output_message)
        except ValueError as e:
            messagebox.showerror("Processing Error", str(e))

    def filter_dataset(self, input_path, output_dir):
        try:
            # Read the JSON Lines file
            with open(input_path, 'r') as file:
                data = [json.loads(line) for line in file]

            # Load the YAML file for filtering criteria
            yaml_file_path = Path(output_dir) / 'filter_criteria.yaml'
            with open(yaml_file_path, 'r') as yaml_file:
                filter_criteria = yaml.safe_load(yaml_file)

            # Define a function to check if a conversation matches any filter criteria
            def matches_criteria(conversation):
                if isinstance(conversation, str):
                    try:
                        conversation = json.loads(conversation)
                    except json.JSONDecodeError:
                        return False
                if isinstance(conversation, list):
                    return any(
                        all(
                            key in msg and msg[key] == value
                            for key, value in criteria.items()
                        )
                        for msg in conversation
                        for criteria in filter_criteria
                    )
                return False

            # Convert the data to a Polars DataFrame
            df = pl.DataFrame(data)

            # Create a boolean column based on the filter criteria
            df = df.with_columns(
                pl.col('conversations').map_elements(matches_criteria, return_dtype=pl.Boolean).alias('matches_criteria')
            )

            # Filter the data based on the new boolean column (keep non-matching entries)
            filtered_data = df.filter(~pl.col('matches_criteria'))  # Invert the condition to keep non-matching entries

            # Create the deslop directory if it doesn't exist
            deslop_dir = Path(output_dir) / "deslop"
            deslop_dir.mkdir(exist_ok=True)

            # Save the filtered data as JSONL
            output_file = deslop_dir / f"{Path(input_path).stem}_deslop.jsonl"

            with output_file.open('w') as f:
                for row in filtered_data.drop('matches_criteria').to_dicts():
                    json.dump(row, f)
                    f.write('\n')

            return f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(filtered_data)}"

        except Exception as e:
            raise ValueError(f"Error during filtering: {str(e)}")