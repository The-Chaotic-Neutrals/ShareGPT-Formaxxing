import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import asyncio
import torch  # Import for detecting CPU/GPU
from DeslopTool import filter_dataset  # Import the original filtering functionality
from DeslopTool_classifier import CharacterSlopFilter  # Import the new filtering class

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}. Error: {e}")
    return data

def load_filter_criteria(filter_files):
    filter_criteria = []
    for filter_file in filter_files:
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    return filter_criteria

class DeslopToolApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme

        self.root = tk.Toplevel(self.master)
        self.root.title("DeslopMancer")

        self.set_icon()
        self.root.configure(bg=self.theme['bg'])

        self.filter_files = []  # List to store selected filter files
        self.last_filter_file_path = Path('last_filter_file.txt')  # File to store last filter file path

        # Initialize `selected_filter_method` before creating widgets
        self.selected_filter_method = tk.IntVar(value=1)
        self.batch_size = tk.IntVar(value=32)  # Default batch size
        self.force_gpu = tk.BooleanVar(value=False)  # Option to force GPU usage
        self.save_slop_file = tk.BooleanVar(value=False)  # Option to save slop file

        # Create widgets and update device status
        self.create_widgets()
        self.load_last_filter_file()  # Load last filter file at startup
        self.update_device_status()  # Initial device status

        # Initialize CharacterSlopFilter (updated later during processing)
        self.slop_filter = None

    def set_icon(self):
        icon_path = "icon.ico"
        if Path(icon_path).exists():
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def load_last_filter_file(self):
        """Load the last selected filter file path from a text file."""
        if self.last_filter_file_path.exists():
            with open(self.last_filter_file_path, 'r', encoding='utf-8', errors='replace') as f:
                last_filter_file = f.read().strip()
                if last_filter_file:
                    self.filter_files.append(last_filter_file)  # Pre-fill the last filter file
                    self.last_filter_label.config(text=f"Last Selected Filter File: {last_filter_file}")  # Update label

    def save_last_filter_file(self, filter_file):
        """Save the last selected filter file path to a text file."""
        with open(self.last_filter_file_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(filter_file)

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Dataset File:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.dataset_entry = tk.Entry(self.root, width=50, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.dataset_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self.root, text="Browse...", command=self.select_file, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.filter_button = tk.Button(self.root, text="Select Filter Files...", command=self.select_filter_files, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.filter_button.grid(row=1, column=0, columnspan=3, pady=10)

        self.last_filter_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.last_filter_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        # Add threshold input
        self.threshold_label = tk.Label(self.root, text="Threshold (as a multiple of average):", bg=self.theme['bg'], fg=self.theme['fg'])
        self.threshold_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')

        self.threshold_entry = tk.Entry(self.root, width=10, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.threshold_entry.grid(row=3, column=1, padx=10, pady=10)

        # Add batch size input
        self.batch_label = tk.Label(self.root, text="Batch Size:", bg=self.theme['bg'], fg=self.theme['fg'])
        self.batch_label.grid(row=4, column=0, padx=10, pady=10, sticky='w')

        self.batch_entry = tk.Entry(self.root, width=10, textvariable=self.batch_size, bg=self.theme['entry_bg'], fg=self.theme['entry_fg'], insertbackground=self.theme['entry_fg'])
        self.batch_entry.grid(row=4, column=1, padx=10, pady=10)

        # Create a frame for the selection buttons
        self.selection_frame = tk.Frame(self.root, bg=self.theme['bg'])
        self.selection_frame.grid(row=5, column=0, columnspan=3, pady=10)

        # Add styled radiobuttons for filtering methods
        self.string_match_radiobutton = tk.Radiobutton(
            self.selection_frame,
            text="String Matching Filter",
            variable=self.selected_filter_method,
            value=1,
            command=self.update_button_styles,
            bg='#333333', fg='gold', font=("Arial", 12, "bold"),
            indicatoron=0, width=20, relief="solid"
        )
        self.string_match_radiobutton.pack(side=tk.LEFT, padx=10, pady=5)

        self.classifier_radiobutton = tk.Radiobutton(
            self.selection_frame,
            text="Classifier Removal",
            variable=self.selected_filter_method,
            value=2,
            command=self.update_button_styles,
            bg='#333333', fg='gold', font=("Arial", 12, "bold"),
            indicatoron=0, width=20, relief="solid"
        )
        self.classifier_radiobutton.pack(side=tk.LEFT, padx=10, pady=5)

        # Add styled checkbox for GPU toggle
        self.force_gpu_checkbox = tk.Checkbutton(
            self.selection_frame,
            text="Force GPU",
            variable=self.force_gpu,
            command=self.update_device_status,
            bg='#333333', fg='gold', font=("Arial", 12, "bold"),
            indicatoron=0, width=15, relief="solid"
        )
        self.force_gpu_checkbox.pack(side=tk.LEFT, padx=10, pady=5)

        # Update button styles initially
        self.update_button_styles()

        # Display device information
        self.device_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.device_label.grid(row=8, column=0, columnspan=3, pady=10)

        self.process_button = tk.Button(self.root, text="Process Dataset", command=self.process_dataset, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.process_button.grid(row=9, column=0, columnspan=3, pady=10)

        self.result_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.result_label.grid(row=10, column=0, columnspan=3, padx=10, pady=10)

    def update_button_styles(self):
        """Update the styles of the radiobuttons and checkbox based on the selection."""
        # Update the filtering method buttons
        if self.selected_filter_method.get() == 1:  # String Matching Filter selected
            self.string_match_radiobutton.config(bg='white', fg='black')
            self.classifier_radiobutton.config(bg='#333333', fg='gold')
        elif self.selected_filter_method.get() == 2:  # Classifier Removal selected
            self.string_match_radiobutton.config(bg='#333333', fg='gold')
            self.classifier_radiobutton.config(bg='white', fg='black')

        # Update the GPU checkbox style based on its state
        if self.force_gpu.get():
            self.force_gpu_checkbox.config(bg='white', fg='black')
        else:
            self.force_gpu_checkbox.config(bg='#333333', fg='gold')


    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Lines files", "*.jsonl")]
        )
        if file_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, file_path)

    def select_filter_files(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Text files", "*.txt")]
        )
        if file_paths:
            self.filter_files = list(file_paths)
            last_filter_file = self.filter_files[-1]  # Get the last selected filter file
            self.save_last_filter_file(last_filter_file)  # Save the last selected filter file
            self.last_filter_label.config(text=f"Last Selected Filter File: {last_filter_file}")  # Update label

    def update_device_status(self):
        # Update the device information based on the force GPU checkbox
        if self.force_gpu.get() or torch.cuda.is_available():
            device_info = "GPU (Forced)" if self.force_gpu.get() else "GPU"
        else:
            device_info = "CPU"
        self.device_label.config(text=f"Running on: {device_info}")

    def process_dataset(self):
        dataset_file = self.dataset_entry.get().strip()
        if not dataset_file:
            messagebox.showerror("Input Error", "Please select a dataset file.")
            return

        # Get the batch size from user input
        batch_size = self.batch_size.get()

        # Determine device
        device = 0 if self.force_gpu.get() or torch.cuda.is_available() else -1

        if self.selected_filter_method.get() == 1:
            # Use string matching filter method
            self.process_with_original_method(dataset_file)
        elif self.selected_filter_method.get() == 2:
            # Use classifier removal filter method
            self.slop_filter = CharacterSlopFilter(batch_size=batch_size, confidence_margin=0.1)
            asyncio.run(self.process_with_slop_filter(dataset_file, device))
        else:
            messagebox.showerror("Error", "Invalid filter method selected.")

    def process_with_original_method(self, dataset_file):
        if not self.filter_files:
            messagebox.showerror("Input Error", "Please select at least one filter file.")
            return

        filter_criteria = load_filter_criteria(self.filter_files) or []
        if not filter_criteria:
            messagebox.showerror("Input Error", "Filter criteria are empty. Please check your filter files.")
            return

        threshold_value = self.threshold_entry.get().strip()
        
        try:
            average_results = self.calculate_average_phrases(dataset_file, filter_criteria)
            average_matched = average_results['average']

            if threshold_value:
                threshold_multiplier = float(threshold_value)
                threshold = average_matched * threshold_multiplier
                output_message = filter_dataset(dataset_file, self.filter_files, threshold)
            else:
                output_message = f"Average matched phrases per conversation: {average_matched:.2f}\n" \
                                 f"Total conversations: {average_results['total_conversations']}\n" \
                                 f"Conversations with phrases above average: {average_results['above_average']}\n"

            self.result_label.config(text=output_message)
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    async def process_with_slop_filter(self, dataset_file, device):
        # Create the 'deslopped' directory if it doesn't exist
        output_dir = Path('./deslopped')
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
       
       # Define the output file path inside the 'deslopped' directory with the '_deslopped.jsonl' extension
        output_jsonl_filepath = output_dir / (Path(dataset_file).stem + "_deslopped.jsonl")
        await self.slop_filter.filter_conversations(dataset_file, output_jsonl_filepath)
        self.result_label.config(text=f"Filtered conversations saved to {output_jsonl_filepath}")

    def calculate_average_phrases(self, dataset_file, filter_criteria):
        data = load_jsonl(dataset_file)

        total_phrases = 0
        total_conversations = len(data)
        above_average_count = 0

        for conversation in data:
            # Ensure conversation is a dictionary
            if not isinstance(conversation, dict):
                print(f"Skipping non-dictionary conversation: {conversation}")
                continue

            # Get the conversation list, ensure it's a list
            conversation_list = conversation.get("conversations", [])
            if not isinstance(conversation_list, list):
                print(f"Invalid conversation format: {conversation_list}")
                continue

            matched_count = sum(
                sum(1 for phrase in filter_criteria if phrase in (msg.get("value") or ""))
                for msg in conversation_list if msg.get("from") == "gpt"
            )
            total_phrases += matched_count
            
            if matched_count > (total_phrases / total_conversations if total_conversations > 0 else 0):
                above_average_count += 1

        average = total_phrases / total_conversations if total_conversations > 0 else 0

        return {
            "average": average,
            "total_conversations": total_conversations,
            "above_average": above_average_count
        }

def run_app():
    root = tk.Tk()
    root.withdraw()  # Hide the main root window

    theme = {
        'bg': 'lightgray',
        'fg': 'black',
        'entry_bg': 'white',
        'entry_fg': 'black',
        'button_bg': 'gray',
        'button_fg': 'white'
    }
    app = DeslopToolApp(root, theme)

    root.mainloop()

if __name__ == "__main__":
    run_app()