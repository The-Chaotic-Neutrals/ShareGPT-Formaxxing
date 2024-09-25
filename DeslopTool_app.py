import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from DeslopTool import filter_dataset  # Import the filtering functionality

class DeslopToolApp:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme

        self.root = tk.Toplevel(self.master)
        self.root.title("Deslop Tool")

        self.set_icon()
        self.root.configure(bg=self.theme['bg'])

        self.filter_files = []  # List to store selected filter files
        self.last_filter_file_path = Path('last_filter_file.txt')  # File to store last filter file path

        self.create_widgets()  # Create the widgets first
        self.load_last_filter_file()  # Load last filter file at startup

    def set_icon(self):
        icon_path = "icon.ico"
        if Path(icon_path).exists():
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def load_last_filter_file(self):
        """Load the last selected filter file path from a text file."""
        if self.last_filter_file_path.exists():
            with open(self.last_filter_file_path, 'r') as f:
                last_filter_file = f.read().strip()
                if last_filter_file:
                    self.filter_files.append(last_filter_file)  # Pre-fill the last filter file
                    self.last_filter_label.config(text=f"Last Selected Filter File: {last_filter_file}")  # Update label

    def save_last_filter_file(self, filter_file):
        """Save the last selected filter file path to a text file."""
        with open(self.last_filter_file_path, 'w') as f:
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

        # Initialize last_filter_label here
        self.last_filter_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.last_filter_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.process_button = tk.Button(self.root, text="Process Dataset", command=self.process_dataset, bg=self.theme['button_bg'], fg=self.theme['button_fg'])
        self.process_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.result_label = tk.Label(self.root, text="", bg=self.theme['bg'], fg=self.theme['fg'])
        self.result_label.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

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

    def process_dataset(self):
        dataset_file = self.dataset_entry.get().strip()
        if not dataset_file:
            messagebox.showerror("Input Error", "Please select a dataset file.")
            return

        if not self.filter_files:
            messagebox.showerror("Input Error", "Please select at least one filter file.")
            return

        try:
            output_message = filter_dataset(dataset_file, self.filter_files)  # Call the filter function
            self.result_label.config(text=output_message)
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

def run_app():
    root = tk.Tk()
    root.withdraw()  # Hide the main root window

    # Create an instance of the DeslopToolApp class
    theme = {
        'bg': 'lightgray',
        'fg': 'black',
        'entry_bg': 'white',
        'entry_fg': 'black',
        'button_bg': 'gray',
        'button_fg': 'white'
    }
    app = DeslopToolApp(root, theme)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    run_app()
