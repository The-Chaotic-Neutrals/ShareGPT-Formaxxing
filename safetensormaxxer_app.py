import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import concurrent.futures
import queue
import threading
from safetensormaxxer import SafetensorMaxxer

class SafetensorMaxxerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.safetensor_maxxer = SafetensorMaxxer()
        self.queue = queue.Queue()  # Queue for thread-safe status updates

        # Output folder is automatically set to the folder 'safetensorfied' in the current working directory
        self.safetensor_maxxer.output_folder = os.path.join(os.getcwd(), "safetensorfied")
        if not os.path.exists(self.safetensor_maxxer.output_folder):
            os.makedirs(self.safetensor_maxxer.output_folder)

        # Apply theme and icon
        self.apply_theme()
        self.set_icon()

        # Set initial window size
        self.root.geometry("400x400")  # Set initial window size to 400x400 pixels

        # UI Elements
        self.setup_ui()

        # Start the UI update loop
        self.update_ui()

    def apply_theme(self):
        """Apply theme to the app's root window."""
        self.root.configure(bg=self.theme.get('bg', 'white'))

    def set_icon(self):
        """Set the icon for the window."""
        icon_path = "icon.ico"  # Use the icon file path if it's in the same directory
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        else:
            print("Icon file not found.")

    def setup_ui(self):
        """Set up the user interface with grid layout."""
        frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.9, relheight=0.6)

        # Center the buttons in the frame
        self.select_button = tk.Button(frame, text="Select Model Folder", command=self.select_model_path, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
        self.select_button.grid(row=0, column=0, pady=10, padx=10, sticky='ew')

        self.convert_button = tk.Button(frame, text="Start Conversion", command=self.start_conversion, bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
        self.convert_button.grid(row=1, column=0, pady=20, padx=10, sticky='ew')

        # Create a status bar at the bottom of the window
        self.status_bar = tk.Label(self.root, text="Ready", anchor='w', bg=self.theme.get('status_bar_bg', 'lightgrey'), fg=self.theme.get('status_bar_fg', 'black'))
        self.status_bar.pack(side='bottom', fill='x')

        # Centering the buttons horizontally in the frame
        frame.columnconfigure(0, weight=1)

    def start_conversion(self):
        """Start the conversion process using a thread pool executor."""
        # Disable the convert button to prevent multiple clicks
        self.convert_button.config(state=tk.DISABLED)

        # Create a ThreadPoolExecutor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Submit the conversion task to the executor
        self.future = self.executor.submit(self.run_conversion)

    def run_conversion(self):
        """Run the conversion process and update the status bar."""
        self.queue.put("Conversion in progress...")
        try:
            if not self.safetensor_maxxer.model_path:
                self.queue.put("Conversion stopped. No folder selected.")
                return

            index_filename = os.path.join(self.safetensor_maxxer.model_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_filename):
                self.safetensor_maxxer.index_filename = index_filename
                operations, errors = self.safetensor_maxxer.convert_multi_local(
                    self.safetensor_maxxer.index_filename,
                    self.safetensor_maxxer.model_path,
                    self.safetensor_maxxer.output_folder,
                    self.safetensor_maxxer.discard_names
                )
            else:
                pt_filename = os.path.join(self.safetensor_maxxer.model_path, "pytorch_model.bin")
                sf_filename = os.path.join(self.safetensor_maxxer.output_folder, "model.safetensors")
                operations, errors = self.safetensor_maxxer.convert_single_local(
                    pt_filename, sf_filename, self.safetensor_maxxer.discard_names
                )

            if operations:
                self.queue.put(f"Successfully converted files: {operations}")
            if errors:
                error_msg = "\n".join([f"Error converting {filename}: {error}" for filename, error in errors])
                self.queue.put("Conversion completed with errors.")
                self.root.after(0, messagebox.showerror, "Conversion Errors", error_msg)
            else:
                self.queue.put("Conversion completed successfully.")

            # Copy all .json files from the model path to the output folder, excluding specific file
            self.copy_json_files()

        except Exception as e:
            self.queue.put("Conversion stopped due to an error.")
            self.root.after(0, messagebox.showerror, "Error", f"An error occurred: {e}")

        finally:
            # Re-enable the convert button
            self.root.after(0, self.convert_button.config, {'state': tk.NORMAL})

    def update_ui(self):
        """Update the status bar from the queue."""
        while not self.queue.empty():
            message = self.queue.get()
            self.status_bar.config(text=message)
        self.root.after(100, self.update_ui)

    def select_model_path(self):
        """Open a file dialog to select the input folder."""
        self.safetensor_maxxer.model_path = filedialog.askdirectory(title="Select Model Folder")
        if not self.safetensor_maxxer.model_path:
            self.root.after(0, messagebox.showwarning, "Input Folder", "No folder selected")

    def copy_json_files(self):
        """Copy all .json files from the input folder to the output folder, excluding `pytorch_model.bin.index.json`."""
        if not self.safetensor_maxxer.model_path:
            return

        for filename in os.listdir(self.safetensor_maxxer.model_path):
            if filename.endswith(".json") and filename != "pytorch_model.bin.index.json":
                src_path = os.path.join(self.safetensor_maxxer.model_path, filename)
                dest_path = os.path.join(self.safetensor_maxxer.output_folder, filename)
                try:
                    shutil.copy(src_path, dest_path)
                except Exception as e:
                    self.queue.put(f"Error copying {filename}: {e}")
