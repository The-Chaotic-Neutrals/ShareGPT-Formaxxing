import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
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
        self.queue = queue.Queue()
        self.safetensor_maxxer = SafetensorMaxxer()
        self.executor = None
        self.future = None

        self.safetensor_maxxer.output_folder = os.path.join(os.getcwd(), "safetensorfied")
        os.makedirs(self.safetensor_maxxer.output_folder, exist_ok=True)

        self.apply_theme()
        self.set_icon()
        self.root.geometry("600x500")
        self.setup_ui()
        self.update_ui()

    def apply_theme(self):
        self.root.configure(bg=self.theme.get('bg', 'white'))

    def set_icon(self):
        icon_path = "icon.ico"
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'white'))
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Buttons section
        button_frame = tk.LabelFrame(main_frame, text="Model Tools", bg=self.theme.get('bg', 'white'), fg='black')
        button_frame.pack(fill='x', padx=5, pady=5)

        self.select_button = tk.Button(button_frame, text="📁 Select Model Folder", command=self.select_model_path,
                                       bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.select_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

        self.convert_button = tk.Button(button_frame, text="⚙️ Start Conversion", command=self.start_conversion,
                                        bg=self.theme.get('button_bg', 'lightgrey'), fg=self.theme.get('button_fg', 'black'))
        self.convert_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        self.verify_button = tk.Button(button_frame, text="🔍 Verify Folder", command=self.select_verify_folder,
                                       bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.verify_button.grid(row=0, column=2, padx=10, pady=5, sticky='ew')

        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        # Output log
        self.log_output = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=18, bg="#f5f5f5", fg="black")
        self.log_output.pack(fill='both', expand=True, pady=(10, 0))

        # Status bar
        self.status_bar = tk.Label(
            self.root, text="Ready", anchor='w',
            bg=self.theme.get('status_bar_bg', 'lightgrey'), fg=self.theme.get('status_bar_fg', 'black')
        )
        self.status_bar.pack(side='bottom', fill='x')

    def log(self, message):
        self.queue.put(message)

    def print_log(self, message):
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.see(tk.END)

    def select_model_path(self):
        self.safetensor_maxxer.model_path = filedialog.askdirectory(title="Select Model Folder")
        if not self.safetensor_maxxer.model_path:
            self.root.after(0, messagebox.showwarning, "Input Folder", "No folder selected")
        else:
            self.log(f"📁 Selected: {self.safetensor_maxxer.model_path}")

    def start_conversion(self):
        self.convert_button.config(state=tk.DISABLED)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self.run_conversion)

    def run_conversion(self):
        self.log("🚀 Conversion started...")
        try:
            path = self.safetensor_maxxer.model_path
            if not path:
                self.log("⚠️ No model folder selected.")
                return

            index_filename = os.path.join(path, "pytorch_model.bin.index.json")
            if os.path.exists(index_filename):
                self.log("📦 Converting sharded model...")
                self.safetensor_maxxer.index_filename = index_filename
                operations, errors = self.safetensor_maxxer.convert_multi_local(
                    index_filename, path, self.safetensor_maxxer.output_folder, self.safetensor_maxxer.discard_names
                )
            else:
                self.log("📦 Converting single model...")
                pt_file = os.path.join(path, "pytorch_model.bin")
                sf_file = os.path.join(self.safetensor_maxxer.output_folder, "model.safetensors")
                operations, errors = self.safetensor_maxxer.convert_single_local(
                    pt_file, sf_file, self.safetensor_maxxer.discard_names
                )

            for op in operations:
                self.log(f"✅ Converted: {op}")

            if errors:
                error_msg = "\n".join([f"{f}: {e}" for f, e in errors])
                self.log("⚠️ Conversion completed with errors.")
                self.root.after(0, messagebox.showerror, "Conversion Errors", error_msg)
            else:
                self.log("✅ Conversion successful.")

            self.copy_json_files()

            if hasattr(self.safetensor_maxxer, "verify_and_fix_index"):
                self.log("🔍 Verifying index...")
                issues = self.safetensor_maxxer.verify_and_fix_index()
                if issues:
                    for issue in issues:
                        self.log(f"❗ {issue}")
                    self.root.after(0, messagebox.showwarning, "Verification Issues", "\n".join(issues))
                else:
                    self.log("✅ Index verified clean.")

            if hasattr(self.safetensor_maxxer, "show_token_info"):
                self.log("📨 Token Info:")
                self.safetensor_maxxer.show_token_info()
            if hasattr(self.safetensor_maxxer, "show_chat_preview"):
                self.log("💬 Chat Template:")
                self.safetensor_maxxer.show_chat_preview()

        except Exception as e:
            self.log("❌ Conversion error.")
            self.root.after(0, messagebox.showerror, "Error", str(e))
        finally:
            self.root.after(0, self.convert_button.config, {'state': tk.NORMAL})

    def select_verify_folder(self):
        folder = filedialog.askdirectory(title="Select Safetensor Output Folder")
        if not folder:
            self.root.after(0, messagebox.showwarning, "Folder", "No folder selected.")
            return
        self.log(f"📁 Verifying folder: {folder}")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self.verify_only, folder)

    def verify_only(self, folder):
        try:
            self.log("🔎 Verifying safetensors...")
            self.safetensor_maxxer.output_folder = folder
            if hasattr(self.safetensor_maxxer, "verify_and_fix_index"):
                issues = self.safetensor_maxxer.verify_and_fix_index()
                if issues:
                    for issue in issues:
                        self.log(f"❗ {issue}")
                    self.root.after(0, messagebox.showwarning, "Verification Issues", "\n".join(issues))
                else:
                    self.log("✅ All files verified successfully.")
            else:
                self.log("🚫 Core is missing verify_and_fix_index().")
        except Exception as e:
            self.log("❌ Verification failed.")
            self.root.after(0, messagebox.showerror, "Verification Error", str(e))

    def copy_json_files(self):
        path = self.safetensor_maxxer.model_path
        if not path:
            return
        for filename in os.listdir(path):
            if filename.endswith(".json") and filename != "pytorch_model.bin.index.json":
                src = os.path.join(path, filename)
                dst = os.path.join(self.safetensor_maxxer.output_folder, filename)
                try:
                    shutil.copy(src, dst)
                    self.log(f"📄 Copied JSON: {filename}")
                except Exception as e:
                    self.log(f"❌ Error copying {filename}: {e}")

    def update_ui(self):
        while not self.queue.empty():
            msg = self.queue.get()
            self.status_bar.config(text=msg)
            self.print_log(msg)
        self.root.after(100, self.update_ui)
