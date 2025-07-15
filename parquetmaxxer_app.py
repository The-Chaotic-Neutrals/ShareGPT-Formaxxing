import customtkinter as ctk
from tkinter import filedialog, messagebox
from multiprocessing import Process, Manager
from threading import Thread
import os

from parquetmaxxer import jsonl_to_parquet_worker, parquet_to_jsonl_worker

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class ParquetMaxxer(ctk.CTkToplevel):  # 👈 this is now a Toplevel
    def __init__(self, master=None):
        super().__init__(master)
        self.title("ParquetMaxxer — JSONL ⇄ Parquet Converter")
        self.geometry("900x640")
        self.resizable(False, False)
        self.configure(fg_color="#0A0A0A")

        # Icon support
        if os.path.exists("icon.ico"):
            try:
                self.iconbitmap("icon.ico")
            except Exception as e:
                print(f"Could not set icon: {e}")

        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="🧬 ParquetMaxxer",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#EEEEEE"
        )
        self.title_label.pack(pady=(20, 10))

        # Buttons
        self.button_frame = ctk.CTkFrame(self, fg_color="#0A0A0A")
        self.button_frame.pack(pady=10)

        neon_blue = "#00BFFF"
        self.convert_btn = ctk.CTkButton(
            self.button_frame, text="📥 Convert JSONL ➜ Parquet",
            command=self.convert_files, width=280, height=45,
            font=ctk.CTkFont(size=14),
            fg_color="#222222", hover_color="#333333", text_color=neon_blue
        )
        self.convert_btn.grid(row=0, column=0, padx=20, pady=5)

        self.reverse_btn = ctk.CTkButton(
            self.button_frame, text="📤 Convert Parquet ➜ JSONL",
            command=self.revert_files, width=280, height=45,
            font=ctk.CTkFont(size=14),
            fg_color="#222222", hover_color="#333333", text_color=neon_blue
        )
        self.reverse_btn.grid(row=0, column=1, padx=20, pady=5)

        # Preview
        self.preview_label = ctk.CTkLabel(
            self,
            text="📝 Preview Panel",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w", justify="left",
            text_color="#CCCCCC"
        )
        self.preview_label.pack(pady=(15, 5), anchor="w", padx=25)

        self.preview_box = ctk.CTkTextbox(
            self, width=840, height=400,
            font=("Consolas", 11),
            corner_radius=8, wrap="none",
            fg_color="#111111", text_color="#CCCCCC", border_width=0
        )
        self.preview_box.pack(padx=25, pady=5)
        self.preview_box.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            self,
            text="💤 Waiting for action...",
            font=ctk.CTkFont(size=14),
            text_color="#888888"
        )
        self.status_label.pack(pady=(5, 5))

    def run_with_multiprocessing(self, files, worker_func):
        manager = Manager()
        queue = manager.Queue()
        processes = [Process(target=worker_func, args=(path, queue)) for path in files]
        for p in processes:
            p.start()

        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        results = []

        def monitor():
            while True:
                msg = queue.get()
                if isinstance(msg, tuple):
                    results.append(msg)
                    if len(results) == len(files):
                        break
            self.preview_box.configure(state="normal")
            for file_path, preview, error in results:
                filename = os.path.basename(file_path)
                if error:
                    self.preview_box.insert("end", f"❌ {filename} failed: {error}\n\n")
                else:
                    self.preview_box.insert("end", f"✅ {filename}\n{preview}\n\n")
            self.preview_box.configure(state="disabled")
            self.status_label.configure(text="🎉 All files processed!", text_color="#00FFAA")
            messagebox.showinfo("Results", f"Processed {len(files)} files.")

        Thread(target=monitor, daemon=True).start()

    def convert_files(self):
        files = filedialog.askopenfilenames(
            title="Select JSONL files",
            filetypes=[("JSONL files", "*.jsonl")]
        )
        if files:
            self.status_label.configure(text="⏳ Converting...", text_color="white")
            self.run_with_multiprocessing(files, jsonl_to_parquet_worker)

    def revert_files(self):
        files = filedialog.askopenfilenames(
            title="Select Parquet files",
            filetypes=[("Parquet files", "*.parquet")]
        )
        if files:
            self.status_label.configure(text="⏳ Reverting...", text_color="white")
            self.run_with_multiprocessing(files, parquet_to_jsonl_worker)


# ✅ Optional standalone test
if __name__ == "__main__":
    # Create a simple CTk root and launch a ParquetMaxxer window attached to it
    root = ctk.CTk()
    root.withdraw()  # hide the empty root window
    ParquetMaxxer(root)  # launch as a Toplevel
    root.mainloop()
