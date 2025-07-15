import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from english_filter import filter_english_jsonl

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class EnglishFilterApp(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("EnglishFilter — FastText JSONL Cleaner")
        self.geometry("700x400")
        self.resizable(False, False)
        if os.path.exists("icon.ico"):
            try:
                self.iconbitmap("icon.ico")
            except Exception as e:
                print(f"Icon error: {e}")

        self.input_path = ""
        self.output_path = ""
        self.rejected_path = ""

        # UI Elements
        self.title_label = ctk.CTkLabel(self, text="🧹 EnglishFilter", font=ctk.CTkFont(size=28, weight="bold"))
        self.title_label.pack(pady=(20, 10))

        self.browse_btn = ctk.CTkButton(self, text="📂 Select Input JSONL", command=self.select_input)
        self.browse_btn.pack(pady=10)

        self.output_btn = ctk.CTkButton(self, text="💾 Select Output Location", command=self.select_output)
        self.output_btn.pack(pady=10)

        self.rejected_btn = ctk.CTkButton(self, text="🗑️ Select Rejected File (Optional)", command=self.select_rejected)
        self.rejected_btn.pack(pady=10)

        self.threshold_label = ctk.CTkLabel(self, text="Threshold (default 0.69):")
        self.threshold_label.pack(pady=(15, 5))
        self.threshold_entry = ctk.CTkEntry(self, width=200)
        self.threshold_entry.insert(0, "0.69")
        self.threshold_entry.pack()

        self.start_btn = ctk.CTkButton(self, text="🚀 Start Filtering", command=self.start_filter)
        self.start_btn.pack(pady=(20, 10))

        self.status_label = ctk.CTkLabel(self, text="💤 Waiting...", font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=(5, 10))

    def select_input(self):
        path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if path:
            self.input_path = path
            self.status_label.configure(text=f"📥 Input: {os.path.basename(path)}")

    def select_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSONL files", "*.jsonl")])
        if path:
            self.output_path = path
            self.status_label.configure(text=f"📤 Output: {os.path.basename(path)}")

    def select_rejected(self):
        path = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSONL files", "*.jsonl")])
        if path:
            self.rejected_path = path
            self.status_label.configure(text=f"🗑️ Rejected: {os.path.basename(path)}")

    def start_filter(self):
        if not self.input_path or not self.output_path:
            messagebox.showerror("Error", "Please select input and output files first.")
            return
        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Threshold must be a number.")
            return

        self.status_label.configure(text="⏳ Filtering in progress...")
        self.update_idletasks()
        stats = filter_english_jsonl(
            input_path=self.input_path,
            output_path=self.output_path,
            rejected_path=self.rejected_path if self.rejected_path else None,
            threshold=threshold,
            batch_size=256,
            workers=None
        )
        messagebox.showinfo("Done", f"✅ English Filter Complete!\n"
                                    f"Total: {stats['total_lines']}\n"
                                    f"Kept: {stats['english_total']}\n"
                                    f"Removed: {stats['non_english_total']}\n"
                                    f"Errors: {stats['json_error_total']}")
        self.status_label.configure(text="🎉 Done!")

# Optional standalone
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    app = EnglishFilterApp(master=root)
    app.mainloop()
