import tkinter as tk
from tkinter import filedialog, messagebox
import os
from linemancer import LineMancerCore

class LineMancerFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg="#2e2e2e", padx=10, pady=10)
        self.core = LineMancerCore()

        # Color theme
        self.bg_color = "#2e2e2e"
        self.fg_color = "#d4af37"
        self.entry_bg = "#3b3b3b"
        self.entry_fg = "#d4af37"
        self.button_bg = "#3b3b3b"
        self.button_fg = "#d4af37"

        # Variables
        self.mode = tk.StringVar(value="split")
        self.input_path = tk.StringVar()
        self.input_paths = []
        self.output_dir = tk.StringVar()
        self.output_prefix = tk.StringVar(value="output_part")
        self.lines_per_file = tk.IntVar(value=1000)
        self.output_filename = tk.StringVar(value="merged_output.jsonl")

        self.setup_ui()

    def setup_ui(self):
        # Title bar within frame
        top_frame = tk.Frame(self, bg=self.bg_color)
        top_frame.pack(pady=8, fill=tk.X)

        title = tk.Label(top_frame,
                         text="✨ LineMancer — JSONL Tool ✨",
                         font=("Segoe UI", 14, "bold"),
                         bg=self.bg_color,
                         fg=self.fg_color)
        title.pack(side=tk.LEFT, padx=(10, 20))

        # Mode selector
        mode_frame = tk.Frame(self, bg=self.bg_color)
        mode_frame.pack(pady=8, fill=tk.X)
        tk.Label(mode_frame, text="Mode:", bg=self.bg_color, fg=self.fg_color,
                 font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(10, 5))
        for mode, label in [("split", "🪓 Split"), ("merge", "🧵 Merge"), ("shuffle", "🎲 Shuffle")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self.mode, value=mode,
                bg=self.bg_color, fg=self.fg_color, selectcolor=self.entry_bg,
                activebackground=self.entry_bg, activeforeground=self.fg_color,
                font=("Segoe UI", 10),
                command=self.render_mode
            ).pack(side=tk.LEFT, padx=8)

        # Main area
        self.main_frame = tk.Frame(self, bg=self.bg_color, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.render_mode()

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def render_mode(self):
        self.clear_main_frame()
        mode = self.mode.get()
        if mode == "split":
            self.build_split_ui()
        elif mode == "merge":
            self.build_merge_ui()
        elif mode == "shuffle":
            self.build_shuffle_ui()

    def build_split_ui(self):
        self._labeled_entry("Input JSONL File:", self.input_path, self.browse_input)
        self._labeled_entry("Output Directory:", self.output_dir, self.browse_output_dir)
        self._labeled_entry("Output File Prefix:", self.output_prefix)
        self._labeled_entry("Lines per file:", self.lines_per_file, is_int=True)
        self._action_button("Split JSONL", self.split_jsonl)

    def build_merge_ui(self):
        lbl = tk.Label(self.main_frame, text="Input JSONL Files:", bg=self.bg_color, fg=self.fg_color,
                       font=("Segoe UI", 10, "bold"))
        lbl.pack(anchor="w", padx=10, pady=(10, 2))
        self.input_label = tk.Label(self.main_frame, text="No files selected",
                                    bg=self.bg_color, fg=self.fg_color,
                                    wraplength=500, justify="left",
                                    font=("Segoe UI", 9, "italic"))
        self.input_label.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Button(self.main_frame, text="Browse Files...", command=self.browse_inputs,
                  bg=self.button_bg, fg=self.button_fg, font=("Segoe UI", 10, "bold")).pack(pady=5)
        self._labeled_entry("Output Directory:", self.output_dir, self.browse_output_dir)
        self._labeled_entry("Output File Name:", self.output_filename)
        self._action_button("Merge JSONL Files", self.merge_jsonl)

    def build_shuffle_ui(self):
        self._labeled_entry("Input JSONL File:", self.input_path, self.browse_input)
        self._labeled_entry("Output Directory:", self.output_dir, self.browse_output_dir)
        self._action_button("Shuffle JSONL Lines", self.shuffle_jsonl)

    def _labeled_entry(self, label, var, browse_command=None, is_int=False):
        frame = tk.Frame(self.main_frame, bg=self.bg_color)
        frame.pack(fill=tk.X, pady=6)
        tk.Label(frame, text=label, bg=self.bg_color, fg=self.fg_color,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=var, bg=self.entry_bg, fg=self.entry_fg,
                         insertbackground=self.entry_fg, width=40,
                         font=("Consolas", 10))
        entry.pack(side=tk.LEFT, padx=6)
        if is_int:
            def validate_int(*args):
                val = var.get()
                if isinstance(val, str) and not val.isdigit():
                    var.set("1000")
            var.trace_add("write", validate_int)
        if browse_command:
            tk.Button(frame, text="Browse...", command=browse_command,
                      bg=self.button_bg, fg=self.button_fg,
                      font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=6)

    def _action_button(self, text, command):
        tk.Button(self.main_frame, text=text, command=command,
                  bg=self.button_bg, fg=self.button_fg,
                  font=("Segoe UI", 11, "bold"), padx=10, pady=5)\
            .pack(pady=15)

    # --- File browser actions ---
    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(path))
            base = os.path.splitext(os.path.basename(path))[0]
            self.output_prefix.set(base + "_part")

    def browse_inputs(self):
        paths = filedialog.askopenfilenames(filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")])
        if paths:
            self.input_paths = paths
            filenames = "\n".join(os.path.basename(p) for p in paths)
            self.input_label.config(text=filenames)

    def browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    # --- Actions ---
    def split_jsonl(self):
        try:
            count = self.core.split_jsonl(
                self.input_path.get(),
                self.output_dir.get(),
                self.output_prefix.get().strip(),
                self.lines_per_file.get()
            )
            messagebox.showinfo("Success", f"Split complete. {count} parts saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def merge_jsonl(self):
        try:
            output_path, total = self.core.merge_jsonl(
                self.input_paths,
                self.output_dir.get(),
                self.output_filename.get().strip()
            )
            messagebox.showinfo("Success", f"Merged {total} lines into:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def shuffle_jsonl(self):
        try:
            output_path = self.core.shuffle_jsonl(
                self.input_path.get(),
                self.output_dir.get()
            )
            messagebox.showinfo("Success", f"Shuffled lines saved to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# --- Standalone runner with icon support ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("LineMancer — JSONL Split/Merge/Shuffle")
    root.geometry("640x520")
    # Set icon if available
    icon_path = "icon.ico"
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not set icon: {e}")
    app = LineMancerFrame(root)
    app.pack(fill="both", expand=True)
    root.mainloop()
