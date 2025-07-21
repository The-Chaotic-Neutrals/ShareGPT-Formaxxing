import tkinter as tk
from tkinter import filedialog
from threading import Thread
from binary_classification import (
    initialize_models,
    update_device_preference,
    filter_conversations,
    set_filter_mode,
    FILTER_MODE_RP,
    FILTER_MODE_NORMAL
)

class BinaryClassificationApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.device = 'cuda'
        self.total_positive_count = 0
        self.total_negative_count = 0
        self.input_files = []

        self.gpu_var = tk.BooleanVar(value=True)  # ✅ Initialize before use

        initialize_models()
        self.setup_ui()
        self.configure_logging()
        self.update_device_preference()

    def configure_logging(self):
        import logging
        logging.basicConfig(level=logging.WARNING)

    def setup_ui(self):
        self.window = tk.Toplevel(self.root)
        self.window.title("RefusalMancer")
        self.window.configure(bg=self.theme.get('bg', 'white'))
        self.window.iconbitmap('icon.ico')

        main_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.file_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.file_frame.pack(pady=5)

        self.input_file_label = tk.Label(
            self.file_frame, text="Select Input Files:",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.input_file_label.pack(side=tk.LEFT, padx=5)

        self.input_file_entry = tk.Entry(self.file_frame, width=50)
        self.input_file_entry.pack(side=tk.LEFT, padx=5)

        self.input_file_button = tk.Button(
            self.file_frame, text="Browse", command=self.browse_input_file,
            bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')  # ✅ Gold text
        self.input_file_button.pack(side=tk.LEFT, padx=5)

        self.filter_button = tk.Button(
            main_frame, text="Filter Conversations", command=self.start_filtering,
            bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.filter_button.pack(pady=10)

        self.threshold_label = tk.Label(
            main_frame, text="Threshold (0.0 - 1.0):",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.threshold_label.pack(pady=5)

        self.threshold_entry = tk.Entry(main_frame, width=10)
        self.threshold_entry.insert(0, '0.75')
        self.threshold_entry.pack(pady=5)

        self.batch_size_label = tk.Label(
            main_frame, text="Batch Size:",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.batch_size_label.pack(pady=5)

        self.batch_size_entry = tk.Entry(main_frame, width=10)
        self.batch_size_entry.insert(0, '16')
        self.batch_size_entry.pack(pady=5)

        self.status_bar = tk.Label(
            main_frame, text="Status: Ready", bg='lightgrey', fg='black', anchor='w')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.count_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.count_frame.pack(pady=5)

        self.positive_count_label = tk.Label(
            self.count_frame, text="Positive Count: 0",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.positive_count_label.pack(side=tk.LEFT, padx=5)

        self.negative_count_label = tk.Label(
            self.count_frame, text="Negative Count: 0",
            bg=self.theme.get('bg', 'white'), fg=self.theme.get('fg', 'black'))
        self.negative_count_label.pack(side=tk.LEFT, padx=5)

        # ✅ Red/Green GPU Toggle Button
        self.gpu_enabled = True
        self.gpu_button = tk.Button(
            main_frame,
            text="GPU: ON",
            bg="green",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            relief="raised",
            bd=2,
            command=self.toggle_gpu
        )
        self.gpu_button.pack(pady=10)

        self.mode_var = tk.StringVar(value=FILTER_MODE_RP)
        self.mode_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.mode_frame.pack(pady=10)

        self.mode_label = tk.Label(
            self.mode_frame, text="Select Filter Mode:",
            bg=self.theme.get('bg', 'white'), fg='gold')
        self.mode_label.pack(side=tk.LEFT, padx=5)

        self.rp_mode_radio = tk.Radiobutton(
            self.mode_frame, text="RP Filter", variable=self.mode_var, value=FILTER_MODE_RP,
            command=self.update_filter_mode,
            bg=self.theme.get('bg', 'white'), fg='gold',
            indicatoron=0, relief='raised', bd=2,
            selectcolor='white', activebackground='white', activeforeground='black')
        self.rp_mode_radio.pack(side=tk.LEFT, padx=5)

        self.normal_mode_radio = tk.Radiobutton(
            self.mode_frame, text="Normal Filter", variable=self.mode_var, value=FILTER_MODE_NORMAL,
            command=self.update_filter_mode,
            bg=self.theme.get('bg', 'white'), fg='gold',
            indicatoron=0, relief='raised', bd=2,
            selectcolor='white', activebackground='white', activeforeground='black')
        self.normal_mode_radio.pack(side=tk.LEFT, padx=5)

        self.class_logic_label = tk.Label(
            main_frame,
            text=self.get_classification_logic_text(),
            bg=self.theme.get('bg', 'white'),
            fg='gold',
            font=('Arial', 9, 'italic')
        )
        self.class_logic_label.pack(pady=5)

        self.update_filter_radio_styles()

    def style_radio_button(self, button, selected):
        if selected:
            button.config(bg='white', fg='black', relief='raised', bd=2)
        else:
            button.config(bg=self.theme.get('bg', 'white'), fg='gold', relief='raised', bd=2)

    def update_filter_radio_styles(self):
        selected = self.mode_var.get()
        self.style_radio_button(self.rp_mode_radio, selected == FILTER_MODE_RP)
        self.style_radio_button(self.normal_mode_radio, selected == FILTER_MODE_NORMAL)

    def get_classification_logic_text(self):
        mode = self.mode_var.get()
        if mode == FILTER_MODE_RP:
            return "RP Filter: Class 0 = Refusal (positive), Class 1 = Safe"
        else:
            return "Normal Filter: Class 1 = Refusal (positive), Class 0 = Safe"

    def update_filter_mode(self):
        set_filter_mode(self.mode_var.get())
        self.update_filter_radio_styles()
        self.positive_count_label.config(text="Positive Count: 0")
        self.negative_count_label.config(text="Negative Count: 0")
        self.class_logic_label.config(text=self.get_classification_logic_text())
        self.status_bar.config(text="Status: Filter mode switched.")

    def toggle_gpu(self):
        self.gpu_enabled = not self.gpu_enabled
        new_text = "GPU: ON" if self.gpu_enabled else "GPU: OFF"
        new_bg = "green" if self.gpu_enabled else "red"
        self.gpu_button.config(text=new_text, bg=new_bg)
        self.gpu_var.set(self.gpu_enabled)
        self.update_device_preference()

    def update_device_preference(self):
        update_device_preference(self.gpu_var, self.status_bar)

    def browse_input_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("JSONL files", "*.jsonl")])
        if file_paths:
            self.input_files = list(file_paths)
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, ", ".join(self.input_files))

    def start_filtering(self):
        thread = Thread(target=self.filter_conversations)
        thread.start()

    def filter_conversations(self):
        for input_file in self.input_files:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, input_file)
            filter_conversations(
                input_file_entry=self.input_file_entry,
                threshold_entry=self.threshold_entry,
                batch_size_entry=self.batch_size_entry,
                status_bar=self.status_bar,
                positive_count_label=self.positive_count_label,
                negative_count_label=self.negative_count_label
            )

    def update_status(self, message):
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def update_counts(self, positive_count, negative_count):
        if self.root:
            self.root.after(0, lambda: self.positive_count_label.config(text=f"Positive Count: {positive_count}"))
            self.root.after(0, lambda: self.negative_count_label.config(text=f"Negative Count: {negative_count}"))

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = BinaryClassificationApp(root, theme={
        'bg': 'white',
        'fg': 'black',
        'button_bg': 'lightgrey'
    })
    root.mainloop()
