import tkinter as tk
from tkinter import filedialog, scrolledtext
from threading import Thread
import language_tool_python
import logging
from grammar_maxxer import GrammarMaxxer  # Updated import to GrammarMaxxer

class ToggleSwitch(tk.Canvas):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, width=60, height=30, bg='white', *args, **kwargs)
        self.parent = parent
        self.create_rectangle(0, 0, 60, 30, fill='lightgrey', outline='black', tags='bg')
        self.switch = self.create_oval(5, 5, 25, 25, fill='white', outline='black', tags='switch')
        self.state = tk.StringVar(value='off')
        self.bind("<Button-1>", self.toggle)
        self.update_switch()

    def toggle(self, event):
        if self.state.get() == 'off':
            self.state.set('on')
        else:
            self.state.set('off')
        self.update_switch()

    def update_switch(self):
        if self.state.get() == 'on':
            self.coords('switch', 30, 5, 50, 25)
            self.itemconfig('bg', fill='lightgreen')
        else:
            self.coords('switch', 5, 5, 25, 25)
            self.itemconfig('bg', fill='lightgrey')

class GrammarMaxxerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme

        self.toggles = {
            'grammar': tk.StringVar(value="on"),
        }

        self.setup_ui()
        self.configure_logging()

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("app.log"),
                                logging.StreamHandler()
                            ])

    def setup_ui(self):
        """Set up the user interface for the GrammarMaxxer app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("GrammarMaxxer App")
        self.window.configure(bg=self.theme.get('bg', 'white'))
        self.window.iconbitmap('icon.ico')

        main_frame = tk.Frame(self.window, bg=self.theme.get('bg', 'white'))
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.file_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.file_frame.pack(pady=5)

        self.input_file_label = tk.Label(self.file_frame, text="Select Input File:", bg=self.theme.get('bg', 'white'), fg='gold')
        self.input_file_label.pack(side=tk.LEFT, padx=5)

        self.input_file_entry = tk.Entry(self.file_frame, width=50, fg='gold', bg=self.theme.get('entry_bg', 'white'))
        self.input_file_entry.pack(side=tk.LEFT, padx=5)

        self.input_file_button = tk.Button(self.file_frame, text="Browse", command=self.browse_input_file, bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.input_file_button.pack(side=tk.LEFT, padx=5)

        self.correct_button = tk.Button(main_frame, text="Correct Text", command=self.start_text_correction, bg=self.theme.get('button_bg', 'lightgrey'), fg='gold')
        self.correct_button.pack(pady=10)

        self.toggles_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.toggles_frame.pack(pady=5)

        # Removed the hardcoded corrections toggle creation
        self.create_toggle("Grammar Correction", 'grammar')

        self.status_bar = tk.Label(main_frame, text="Status: Ready", bg='lightgrey', fg='black', anchor='w')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.corrections_frame = tk.Frame(main_frame, bg=self.theme.get('bg', 'white'))
        self.corrections_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        self.corrections_label = tk.Label(self.corrections_frame, text="Corrections:", bg=self.theme.get('bg', 'white'), fg='gold')
        self.corrections_label.pack(anchor='w')

        self.corrections_text = scrolledtext.ScrolledText(self.corrections_frame, wrap=tk.WORD, height=15, width=80, bg=self.theme.get('entry_bg', 'white'), fg='gold')
        self.corrections_text.pack(fill=tk.BOTH, expand=True)

    def create_toggle(self, text, key):
        """Create a custom toggle switch for a given option."""
        frame = tk.Frame(self.toggles_frame, bg=self.theme.get('bg', 'white'))
        frame.pack(anchor='w', pady=2)

        label = tk.Label(frame, text=text, bg=self.theme.get('bg', 'white'), fg='gold')
        label.pack(side=tk.LEFT, padx=5)

        toggle = ToggleSwitch(frame)
        toggle.pack(side=tk.LEFT, padx=5)

        # Bind the toggle state to the application state
        def on_toggle(event):
            if toggle.state.get() == 'on':
                self.toggles[key].set('on')
            else:
                self.toggles[key].set('off')

        toggle.bind("<ButtonRelease-1>", on_toggle)

    def update_status(self, message):
        """Update the status bar with a message."""
        if self.root:
            self.root.after(0, lambda: self.status_bar.config(text=f"Status: {message}"))

    def browse_input_file(self):
        """Open a file dialog to select the input JSONL file."""
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def start_text_correction(self):
        """Start text correction in a separate thread to keep the UI responsive."""
        thread = Thread(target=self.correct_text_file)
        thread.start()

    def correct_text_file(self):
        """Correct the text in the input file and write the corrected text to an output file."""
        grammar_maxxer = GrammarMaxxer(self.input_file_entry.get(), self.toggles)  # Create an instance of GrammarMaxxer
        if not grammar_maxxer.validate_file():
            return

        output_file = grammar_maxxer.prepare_output_file()
        self.update_status("Text correction started...")
        self.corrections_text.delete(1.0, tk.END)

        try:
            grammar_maxxer.process_file(output_file, self.update_corrections)
            self.update_status(f"Text correction complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
            self.update_status(f"Error processing file: {e}")

    def update_corrections(self, original_text, corrected_text):
        """Update the corrections tracker with the original and corrected text."""
        correction_entry = f"Original: {original_text}\nCorrected: {corrected_text}\n\n"
        self.root.after(0, lambda: self.corrections_text.insert(tk.END, correction_entry))
        self.root.after(0, lambda: self.corrections_text.yview(tk.END))

def main():
    root = tk.Tk()
    root.title("GrammarMaxxer Tool")
    app = GrammarMaxxerApp(root, theme={
        'bg': 'black',
        'entry_bg': 'lightgrey',
        'button_bg': 'grey'
    })
    root.mainloop()

if __name__ == "__main__":
    main()
