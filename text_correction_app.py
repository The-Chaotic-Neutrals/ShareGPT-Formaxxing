import tkinter as tk
from tkinter import filedialog, scrolledtext
import jsonlines
import os
import logging
import re
import markdown2
import json
import language_tool_python
from threading import Thread

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

class TextCorrectionApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.tool = language_tool_python.LanguageTool('en-US')

        self.toggles = {
            'markdown': tk.StringVar(value="on"),
            'hardcoded': tk.StringVar(value="on"),
            'regex': tk.StringVar(value="on"),
            'spacing': tk.StringVar(value="on"),
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

    def load_json(self, file_path):
        """Load data from a JSON file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load JSON file {file_path}: {e}")
            raise

    def setup_ui(self):
        """Set up the user interface for the Text Correction app."""
        self.window = tk.Toplevel(self.root)
        self.window.title("Text Correction App")
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

        self.create_toggle("Markdown Conversion", 'markdown')
        self.create_toggle("Hardcoded Corrections", 'hardcoded')
        self.create_toggle("Regex-Based Corrections", 'regex')
        self.create_toggle("Spacing and Punctuation", 'spacing')
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
        input_file = self.input_file_entry.get()
        if not self.validate_file(input_file):
            return

        output_file = self.prepare_output_file(input_file)
        self.update_status("Text correction started...")
        self.corrections_text.delete(1.0, tk.END)

        try:
            self.process_file(input_file, output_file)
            self.update_status(f"Text correction complete. Output file: {output_file}")
        except Exception as e:
            logging.error(f"Error processing file: {e}", exc_info=True)
            self.update_status(f"Error processing file: {e}")

    def validate_file(self, file_path):
        """Validate the selected file."""
        if not file_path.endswith('.jsonl'):
            self.update_status("Invalid file type. Please select a .jsonl file.")
            return False
        return True

    def prepare_output_file(self, input_file):
        """Prepare the output file path."""
        output_dir = "corrected"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output_dir, f"{base_name}-corrected.jsonl")

    def process_file(self, input_file, output_file):
        """Process the input file and write the corrected text to the output file."""
        with jsonlines.open(input_file) as reader:
            with jsonlines.open(output_file, mode='w') as writer:
                for conversation in reader:
                    corrected_conversation = self.correct_conversation(conversation)
                    writer.write(corrected_conversation)

    def correct_conversation(self, conversation):
        """Correct the text in a conversation and update the live tracker."""
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                original_text = turn.get('value', '')
                corrected_text = self.correct_text(original_text)
                turn['value'] = corrected_text
                self.update_corrections(original_text, corrected_text)
        return conversation

    def correct_text(self, text):
        """Correct text using a multi-step process."""
        corrections = {
            'markdown': self.convert_markdown,
            'hardcoded': self.apply_hardcoded_corrections,
            'regex': self.apply_regex_corrections,
            'spacing': self.enforce_spacing,
            'grammar': self.correct_with_grammar
        }
        for key, func in corrections.items():
            if self.toggles[key].get() == 'on':
                text = func(text)
        return text.strip()

    def convert_markdown(self, text):
        """Convert markdown to plain text."""
        html = markdown2.markdown(text)
        return re.sub(r'<[^>]+>', '', html)

    def apply_hardcoded_corrections(self, text, corrections_file='common_spellings.txt'):
        """Apply corrections from a file to common spelling mistakes in the text."""
        corrections = {}
        try:
            with open(corrections_file, 'r') as file:
                for line in file:
                    wrong, right = line.strip().split(',')
                    corrections[wrong] = right
        except FileNotFoundError:
            logging.error(f"The file '{corrections_file}' was not found.")
            return text
        except ValueError:
            logging.error("The correction file format is incorrect. Each line should be 'wrong,correct'.")
            return text
        
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        return text

    def apply_regex_corrections(self, text):
        """Apply regex-based corrections."""
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Example regex for duplicate words
        return text

    def enforce_spacing(self, text):
        """Enforce proper spacing and punctuation."""
        text = re.sub(r'\s*\.\s*\.\s*', '...', text)  # Convert multiple dots into ellipses
        text = re.sub(r'\s+([?.!,"](?:\s|$))', r'\1', text)  # Fix spacing before punctuation
        text = re.sub(r'(\S)([,.!?;])(\S)', r'\1 \2 \3', text)  # Add space around punctuation
        return text

    def correct_with_grammar(self, text):
        """Correct grammar using LanguageTool."""
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text

    def update_corrections(self, original_text, corrected_text):
        """Update the corrections tracker with the original and corrected text."""
        correction_entry = f"Original: {original_text}\nCorrected: {corrected_text}\n\n"
        self.root.after(0, lambda: self.corrections_text.insert(tk.END, correction_entry))
        self.root.after(0, lambda: self.corrections_text.yview(tk.END))

def main():
    root = tk.Tk()
    root.title("Text Correction Tool")
    app = TextCorrectionApp(root, theme={
        'bg': 'black',
        'entry_bg': 'lightgrey',
        'button_bg': 'grey'
    })
    root.mainloop()

if __name__ == "__main__":
    main()
