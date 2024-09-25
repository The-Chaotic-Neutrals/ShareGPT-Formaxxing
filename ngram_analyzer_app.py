import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import collections
import re
import json
import time
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NgramAnalyzerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        # File selection
        self.file_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.file_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        self.input_file_label = tk.Label(self.file_frame, text="Select Input File:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.input_file_label.pack(side=tk.LEFT)

        self.input_file_entry = tk.Entry(self.file_frame, width=50, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.input_file_entry.pack(side=tk.LEFT, padx=5)

        self.input_file_button = tk.Button(self.file_frame, text="Browse", command=self.browse_file, bg=self.theme.get('button_bg', 'black'), fg=self.theme.get('button_fg', 'gold'))
        self.input_file_button.pack(side=tk.LEFT, padx=5)

        # Role filter
        self.role_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.role_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        self.role_label = tk.Label(self.role_frame, text="Role Filter:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.role_label.pack(side=tk.LEFT)

        self.role_var = tk.StringVar(value='all')
        self.role_dropdown = ttk.Combobox(self.role_frame, textvariable=self.role_var, values=['all', 'human', 'gpt', 'system'])
        self.role_dropdown.pack(side=tk.LEFT, padx=5)
        self.role_dropdown.set('all')  # Default value

        # N-gram lengths
        self.length_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.length_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')

        self.min_length_label = tk.Label(self.length_frame, text="Min Length:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.min_length_label.pack(side=tk.LEFT)

        self.min_length_entry = tk.Entry(self.length_frame, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.min_length_entry.pack(side=tk.LEFT, padx=5)
        self.min_length_entry.insert(0, '3')  # Default value

        self.max_length_label = tk.Label(self.length_frame, text="Max Length:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.max_length_label.pack(side=tk.LEFT)

        self.max_length_entry = tk.Entry(self.length_frame, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.max_length_entry.pack(side=tk.LEFT, padx=5)
        self.max_length_entry.insert(0, '5')  # Default value

        # Analyze button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.start_analysis, bg=self.theme.get('button_bg', 'black'), fg=self.theme.get('button_fg', 'gold'))
        self.analyze_button.grid(row=3, column=0, pady=10)

        # Results area
        self.results_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.results_area.grid(row=4, column=0, padx=10, pady=10)

        # Set the icon for the root window
        self.set_icon(self.root)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if file_path:
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)

    def start_analysis(self):
        # Start analysis in a separate thread to avoid blocking the UI
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        # Get parameters
        input_filename = self.input_file_entry.get()
        selected_role = self.role_var.get()
        role_filter = ['human', 'gpt', 'system'] if selected_role == 'all' else [selected_role]
        min_length = int(self.min_length_entry.get())
        max_length = int(self.max_length_entry.get())

        # Clear results area
        self.results_area.delete(1.0, tk.END)

        # Start analysis
        start_time = time.time()
        ngrams = count_ngrams(process_jsonl(input_filename, role_filter), min_length, max_length)
        results = self.format_results(ngrams)
        elapsed_time = time.time() - start_time

        # Display results
        self.root.after(0, self.results_area.insert, tk.END, results)
        self.root.after(0, self.results_area.insert, tk.END, f'Took {elapsed_time:.03f} seconds\n')

        # Plot the results
        self.root.after(0, self.plot_results, ngrams)

    def format_results(self, ngrams):
        """Format the n-grams results for display."""
        results = []
        for n in sorted(ngrams):
            results.append(f'----- {10} most common {n}-grams -----\n')
            for gram, count in ngrams[n].most_common(10):
                results.append(f'{" ".join(gram)}: {count}\n')
            results.append('\n')
        return ''.join(results)

    def plot_results(self, ngrams):
        """Plot the n-gram frequencies."""
        def create_plot():
            """Create and embed the plot in the Tkinter window."""
            # Create a new window for the plot
            plot_window = tk.Toplevel(self.root)
            plot_window.title("N-gram Frequencies")

            # Set the icon for the plot window
            self.set_icon(plot_window)

            # Create a figure and axis for plotting
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(self.theme.get('bg', 'black'))
            ax.set_facecolor(self.theme.get('bg', 'black'))
            ax.spines['bottom'].set_color(self.theme.get('fg', 'gold'))
            ax.spines['top'].set_color(self.theme.get('fg', 'gold'))
            ax.spines['right'].set_color(self.theme.get('fg', 'gold'))
            ax.spines['left'].set_color(self.theme.get('fg', 'gold'))

            colors = plt.cm.get_cmap('tab20', len(ngrams))

            for i, n in enumerate(sorted(ngrams)):
                ngram_counts = ngrams[n]
                most_common = ngram_counts.most_common(10)
                grams, counts = zip(*most_common) if most_common else ([], [])

                ax.barh([' '.join(gram) for gram in grams], counts, label=f'{n}-grams', color=colors(i))

            ax.set_xlabel('Frequency', color=self.theme.get('fg', 'gold'))
            ax.set_title('Top 10 Most Common N-grams', color=self.theme.get('fg', 'gold'))
            ax.legend()

            # Adjust margins to avoid clipping
            fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.1)

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()

            # Use grid layout to manage space properly
            canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

            # Make sure the window resizes with the content
            plot_window.grid_rowconfigure(0, weight=1)
            plot_window.grid_columnconfigure(0, weight=1)

            # Optionally, add a scrollbar if the plot is too large
            canvas.get_tk_widget().update_idletasks()  # Update widget dimensions
            plot_window.geometry(f"{canvas.get_tk_widget().winfo_width()}x{canvas.get_tk_widget().winfo_height()}")  # Set window size to match canvas

            # Adjust the plot size dynamically if needed
            plot_window.minsize(800, 600)  # Set minimum size for the plot window

        # Schedule the plot creation to run in the main thread
        self.root.after(0, create_plot)

    def set_icon(self, window):
        """Set the icon for the given window."""
        try:
            window.iconbitmap('icon.ico')
        except Exception as e:
            print(f"Icon could not be set: {e}")

def tokenize(string):
    return re.findall(r'\b\S+\b', string.lower())

def count_ngrams(lines, min_length=3, max_length=5):
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def process_jsonl(filename, role_filter):
    with open(filename, 'r', encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

        with tqdm(total=total_lines, desc="Processing JSONL", unit="line") as pbar:
            for line in f:
                json_obj = json.loads(line)
                for conversation in json_obj.get("conversations", []):
                    if "value" in conversation and (role_filter == ['all'] or conversation["from"] in role_filter):
                        yield conversation["value"]
                pbar.update(1)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    app = NgramAnalyzerApp(root, theme={'bg': 'black', 'fg': 'gold', 'entry_bg': 'black', 'entry_fg': 'gold', 'button_bg': 'black', 'button_fg': 'gold'})
    root.mainloop()
