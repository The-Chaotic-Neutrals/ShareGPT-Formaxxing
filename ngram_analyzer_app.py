import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import time
from ngram_analyzer import process_jsonl, count_ngrams  # Import the necessary functions from ngram_analyzer.py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class NgramAnalyzerApp:
    def __init__(self, root, theme):
        self.root = root
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        # File selection frame
        self.file_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.file_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        self.input_file_label = tk.Label(self.file_frame, text="Select Input File:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.input_file_label.grid(row=0, column=0, sticky='w')

        self.input_file_entry = tk.Entry(self.file_frame, width=50, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.input_file_entry.grid(row=0, column=1, padx=5)

        self.input_file_button = tk.Button(self.file_frame, text="Browse", command=self.browse_file, bg=self.theme.get('button_bg', 'black'), fg=self.theme.get('button_fg', 'gold'))
        self.input_file_button.grid(row=0, column=2, padx=5)

        # Role filter frame
        self.role_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.role_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        self.role_label = tk.Label(self.role_frame, text="Role Filter:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.role_label.grid(row=0, column=0, sticky='w')

        self.role_var = tk.StringVar(value='all')
        self.role_dropdown = ttk.Combobox(self.role_frame, textvariable=self.role_var, values=['all', 'human', 'gpt', 'system'])
        self.role_dropdown.grid(row=0, column=1, padx=5)
        self.role_dropdown.set('all')  # Default value

        # N-gram lengths frame
        self.length_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.length_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')

        self.min_length_label = tk.Label(self.length_frame, text="Min Length:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.min_length_label.grid(row=0, column=0, sticky='w')

        self.min_length_entry = tk.Entry(self.length_frame, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.min_length_entry.grid(row=0, column=1, padx=5)
        self.min_length_entry.insert(0, '3')  # Default value

        self.max_length_label = tk.Label(self.length_frame, text="Max Length:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.max_length_label.grid(row=0, column=2, sticky='w')

        self.max_length_entry = tk.Entry(self.length_frame, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.max_length_entry.grid(row=0, column=3, padx=5)
        self.max_length_entry.insert(0, '5')  # Default value

        # Exclude Stop Words checkbox
        self.stopwords_var = tk.BooleanVar(value=False)
        self.stopwords_checkbox = tk.Checkbutton(self.root, text="Exclude Stop Words", variable=self.stopwords_var, bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'), selectcolor=self.theme.get('entry_bg', 'black'))
        self.stopwords_checkbox.grid(row=3, column=0, pady=5, sticky='w')

        # Exclude Numerical Tokens checkbox
        self.numerical_var = tk.BooleanVar(value=False)
        self.numerical_checkbox = tk.Checkbutton(self.root, text="Exclude Numerical Tokens", variable=self.numerical_var, bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'), selectcolor=self.theme.get('entry_bg', 'black'))
        self.numerical_checkbox.grid(row=4, column=0, pady=5, sticky='w')

        # Punctuation Filtering checkbox
        self.punctuation_var = tk.BooleanVar(value=False)
        self.punctuation_checkbox = tk.Checkbutton(self.root, text="Exclude Punctuation", variable=self.punctuation_var, bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'), selectcolor=self.theme.get('entry_bg', 'black'))
        self.punctuation_checkbox.grid(row=5, column=0, pady=5, sticky='w')

        # Punctuation token limit input
        self.punctuation_limit_label = tk.Label(self.root, text="Max Punctuation in N-grams:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.punctuation_limit_label.grid(row=6, column=0, sticky='w')

        self.punctuation_limit_entry = tk.Entry(self.root, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.punctuation_limit_entry.grid(row=6, column=1, padx=5)
        self.punctuation_limit_entry.insert(0, '1')  # Default value

        # Stop token limit input
        self.stop_token_label = tk.Label(self.root, text="Max Stop Tokens in N-grams:", bg=self.theme.get('bg', 'black'), fg=self.theme.get('fg', 'gold'))
        self.stop_token_label.grid(row=7, column=0, sticky='w')

        self.stop_token_entry = tk.Entry(self.root, width=5, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.stop_token_entry.grid(row=7, column=1, padx=5)
        self.stop_token_entry.insert(0, '1')  # Default value

        # Analyze and Clear buttons
        self.button_frame = tk.Frame(self.root, bg=self.theme.get('bg', 'black'))
        self.button_frame.grid(row=8, column=0, columnspan=2, pady=10, sticky='ew')

        self.analyze_button = tk.Button(self.button_frame, text="Analyze", command=self.start_analysis, bg=self.theme.get('button_bg', 'black'), fg=self.theme.get('button_fg', 'gold'))
        self.analyze_button.grid(row=0, column=0, padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_results, bg=self.theme.get('button_bg', 'black'), fg=self.theme.get('button_fg', 'gold'))
        self.clear_button.grid(row=0, column=1, padx=5)

        # Results area (text box)
        self.results_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, bg=self.theme.get('entry_bg', 'black'), fg=self.theme.get('entry_fg', 'gold'))
        self.results_area.grid(row=9, column=0, padx=10, pady=10, columnspan=2)

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

        # Validate numeric input for lengths
        try:
            min_length = int(self.min_length_entry.get())
            max_length = int(self.max_length_entry.get())
            if min_length < 1 or max_length < min_length:
                raise ValueError("Invalid length range.")
        except ValueError as e:
            self.root.after(0, self.show_error, str(e))
            return

        exclude_stopwords = self.stopwords_var.get()
        exclude_numerical = self.numerical_var.get()
        exclude_punctuation = self.punctuation_var.get()

        try:
            max_stop_tokens = int(self.stop_token_entry.get())
            if max_stop_tokens < 0:
                raise ValueError("Stop tokens must be a non-negative integer.")
        except ValueError as e:
            self.root.after(0, self.show_error, str(e))
            return

        try:
            punctuation_limit = int(self.punctuation_limit_entry.get())
            if punctuation_limit < 0:
                raise ValueError("Punctuation limit must be a non-negative integer.")
        except ValueError as e:
            self.root.after(0, self.show_error, str(e))
            return

        # Clear results area
        self.results_area.delete(1.0, tk.END)

        # Start analysis
        start_time = time.time()
        try:
            # Call ngram_analyzer.py functions with the additional flags for stopwords, numerical exclusion, and punctuation limit
            ngrams = count_ngrams(process_jsonl(input_filename, role_filter), 
                                  min_length, 
                                  max_length, 
                                  stopword_limit=max_stop_tokens if exclude_stopwords else 0,
                                  no_punctuation=exclude_punctuation,
                                  punctuation_limit=punctuation_limit)  # Pass the punctuation limit
            results = self.format_results(ngrams)
        except Exception as e:
            self.root.after(0, self.show_error, f"Error during processing: {str(e)}")
            return

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

            ax.spines['left'].set_color(self.theme.get('fg', 'gold'))
            ax.spines['right'].set_color(self.theme.get('fg', 'gold'))
            ax.spines['top'].set_color(self.theme.get('fg', 'gold'))

            # Customize tick labels color
            ax.tick_params(axis='both', colors=self.theme.get('fg', 'gold'))

            colors = plt.cm.get_cmap('tab20', len(ngrams))

            for i, n in enumerate(sorted(ngrams)):
                data = ngrams[n].most_common(10)
                words = [' '.join(gram) for gram, _ in data]
                counts = [count for _, count in data]
                ax.barh(words, counts, color=colors(i), label=f"{n}-grams")

            ax.set_xlabel('Frequency', color=self.theme.get('fg', 'gold'))
            ax.set_ylabel('N-grams', color=self.theme.get('fg', 'gold'))
            ax.set_title('Top N-grams Frequency', color=self.theme.get('fg', 'gold'))
            ax.legend()

            # Adjust padding/margins around the plot (adds space around the plot area)
            fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

            # Embed the plot in Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        create_plot()

    def show_error(self, message):
        """Display an error message in the results area.""" 
        self.results_area.insert(tk.END, f"ERROR: {message}\n")

    def clear_results(self):
        """Clear the results area.""" 
        self.results_area.delete(1.0, tk.END)

    def set_icon(self, window):
        """Set the icon for the window.""" 
        window.iconbitmap('icon.ico')

if __name__ == "__main__":
    root = tk.Tk()
    root.title("N-gram Analyzer")
    app = NgramAnalyzerApp(root, theme={'bg': 'black', 'fg': 'gold', 'entry_bg': 'black', 'entry_fg': 'gold', 'button_bg': 'black', 'button_fg': 'gold'})
    root.mainloop()
