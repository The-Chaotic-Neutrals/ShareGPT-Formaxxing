import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QTextEdit, QFileDialog, QDialog
from PyQt5.QtGui import QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import threading
import time
from ngram_analyzer import process_jsonl, count_ngrams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from collections import Counter


def format_results(ngrams):
    results = []
    for n in sorted(ngrams):
        results.append(f'----- 10 most common {n}-grams -----\n')
        for gram, count in ngrams[n].most_common(10):
            results.append(f'{" ".join(gram)}: {count}\n')
        results.append('\n')
    return ''.join(results)


class PlotDialog(QDialog):
    def __init__(self, parent, ngrams, theme):
        super().__init__(parent)
        self.setWindowTitle("N-gram Frequencies")
        self.setWindowIcon(QIcon('icon.ico'))
        layout = QVBoxLayout(self)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(theme.get('bg', 'black'))
        ax.set_facecolor(theme.get('bg', 'black'))

        for spine in ['left', 'right', 'bottom', 'top']:
            ax.spines[spine].set_color(theme.get('fg', 'white'))

        ax.tick_params(axis='both', colors=theme.get('fg', 'white'))

        for i, n in enumerate(sorted(ngrams)):
            data = ngrams[n].most_common(10)
            words = [' '.join(gram) for gram, _ in data]
            counts = [count for _, count in data]
            ax.barh(words, counts, color='#1e90ff', label=f"{n}-grams")

        ax.set_xlabel('Frequency', color=theme.get('fg', 'white'))
        ax.set_ylabel('N-grams', color=theme.get('fg', 'white'))
        ax.set_title('Top N-grams Frequency', color=theme.get('fg', 'white'))
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

class Worker(QThread):
    update_results = pyqtSignal(str)
    show_error = pyqtSignal(str)
    plot_results = pyqtSignal(dict)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        start_time = time.time()
        try:
            input_filename = self.params['input_filename']
            role_filter = self.params['role_filter']
            min_length = self.params['min_length']
            max_length = self.params['max_length']
            exclude_stopwords = self.params['exclude_stopwords']
            exclude_numerical = self.params['exclude_numerical']
            exclude_punctuation = self.params['exclude_punctuation']
            max_stop_tokens = self.params['max_stop_tokens']
            punctuation_limit = self.params['punctuation_limit']

            content = process_jsonl(input_filename, role_filter)
            ngrams = count_ngrams(content, min_length, max_length, 
                                 stopword_limit=max_stop_tokens if exclude_stopwords else 0, 
                                 no_punctuation=exclude_punctuation, 
                                 punctuation_limit=punctuation_limit)

            results = format_results(ngrams)
            elapsed_time = time.time() - start_time
            results += f'Took {elapsed_time:.03f} seconds\n'
            self.update_results.emit(results)
            self.plot_results.emit(ngrams)
        except Exception as e:
            self.show_error.emit(f"Error during processing: {str(e)}")

class NgramAnalyzerApp(QMainWindow):
    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.setWindowTitle("N-gram Analyzer")
        self.setWindowIcon(QIcon('icon.ico'))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_ui()

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.theme.get('bg', 'black')))
        palette.setColor(QPalette.WindowText, QColor(self.theme.get('fg', 'gold')))
        palette.setColor(QPalette.Base, QColor(self.theme.get('entry_bg', '#222222')))
        palette.setColor(QPalette.Text, QColor(self.theme.get('entry_fg', 'gold')))
        palette.setColor(QPalette.Button, QColor(self.theme.get('button_bg', '#333333')))
        palette.setColor(QPalette.ButtonText, QColor(self.theme.get('button_fg', 'gold')))
        self.setPalette(palette)

        self.setStyleSheet(f"""
            QLabel {{
                color: {self.theme.get('fg', 'gold')};
            }}
            QLineEdit, QComboBox, QTextEdit {{
                background-color: {self.theme.get('entry_bg', '#222222')};
                color: {self.theme.get('entry_fg', 'gold')};
                border: 1px solid {self.theme.get('fg', 'gold')};
                padding: 4px;
            }}
            QCheckBox {{
                color: {self.theme.get('fg', 'gold')};
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#333333')};
                color: {self.theme.get('button_fg', 'gold')};
                border: 1px solid {self.theme.get('fg', 'gold')};
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('fg', 'gold')};
                color: {self.theme.get('bg', 'black')};
            }}
        """)

    def setup_ui(self):
        file_layout = QHBoxLayout()
        self.input_file_label = QLabel("Select Input File:")
        file_layout.addWidget(self.input_file_label)
        self.input_file_entry = QLineEdit()
        file_layout.addWidget(self.input_file_entry)
        self.input_file_button = QPushButton("Browse")
        self.input_file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.input_file_button)
        self.main_layout.addLayout(file_layout)

        role_layout = QHBoxLayout()
        self.role_label = QLabel("Role Filter:")
        role_layout.addWidget(self.role_label)
        self.role_combo = QComboBox()
        self.role_combo.addItems(['all', 'human', 'gpt', 'system'])
        self.role_combo.setCurrentIndex(0)
        role_layout.addWidget(self.role_combo)
        self.main_layout.addLayout(role_layout)

        length_layout = QHBoxLayout()
        self.min_length_label = QLabel("Min Length:")
        length_layout.addWidget(self.min_length_label)
        self.min_length_entry = QLineEdit()
        self.min_length_entry.setText('3')
        self.min_length_entry.setMaximumWidth(50)
        length_layout.addWidget(self.min_length_entry)
        self.max_length_label = QLabel("Max Length:")
        length_layout.addWidget(self.max_length_label)
        self.max_length_entry = QLineEdit()
        self.max_length_entry.setText('5')
        self.max_length_entry.setMaximumWidth(50)
        length_layout.addWidget(self.max_length_entry)
        self.main_layout.addLayout(length_layout)

        self.stopwords_checkbox = QCheckBox("Exclude Stop Words")
        self.main_layout.addWidget(self.stopwords_checkbox)
        self.numerical_checkbox = QCheckBox("Exclude Numerical Tokens")
        self.main_layout.addWidget(self.numerical_checkbox)
        self.punctuation_checkbox = QCheckBox("Exclude Punctuation")
        self.main_layout.addWidget(self.punctuation_checkbox)

        punc_limit_layout = QHBoxLayout()
        self.punctuation_limit_label = QLabel("Max Punctuation in N-grams:")
        punc_limit_layout.addWidget(self.punctuation_limit_label)
        self.punctuation_limit_entry = QLineEdit()
        self.punctuation_limit_entry.setText('1')
        self.punctuation_limit_entry.setMaximumWidth(50)
        punc_limit_layout.addWidget(self.punctuation_limit_entry)
        self.main_layout.addLayout(punc_limit_layout)

        stop_limit_layout = QHBoxLayout()
        self.stop_token_label = QLabel("Max Stop Tokens in N-grams:")
        stop_limit_layout.addWidget(self.stop_token_label)
        self.stop_token_entry = QLineEdit()
        self.stop_token_entry.setText('1')
        self.stop_token_entry.setMaximumWidth(50)
        stop_limit_layout.addWidget(self.stop_token_entry)
        self.main_layout.addLayout(stop_limit_layout)

        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)
        self.main_layout.addLayout(button_layout)

        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.main_layout.addWidget(self.results_area)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "JSONL files (*.jsonl)")
        if file_path:
            self.input_file_entry.setText(file_path)

    def start_analysis(self):
        input_filename = self.input_file_entry.text()
        selected_role = self.role_combo.currentText()
        role_filter = ['human', 'gpt', 'system'] if selected_role == 'all' else [selected_role]

        try:
            min_length = int(self.min_length_entry.text())
            max_length = int(self.max_length_entry.text())
            if min_length < 1 or max_length < min_length:
                raise ValueError("Invalid length range.")
        except ValueError as e:
            self.show_error(str(e))
            return

        exclude_stopwords = self.stopwords_checkbox.isChecked()
        exclude_numerical = self.numerical_checkbox.isChecked()
        exclude_punctuation = self.punctuation_checkbox.isChecked()

        try:
            max_stop_tokens = int(self.stop_token_entry.text())
            if max_stop_tokens < 0:
                raise ValueError("Stop tokens must be a non-negative integer.")
        except ValueError as e:
            self.show_error(str(e))
            return

        try:
            punctuation_limit = int(self.punctuation_limit_entry.text())
            if punctuation_limit < 0:
                raise ValueError("Punctuation limit must be a non-negative integer.")
        except ValueError as e:
            self.show_error(str(e))
            return

        self.results_area.clear()

        params = {
            'input_filename': input_filename,
            'role_filter': role_filter,
            'min_length': min_length,
            'max_length': max_length,
            'exclude_stopwords': exclude_stopwords,
            'exclude_numerical': exclude_numerical,
            'exclude_punctuation': exclude_punctuation,
            'max_stop_tokens': max_stop_tokens,
            'punctuation_limit': punctuation_limit
        }

        self.worker = Worker(params, self)
        self.worker.update_results.connect(self.update_results_area)
        self.worker.show_error.connect(self.show_error)
        self.worker.plot_results.connect(self.plot_results)
        self.worker.start()

    def update_results_area(self, text):
        self.results_area.setText(text)

    def show_error(self, message):
        self.results_area.append(f"ERROR: {message}\n")

    def clear_results(self):
        self.results_area.clear()

    def plot_results(self, ngrams):
        dialog = PlotDialog(self, ngrams, self.theme)
        dialog.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NgramAnalyzerApp(theme={
        'bg': '#000000',
        'fg': '#ffffff',
        'entry_bg': '#111111',
        'entry_fg': '#ffffff',
        'button_bg': '#1e1e1e',
        'button_fg': '#1e90ff'
    })
    window.show()
    sys.exit(app.exec_())
