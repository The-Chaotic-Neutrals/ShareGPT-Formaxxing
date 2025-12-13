import sys
import os
import json
import yaml
import random
import subprocess
import threading
import time
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton,
                             QProgressBar, QFrame, QFileDialog, QMessageBox, QMainWindow)
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QPointF, pyqtSignal, QThread
from PyQt5.QtGui import QPainter

# === Helper functions ===
def detect_cuda():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def run_command(cmd, output_callback, progress_callback=None):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    for line in iter(process.stdout.readline, ''):
        output_callback(line)
    process.stdout.close()
    process.wait()
    if progress_callback:
        progress_callback()

def clean_index_json(index_path, output_callback):
    import base64
    metadata_key = base64.b64decode(b'bWV0YWRhdGE=').decode('utf-8')
    mergekit_version_key = base64.b64decode(b'bWVyZ2VraXRfdmVyc2lvbg==').decode('utf-8')
    if not os.path.exists(index_path):
        output_callback("[INFO] No index.json found to clean.\n")
        return
    try:
        with open(index_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if metadata_key in data and mergekit_version_key in data[metadata_key]:
            del data[metadata_key][mergekit_version_key]
        with open(index_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        output_callback("[INFO] Cleaned model.safetensors.index.json in place.\n")
    except Exception as e:
        output_callback(f"[ERROR] Failed to clean index: {str(e)}\n")

class MergeThread(QThread):
    output_signal = pyqtSignal(str)
    progress_stop_signal = pyqtSignal()
    def __init__(self, config_data, output_dir):
        super().__init__()
        self.config_data = config_data
        self.output_dir = output_dir
    def run(self):
        internal_config_path = "internal_config.yaml"
        try:
            with open(internal_config_path, "w", encoding="utf-8") as f:
                f.write(self.config_data)
            venv_dir = "venv"
            python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
            pip_bin = os.path.join(venv_dir, "Scripts", "pip.exe")
            mergekit_cmd = os.path.join(venv_dir, "Scripts", "mergekit-yaml.exe")
            if not os.path.exists(venv_dir) or not os.path.exists(os.path.join(venv_dir, "Scripts", "activate.bat")):
                self.output_signal.emit("[INFO] Creating virtual environment...\n")
                subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            if not os.path.exists(mergekit_cmd):
                self.output_signal.emit("[INFO] Required dependencies missing. Installing...\n")
                torch_index = "https://download.pytorch.org/whl/cu121" if detect_cuda() else "https://download.pytorch.org/whl/cpu"
                self.output_signal.emit(f"[INFO] Installing Torch from {torch_index}\n")
                subprocess.run([pip_bin, "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                run_command(f"{pip_bin} install --no-cache-dir --upgrade torch torchvision torchaudio --index-url {torch_index}",
                            lambda line: self.output_signal.emit(line))
                self.output_signal.emit("[INFO] Installing mergekit (if not present)...\n")
                subprocess.run([pip_bin, "install", "mergekit"], check=True)
            else:
                self.output_signal.emit("[INFO] Using existing venv and dependencies.\n")
            os.makedirs(self.output_dir, exist_ok=True)
            self.output_signal.emit(f"[INFO] Running MergeKit...\n")
            run_command(f"{mergekit_cmd} {internal_config_path} {self.output_dir} --lazy-unpickle --allow-crimes --lora-merge-cache ./cache",
                        lambda line: self.output_signal.emit(line))
            self.output_signal.emit("\n[✅] Merge Completed!\n")
            index_path = os.path.join(self.output_dir, "model.safetensors.index.json")
            clean_index_json(index_path, lambda msg: self.output_signal.emit(msg))
            self.progress_stop_signal.emit()
        except Exception as e:
            self.output_signal.emit(f"[ERROR] Merge failed: {str(e)}\n")
            self.progress_stop_signal.emit()
        finally:
            if os.path.exists(internal_config_path):
                os.remove(internal_config_path)

# === GUI Setup ===
class MergeKitGUI(QMainWindow):
    def __init__(self, theme=None):
        super().__init__()
        # Get theme or use default
        if theme is None:
            from App.Other.Theme import Theme
            theme = Theme.DARK
        self.theme = theme
        
        # Get repo root for output path
        repo_root = Path(__file__).parent.parent.parent
        self.default_output_dir = repo_root / "Outputs" / "models"
        
        self.setWindowTitle("Chaotic Neutral's MergeKit GUI")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(600, 400)
        
        # Set up central widget (background will be added by UI manager)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.content_frame = QFrame(central_widget)
        self.apply_theme()
        layout = QVBoxLayout(self.content_frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(12)
        
        # Default YAML without output_dir (will be set dynamically)
        self.default_yaml = """
merge_method: linear
models:
  - model: model1
    weight: 0.5
  - model: model2
    weight: 0.5
""".strip()
        
        yaml_label = QLabel("Config Editor:")
        yaml_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(yaml_label)
        
        self.yaml_editor = QTextEdit()
        self.yaml_editor.setPlainText(self.default_yaml)
        self.yaml_editor.setFont(QFont("Consolas", 10))
        self.yaml_editor.setAcceptDrops(True)
        layout.addWidget(self.yaml_editor, stretch=2)
        
        import_label = QLabel("Import Config File (Optional):")
        import_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        layout.addWidget(import_label)
        
        self.import_button = QPushButton("Browse")
        self.import_button.clicked.connect(self.browse_file)
        layout.addWidget(self.import_button)
        
        self.cuda_label = QLabel("[Checking CUDA status...]")
        layout.addWidget(self.cuda_label)
        QTimer.singleShot(500, self.update_cuda_status)
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        
        run_button = QPushButton("Run Merge")
        run_button.clicked.connect(self.start_merge)
        layout.addWidget(run_button)
        
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setFont(QFont("Consolas", 10))
        layout.addWidget(self.output_box, stretch=2)
        
        self.content_frame.setLayout(layout)
        self.setup_layout()
    
    def apply_theme(self):
        """Apply theme colors to the GUI"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(self.theme.get('bg', '#05050F')))
        palette.setColor(QPalette.WindowText, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Base, QColor(self.theme.get('entry_bg', '#05050F')))
        palette.setColor(QPalette.AlternateBase, QColor(self.theme.get('entry_bg', '#05050F')))
        palette.setColor(QPalette.Text, QColor(self.theme.get('text_fg', '#ffffff')))
        palette.setColor(QPalette.Button, QColor(self.theme.get('button_bg', '#1e90ff')))
        palette.setColor(QPalette.ButtonText, QColor(self.theme.get('button_fg', '#ffffff')))
        palette.setColor(QPalette.Highlight, QColor(self.theme.get('fg', '#1e90ff')))
        palette.setColor(QPalette.HighlightedText, QColor(self.theme.get('text_fg', '#ffffff')))
        self.setPalette(palette)
        
        # Get theme colors
        bg = self.theme.get('bg', '#05050F')
        text_bg = self.theme.get('text_bg', 'rgba(2, 2, 10, 220)')
        text_fg = self.theme.get('text_fg', '#ffffff')
        entry_bg = self.theme.get('entry_bg', 'rgba(5, 5, 15, 200)')
        entry_fg = self.theme.get('entry_fg', '#ffffff')
        button_bg = self.theme.get('button_bg', 'rgba(2, 6, 23, 200)')
        button_fg = self.theme.get('button_fg', '#ffffff')
        fg = self.theme.get('fg', '#1e90ff')
        
        stylesheet = f"""
            QMainWindow, QWidget {{
                background-color: transparent;
                color: {text_fg};
            }}
            QFrame#content_frame {{
                background: {text_bg};
                border-radius: 12px;
                border: 1px solid {fg};
            }}
            QLabel {{
                background: transparent;
                color: {text_fg};
            }}
            QTextEdit {{
                background: {entry_bg};
                color: {entry_fg};
                border: 1px solid {fg};
                border-radius: 8px;
                padding: 5px;
            }}
            QTextEdit QScrollBar:vertical {{
                border: none;
                background: {entry_bg};
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QTextEdit QScrollBar::handle:vertical {{
                background: {button_bg};
                border-radius: 6px;
                min-height: 20px;
            }}
            QTextEdit QScrollBar::add-line:vertical,
            QTextEdit QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}
            QTextEdit QScrollBar::add-page:vertical,
            QTextEdit QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QPushButton {{
                background: {button_bg};
                color: {button_fg};
                border: 1px solid {fg};
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {fg};
                color: {button_fg};
                border: 1px solid {text_fg};
            }}
            QPushButton:pressed {{
                background: {bg};
            }}
            QProgressBar {{
                background: {entry_bg};
                color: {text_fg};
                border: 1px solid {fg};
                border-radius: 8px;
                padding: 2px;
            }}
            QProgressBar::chunk {{
                background: {button_bg};
                border-radius: 8px;
            }}
        """
        self.setStyleSheet(stylesheet)
        self.content_frame.setObjectName("content_frame")
    
    def setup_layout(self):
        """Setup the layout geometry"""
        central = self.centralWidget()
        if central:
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.content_frame)
            # Ensure content_frame is on top of any background widgets
            self.content_frame.raise_()
            # Make sure content_frame fills the central widget
            self.content_frame.setMinimumSize(central.size())
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.content_frame:
            self.content_frame.resize(self.size())
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_yaml_file(files[0])
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose YAML File", "", "YAML Files (*.yaml *.yml);;All Files (*.*)")
        if filename:
            self.load_yaml_file(filename)
    def load_yaml_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            yaml.safe_load(content)
            self.yaml_editor.setPlainText(content)
            QMessageBox.information(self, "Success", "YAML file imported successfully!")
        except yaml.YAMLError:
            QMessageBox.critical(self, "Error", "Invalid YAML file. Please check the file format.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    def update_cuda_status(self):
        has_cuda = detect_cuda()
        status = "🟦 NVIDIA GPU detected - CUDA enabled" if has_cuda else "❌ No GPU detected - CPU mode"
        self.cuda_label.setText(status)
    def start_merge(self):
        try:
            config_data = self.yaml_editor.toPlainText().strip()
            config_dict = yaml.safe_load(config_data)
            
            if config_dict is None:
                config_dict = {}
            
            # Ensure output directory is set to the default location
            if 'output_dir' not in config_dict or not config_dict.get('output_dir'):
                # Create a unique output directory name based on timestamp
                timestamp = int(time.time())
                output_subdir = f"merged_model_{timestamp}"
                output_dir = str(self.default_output_dir / output_subdir)
            else:
                # If output_dir is specified, use it but ensure it's an absolute path
                output_dir = config_dict['output_dir']
                if not os.path.isabs(output_dir):
                    output_dir = str(self.default_output_dir / output_dir)
            
            # Update config with absolute output path
            config_dict['output_dir'] = output_dir
            config_data = yaml.dump(config_dict, default_flow_style=False)
            
            self.output_box.clear()
            self.progress.setRange(0, 0)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            self.merge_thread = MergeThread(config_data, output_dir)
            self.merge_thread.output_signal.connect(self.update_output)
            self.merge_thread.progress_stop_signal.connect(self.stop_progress)
            self.merge_thread.start()
        except yaml.YAMLError as e:
            QMessageBox.critical(self, "Error", f"Invalid YAML content in editor. Please check the syntax.\n\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start merge: {str(e)}")
    def update_output(self, text):
        self.output_box.insertPlainText(text)
        self.output_box.moveCursor(QTextCursor.End)
    def stop_progress(self):
        self.progress.setRange(0, 1)

if __name__ == "__main__":
    from App.Other.Theme import Theme
    app = QApplication(sys.argv)
    window = MergeKitGUI(Theme.DARK)
    window.show()
    sys.exit(app.exec_())