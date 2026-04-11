"""
SpaceMancer UI - ShareGPT spacing restoration tool.
"""

import json
import os
import sys
from pathlib import Path
import requests

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from App.SpaceMancer.SpaceMancer import SpaceMancer

try:
    import keyring
except Exception:
    keyring = None


KEYRING_SERVICE = "sharegpt-formaxxing"
KEYRING_ACCOUNT = "spacemancer_hf_token"


class SpaceMancerWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        source_mode,
        local_input,
        dataset_id,
        config_name,
        split,
        hf_token,
        output_name,
        write_audit,
        dry_run,
        preview_limit,
        model_profile,
        enable_llm_refine,
        llm_model_id,
        allowed_roles,
    ):
        super().__init__()
        self.source_mode = source_mode
        self.local_input = local_input
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.hf_token = hf_token
        self.output_name = output_name
        self.write_audit = write_audit
        self.dry_run = dry_run
        self.preview_limit = preview_limit
        self.model_profile = model_profile
        self.enable_llm_refine = enable_llm_refine
        self.llm_model_id = llm_model_id
        self.allowed_roles = allowed_roles

    def _engine_options(self):
        if self.model_profile == "rules_only":
            return {
                "enable_glued_token_split": False,
                "min_score_delta": 99.0,
                "min_token_len": 99,
            }
        if self.model_profile == "spell_aggressive":
            return {
                "enable_glued_token_split": True,
                "min_score_delta": 2.5,
                "min_token_len": 6,
            }
        return {
            "enable_glued_token_split": True,
            "min_score_delta": 8.0,
            "min_token_len": 9,
        }

    def run(self):
        try:
            llm_refiner = None
            if self.enable_llm_refine:
                if not self.hf_token:
                    raise ValueError("LLM refinement is enabled but no HF token was provided.")
                if not self.llm_model_id:
                    raise ValueError("LLM refinement is enabled but no model ID was provided.")
                llm_refiner = self._build_hf_llm_refiner(self.llm_model_id, self.hf_token)

            engine = SpaceMancer(
                log_callback=self.status.emit,
                allowed_roles=self.allowed_roles,
                llm_refiner=llm_refiner,
                **self._engine_options(),
            )
            if self.source_mode == "local":
                self.status.emit("Processing local ShareGPT file...")
                stats = engine.process_local_file(
                    input_path=self.local_input,
                    output_filename=self.output_name or None,
                    write_audit=self.write_audit,
                    dry_run=self.dry_run,
                    preview_limit=self.preview_limit,
                )
            else:
                self.status.emit("Processing Hugging Face dataset split...")
                stats = engine.process_hf_dataset(
                    dataset_id=self.dataset_id,
                    split=self.split,
                    config_name=self.config_name or None,
                    hf_token=self.hf_token or None,
                    output_filename=self.output_name or None,
                    write_audit=self.write_audit,
                    dry_run=self.dry_run,
                    preview_limit=self.preview_limit,
                )
            self.finished.emit(stats)
        except Exception as exc:
            self.error.emit(str(exc))

    def _build_hf_llm_refiner(self, model_id, hf_token):
        endpoint = "https://router.huggingface.co/v1/chat/completions"

        def refine(text):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You restore missing spaces in text. "
                        "Only adjust whitespace. Do not add, remove, or rewrite words. "
                        "Return only corrected text."
                    ),
                },
                {"role": "user", "content": text},
            ]
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0,
            }
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            return None

        return refine


class SpaceMancerApp(QWidget):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme or {}
        self.worker = None
        self.config_path = Path(__file__).with_name("spacemancer_config.json")
        self.setWindowTitle("SpaceMancer")
        self.resize(900, 700)
        self.apply_theme()
        self.setup_ui()
        self.set_icon()
        self.load_config()
        self.refresh_source_mode()
        self.refresh_keyring_state(initial=True)
        self._toggle_dry_run()

    def apply_theme(self):
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.theme.get('bg', '#000000')};
                color: {self.theme.get('text_fg', '#ffffff')};
            }}
            QLabel {{
                color: {self.theme.get('text_fg', '#ffffff')};
                font-size: 13px;
            }}
            QLineEdit, QTextEdit {{
                background-color: {self.theme.get('entry_bg', '#0b0b0b')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                padding: 6px;
            }}
            QComboBox, QSpinBox {{
                background-color: {self.theme.get('entry_bg', '#0b0b0b')};
                color: {self.theme.get('entry_fg', '#ffffff')};
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                padding: 4px;
                min-height: 24px;
            }}
            QPushButton {{
                background-color: {self.theme.get('button_bg', '#1e90ff')};
                color: {self.theme.get('button_fg', '#ffffff')};
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.theme.get('fg', '#1e90ff')};
                color: {self.theme.get('bg', '#000000')};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
            }}
            QGroupBox {{
                border: 1px solid {self.theme.get('fg', '#1e90ff')};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                left: 10px;
                padding: 0 4px;
            }}
            """
        )

    def set_icon(self):
        icon_path = Path(__file__).parent.parent / "Assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("SpaceMancer - Conservative spacing fixes for ShareGPT datasets")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_group)

        mode_row = QHBoxLayout()
        self.local_radio = QRadioButton("Local file")
        self.hf_radio = QRadioButton("Hugging Face dataset")
        self.local_radio.setChecked(True)
        self.local_radio.toggled.connect(self.refresh_source_mode)
        mode_row.addWidget(self.local_radio)
        mode_row.addWidget(self.hf_radio)
        mode_row.addStretch()
        source_layout.addLayout(mode_row)

        local_row = QHBoxLayout()
        self.local_path_edit = QLineEdit()
        self.local_path_edit.setPlaceholderText("Path to .json or .jsonl ShareGPT dataset")
        self.local_browse_btn = QPushButton("Browse")
        self.local_browse_btn.clicked.connect(self.browse_local_file)
        local_row.addWidget(QLabel("Local file:"))
        local_row.addWidget(self.local_path_edit)
        local_row.addWidget(self.local_browse_btn)
        source_layout.addLayout(local_row)

        self.hf_panel = QWidget()
        hf_grid = QGridLayout(self.hf_panel)
        hf_grid.setContentsMargins(0, 0, 0, 0)

        self.dataset_id_edit = QLineEdit()
        self.dataset_id_edit.setPlaceholderText("e.g. org/dataset")
        self.config_name_edit = QLineEdit()
        self.config_name_edit.setPlaceholderText("Optional dataset config/subset")
        self.split_edit = QLineEdit("train")

        hf_grid.addWidget(QLabel("Dataset ID:"), 0, 0)
        hf_grid.addWidget(self.dataset_id_edit, 0, 1)
        hf_grid.addWidget(QLabel("Config (optional):"), 1, 0)
        hf_grid.addWidget(self.config_name_edit, 1, 1)
        hf_grid.addWidget(QLabel("Split:"), 2, 0)
        hf_grid.addWidget(self.split_edit, 2, 1)

        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        self.hf_token_edit.setPlaceholderText("Optional Hugging Face token")
        self.toggle_token_btn = QPushButton("Show")
        self.toggle_token_btn.clicked.connect(self.toggle_token_visibility)
        self.save_token_btn = QPushButton("Save Token")
        self.save_token_btn.clicked.connect(self.save_token)
        self.load_token_btn = QPushButton("Load Token")
        self.load_token_btn.clicked.connect(self.load_token)
        self.clear_token_btn = QPushButton("Clear Token")
        self.clear_token_btn.clicked.connect(self.clear_token)

        token_row = QHBoxLayout()
        token_row.addWidget(self.hf_token_edit)
        token_row.addWidget(self.toggle_token_btn)
        token_row.addWidget(self.save_token_btn)
        token_row.addWidget(self.load_token_btn)
        token_row.addWidget(self.clear_token_btn)

        hf_grid.addWidget(QLabel("HF token:"), 3, 0)
        hf_grid.addLayout(token_row, 3, 1)
        self.keyring_status_label = QLabel("")
        hf_grid.addWidget(self.keyring_status_label, 4, 1)

        source_layout.addWidget(self.hf_panel)
        root.addWidget(source_group)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        out_row = QHBoxLayout()
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Optional output filename (without extension)")
        out_row.addWidget(QLabel("Output name:"))
        out_row.addWidget(self.output_name_edit)
        options_layout.addLayout(out_row)

        self.audit_checkbox = QCheckBox("Write audit JSONL with before/after changes")
        self.audit_checkbox.setChecked(True)
        options_layout.addWidget(self.audit_checkbox)

        dry_row = QHBoxLayout()
        self.dry_run_checkbox = QCheckBox("Dry run preview only (no output file written)")
        self.dry_run_checkbox.setChecked(False)
        self.dry_run_checkbox.stateChanged.connect(self._toggle_dry_run)
        self.preview_limit_spin = QSpinBox()
        self.preview_limit_spin.setRange(1, 200)
        self.preview_limit_spin.setValue(20)
        self.preview_limit_spin.setFixedWidth(80)
        dry_row.addWidget(self.dry_run_checkbox)
        dry_row.addWidget(QLabel("Preview limit:"))
        dry_row.addWidget(self.preview_limit_spin)
        dry_row.addStretch()
        options_layout.addLayout(dry_row)

        model_row = QHBoxLayout()
        self.model_profile_combo = QComboBox()
        self.model_profile_combo.addItem("Conservative splitter (recommended)", "spell_conservative")
        self.model_profile_combo.addItem("Rules only (no word splitter)", "rules_only")
        self.model_profile_combo.addItem("Aggressive splitter", "spell_aggressive")
        self.model_profile_combo.setCurrentIndex(0)
        model_row.addWidget(QLabel("Correction model:"))
        model_row.addWidget(self.model_profile_combo)
        model_row.addStretch()
        options_layout.addLayout(model_row)

        llm_group = QGroupBox("Optional LLM Refinement")
        llm_layout = QVBoxLayout(llm_group)
        self.enable_llm_checkbox = QCheckBox("Enable LLM spacing pass (HF Inference)")
        self.enable_llm_checkbox.setChecked(False)
        llm_layout.addWidget(self.enable_llm_checkbox)
        llm_row = QHBoxLayout()
        self.llm_model_id_edit = QLineEdit()
        self.llm_model_id_edit.setPlaceholderText("LLM model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
        llm_row.addWidget(QLabel("Model ID:"))
        llm_row.addWidget(self.llm_model_id_edit)
        llm_layout.addLayout(llm_row)
        llm_note = QLabel("Uses the HF token above. LLM output is accepted only if non-space characters are unchanged.")
        llm_note.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        llm_layout.addWidget(llm_note)
        options_layout.addWidget(llm_group)

        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("Process roles:"))
        self.role_human = QCheckBox("human/user")
        self.role_human.setChecked(True)
        self.role_gpt = QCheckBox("gpt/assistant")
        self.role_gpt.setChecked(True)
        self.role_system = QCheckBox("system")
        self.role_system.setChecked(False)
        role_row.addWidget(self.role_human)
        role_row.addWidget(self.role_gpt)
        role_row.addWidget(self.role_system)
        role_row.addStretch()
        options_layout.addLayout(role_row)

        note = QLabel(
            "Conservative mode: applies high-confidence spacing fixes and leaves ambiguous text unchanged."
        )
        note.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        options_layout.addWidget(note)

        root.addWidget(options_group)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run SpaceMancer")
        self.run_btn.clicked.connect(self.run_processing)
        run_row.addWidget(self.run_btn)
        run_row.addStretch()
        root.addLayout(run_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        root.addWidget(self.log_text, stretch=1)

        self.status_label = QLabel("Status: Ready")
        root.addWidget(self.status_label)

    def config_payload(self):
        return {
            "source_mode": "local" if self.local_radio.isChecked() else "hf",
            "local_path": self.local_path_edit.text().strip(),
            "dataset_id": self.dataset_id_edit.text().strip(),
            "config_name": self.config_name_edit.text().strip(),
            "split": self.split_edit.text().strip() or "train",
            "output_name": self.output_name_edit.text().strip(),
            "write_audit": self.audit_checkbox.isChecked(),
            "dry_run": self.dry_run_checkbox.isChecked(),
            "preview_limit": self.preview_limit_spin.value(),
            "model_profile": self.model_profile_combo.currentData(),
            "enable_llm_refine": self.enable_llm_checkbox.isChecked(),
            "llm_model_id": self.llm_model_id_edit.text().strip(),
            "role_human": self.role_human.isChecked(),
            "role_gpt": self.role_gpt.isChecked(),
            "role_system": self.role_system.isChecked(),
            "has_saved_hf_token": self.has_saved_token(),
        }

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as fh:
                json.dump(self.config_payload(), fh, indent=2)
        except Exception:
            pass

    def load_config(self):
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return

        source_mode = data.get("source_mode", "local")
        self.local_radio.setChecked(source_mode == "local")
        self.hf_radio.setChecked(source_mode == "hf")
        self.local_path_edit.setText(data.get("local_path", ""))
        self.dataset_id_edit.setText(data.get("dataset_id", ""))
        self.config_name_edit.setText(data.get("config_name", ""))
        self.split_edit.setText(data.get("split", "train"))
        self.output_name_edit.setText(data.get("output_name", ""))
        self.audit_checkbox.setChecked(bool(data.get("write_audit", True)))
        self.dry_run_checkbox.setChecked(bool(data.get("dry_run", False)))
        self.preview_limit_spin.setValue(int(data.get("preview_limit", 20)))
        model_profile = data.get("model_profile", "spell_conservative")
        index = self.model_profile_combo.findData(model_profile)
        if index >= 0:
            self.model_profile_combo.setCurrentIndex(index)
        self.enable_llm_checkbox.setChecked(bool(data.get("enable_llm_refine", False)))
        self.llm_model_id_edit.setText(data.get("llm_model_id", ""))
        self.role_human.setChecked(bool(data.get("role_human", True)))
        self.role_gpt.setChecked(bool(data.get("role_gpt", True)))
        self.role_system.setChecked(bool(data.get("role_system", False)))

    def closeEvent(self, event):
        self.save_config()
        super().closeEvent(event)

    def append_log(self, message):
        self.log_text.append(message)
        self.status_label.setText(f"Status: {message}")
        bar = self.log_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def browse_local_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select dataset", "", "JSON files (*.json *.jsonl)")
        if path:
            self.local_path_edit.setText(path)

    def refresh_source_mode(self):
        local = self.local_radio.isChecked()
        self.local_path_edit.setEnabled(local)
        self.local_browse_btn.setEnabled(local)
        self.hf_panel.setEnabled(not local)

    def toggle_token_visibility(self):
        if self.hf_token_edit.echoMode() == QLineEdit.Password:
            self.hf_token_edit.setEchoMode(QLineEdit.Normal)
            self.toggle_token_btn.setText("Hide")
        else:
            self.hf_token_edit.setEchoMode(QLineEdit.Password)
            self.toggle_token_btn.setText("Show")

    def keyring_available(self):
        if keyring is None:
            return False
        try:
            backend = keyring.get_keyring()
            backend_name = backend.__class__.__name__.lower()
            return getattr(backend, "priority", 0) > 0 and "fail" not in backend_name
        except Exception:
            return False

    def refresh_keyring_state(self, initial=False):
        available = self.keyring_available()
        self.save_token_btn.setEnabled(available)
        self.load_token_btn.setEnabled(available)
        self.clear_token_btn.setEnabled(available)
        if available:
            msg = "Secure token storage available (OS keychain)."
        else:
            msg = "Secure token storage unavailable. Session-only token input is still supported."
        self.keyring_status_label.setText(msg)
        if not initial:
            self.append_log(msg)

    def has_saved_token(self):
        if not self.keyring_available():
            return False
        try:
            token = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            return bool(token)
        except Exception:
            return False

    def save_token(self):
        if not self.keyring_available():
            QMessageBox.warning(self, "Secure storage unavailable", "No usable keyring backend found. Token cannot be persisted securely on this system.")
            return
        token = self.hf_token_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "Missing token", "Enter a token first.")
            return
        try:
            keyring.set_password(KEYRING_SERVICE, KEYRING_ACCOUNT, token)
            self.append_log("HF token saved securely to OS keychain.")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save token securely: {exc}")

    def load_token(self):
        if not self.keyring_available():
            QMessageBox.warning(self, "Secure storage unavailable", "No usable keyring backend found.")
            return
        try:
            token = keyring.get_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load token: {exc}")
            return
        if not token:
            QMessageBox.information(self, "No token", "No saved token found in keychain.")
            return
        self.hf_token_edit.setText(token)
        self.append_log("HF token loaded from OS keychain into current session.")

    def clear_token(self):
        if not self.keyring_available():
            QMessageBox.warning(self, "Secure storage unavailable", "No usable keyring backend found.")
            return
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_ACCOUNT)
            self.append_log("HF token removed from OS keychain.")
        except Exception:
            self.append_log("No saved HF token to clear.")

    def selected_roles(self):
        roles = set()
        if self.role_human.isChecked():
            roles.add("human")
        if self.role_gpt.isChecked():
            roles.add("gpt")
        if self.role_system.isChecked():
            roles.add("system")
        return roles or None

    def _toggle_dry_run(self, _state=None):
        is_dry = self.dry_run_checkbox.isChecked()
        self.output_name_edit.setEnabled(not is_dry)
        self.audit_checkbox.setEnabled(not is_dry)

    def run_processing(self):
        source_mode = "local" if self.local_radio.isChecked() else "hf"
        local_input = self.local_path_edit.text().strip()
        dataset_id = self.dataset_id_edit.text().strip()
        config_name = self.config_name_edit.text().strip()
        split = self.split_edit.text().strip() or "train"
        hf_token = self.hf_token_edit.text().strip()
        output_name = self.output_name_edit.text().strip()
        write_audit = self.audit_checkbox.isChecked()
        dry_run = self.dry_run_checkbox.isChecked()
        preview_limit = self.preview_limit_spin.value()
        model_profile = self.model_profile_combo.currentData() or "spell_conservative"
        enable_llm_refine = self.enable_llm_checkbox.isChecked()
        llm_model_id = self.llm_model_id_edit.text().strip()
        allowed_roles = self.selected_roles()

        if source_mode == "local":
            if not local_input:
                QMessageBox.critical(self, "Missing input", "Please select a local .json or .jsonl file.")
                return
            if not os.path.exists(local_input):
                QMessageBox.critical(self, "Input not found", "Selected local file does not exist.")
                return
        else:
            if not dataset_id:
                QMessageBox.critical(self, "Missing dataset ID", "Please enter a Hugging Face dataset ID.")
                return

        self.run_btn.setEnabled(False)
        self.log_text.clear()
        self.append_log("Starting SpaceMancer...")
        self.append_log(f"Profile: {self.model_profile_combo.currentText()}")
        if enable_llm_refine:
            self.append_log(f"LLM refinement model: {llm_model_id}")
        if dry_run:
            self.append_log(f"Dry-run preview limit: {preview_limit}")

        self.worker = SpaceMancerWorker(
            source_mode=source_mode,
            local_input=local_input,
            dataset_id=dataset_id,
            config_name=config_name,
            split=split,
            hf_token=hf_token,
            output_name=output_name,
            write_audit=write_audit,
            dry_run=dry_run,
            preview_limit=preview_limit,
            model_profile=model_profile,
            enable_llm_refine=enable_llm_refine,
            llm_model_id=llm_model_id,
            allowed_roles=allowed_roles,
        )
        self.worker.status.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, stats):
        self.run_btn.setEnabled(True)
        self.save_config()

        self.append_log("SpaceMancer completed.")
        self.append_log(f"Rows processed: {stats.get('total_rows', 0)}")
        self.append_log(f"Rows changed: {stats.get('changed_rows', 0)}")
        self.append_log(f"Turns changed: {stats.get('changed_turns', 0)}")
        self.append_log(f"Rows skipped: {stats.get('skipped_rows', 0)}")
        if stats.get("dry_run"):
            self.append_log("Dry run complete: no files were written.")
            preview = stats.get("preview", [])
            self.append_log(f"Preview edits shown: {len(preview)}")
            for idx, entry in enumerate(preview, start=1):
                before = entry.get("before", "").replace("\n", " ")[:140]
                after = entry.get("after", "").replace("\n", " ")[:140]
                self.append_log(
                    f"[{idx}] row {entry.get('row_index')} turn {entry.get('turn_index')} ({entry.get('role', 'unknown')}):"
                )
                self.append_log(f"  before: {before}")
                self.append_log(f"  after : {after}")
        else:
            self.append_log(f"Output: {stats.get('output_path', '')}")
        if not stats.get("dry_run") and stats.get("audit_path"):
            self.append_log(f"Audit: {stats.get('audit_path')}")

    def on_error(self, message):
        self.run_btn.setEnabled(True)
        self.append_log(f"Error: {message}")
        QMessageBox.critical(self, "SpaceMancer error", message)


if __name__ == "__main__":
    from App.Other.Theme import Theme

    app = QApplication(sys.argv)
    win = SpaceMancerApp(Theme.DARK)
    win.show()
    sys.exit(app.exec_())
