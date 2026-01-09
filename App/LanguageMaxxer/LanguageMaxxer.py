"""
LanguageMaxxer - Combined English filtering and grammar correction tool.

This module combines:
- EnglishMaxxer: Filter conversations by language (English detection using FastText)
- GrammarMaxxer: Correct grammar in conversations using LanguageTool
"""

import os
import json
import io
import sys
import logging
from pathlib import Path
from contextlib import nullcontext
from multiprocessing import Pool, cpu_count

import regex as re
import jsonlines
import fasttext
import urllib.request
import language_tool_python
from tqdm import tqdm

# =============================================================================
# English Filtering (formerly EnglishMaxxer)
# =============================================================================

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_FILENAME = "lid.176.ftz"

UNICODE_FILTER_RE = re.compile(
    r'[\u00C0-\u017F\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF]'
)

_model = None


def download_model_if_missing(model_path):
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)


def load_fasttext_model():
    """
    Load the fasttext model while suppressing the warning emitted internally.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    download_model_if_missing(model_path)
    
    # Suppress internal warning by redirecting stderr temporarily
    stderr = sys.stderr
    with io.StringIO() as fake_stderr:
        try:
            sys.stderr = fake_stderr
            model = fasttext.load_model(model_path)
        finally:
            sys.stderr = stderr
    return model


def init_worker():
    global _model
    _model = load_fasttext_model()


def get_model():
    return _model


def is_mostly_english_batch(texts, threshold):
    model = get_model()
    sanitized = [t.replace('\n', ' ') for t in texts]
    predictions = model.predict(sanitized, k=1)
    return [(lang[0], prob[0]) for lang, prob in zip(predictions[0], predictions[1])]


def contains_unwanted_unicode(text):
    return bool(UNICODE_FILTER_RE.search(text))


def extract_gpt_text(conversation):
    return " ".join(
        turn.get("value", "") for turn in conversation.get("conversations", [])
        if isinstance(turn, dict) and turn.get("from") == "gpt" and "value" in turn
    )


def process_batch(args):
    lines, batch_indices, threshold = args
    valid_entries, rejected_entries = [], []
    english_count = non_english_count = json_error_count = 0
    data_objects = []

    for idx, line in zip(batch_indices, lines):
        try:
            data = json.loads(line)
            text = extract_gpt_text(data)
            data_objects.append((idx, line, data, text))
        except Exception:
            rejected_entries.append(line.strip())
            json_error_count += 1

    if not data_objects:
        return valid_entries, rejected_entries, english_count, non_english_count, json_error_count

    predictions = is_mostly_english_batch([x[3] for x in data_objects], threshold)
    for (idx, line, data, text), (lang, prob) in zip(data_objects, predictions):
        if lang == "__label__en" and prob >= threshold and not contains_unwanted_unicode(text):
            valid_entries.append(json.dumps(data, ensure_ascii=False))
            english_count += 1
        else:
            rejected_entries.append(line.strip())
            non_english_count += 1

    return valid_entries, rejected_entries, english_count, non_english_count, json_error_count


def filter_english_jsonl(input_path, output_path=None, rejected_path=None, threshold=0.69, batch_size=128, workers=None):
    """
    Filter a JSONL file to keep only English conversations.
    
    Returns:
        dict: Statistics about the filtering process
        str: Path to the output file
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    # Get repo root (parent of script directory)
    script_dir = Path(__file__).parent.absolute()
    repo_root = script_dir.parent.absolute()
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)

    # Setup default output path if not provided or relative
    if output_path is None:
        filtered_dir = outputs_dir / "languagemaxxer" / "english_filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        output_path = filtered_dir / f"{input_path.stem}_english.jsonl"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            filtered_dir = outputs_dir / "languagemaxxer" / "english_filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            output_path = filtered_dir / output_path.name

    # Setup default rejected path if provided and relative
    if rejected_path is not None:
        rejected_path = Path(rejected_path)
        if not rejected_path.is_absolute():
            rejected_dir = outputs_dir / "languagemaxxer" / "rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            rejected_path = rejected_dir / rejected_path.name

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    batches = [
        (lines[i:i + batch_size], list(range(i, min(i + batch_size, total_lines))), threshold)
        for i in range(0, total_lines, batch_size)
    ]

    english_total = non_english_total = json_error_total = 0

    with Pool(workers, initializer=init_worker) as pool, \
         open(output_path, 'w', encoding='utf-8') as outfile, \
         open(rejected_path, 'w', encoding='utf-8') if rejected_path else nullcontext() as rejfile:

        with tqdm(total=total_lines, desc="Filtering English JSONL") as pbar:
            for valid_entries, rejected_entries, eng, non_eng, err in pool.imap_unordered(process_batch, batches):
                for entry in valid_entries:
                    outfile.write(entry + "\n")
                if rejfile:
                    for entry in rejected_entries:
                        rejfile.write(entry + "\n")
                english_total += eng
                non_english_total += non_eng
                json_error_total += err
                pbar.update(eng + non_eng + err)

    stats = {
        "total_lines": total_lines,
        "english_total": english_total,
        "non_english_total": non_english_total,
        "json_error_total": json_error_total
    }
    return stats, str(output_path)


# =============================================================================
# Grammar Correction (formerly GrammarMaxxer)
# =============================================================================

class GrammarCorrector:
    def __init__(self, input_file, enable_grammar=True):
        self.input_file = input_file
        self.enable_grammar = enable_grammar
        self.tool = language_tool_python.LanguageTool('en-US') if enable_grammar else None

    def validate_file(self):
        """Validate the selected file."""
        if not self.input_file.endswith('.jsonl'):
            logging.error("Invalid file type. Please select a .jsonl file.")
            return False
        return True

    def prepare_output_file(self, output_dir=None):
        """Prepare the output file path."""
        if output_dir is None:
            script_dir = Path(__file__).parent.absolute()
            repo_root = script_dir.parent.absolute()
            output_dir = repo_root / "outputs" / "languagemaxxer" / "grammar_corrected"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        return str(output_dir / f"{base_name}_corrected.jsonl")

    def process_file(self, output_file, update_corrections_callback=None):
        """Process the input file and write the corrected text to the output file."""
        with jsonlines.open(self.input_file) as reader:
            with jsonlines.open(output_file, mode='w') as writer:
                for conversation in reader:
                    corrected_conversation = self.correct_conversation(conversation, update_corrections_callback)
                    writer.write(corrected_conversation)

    def correct_conversation(self, conversation, update_corrections_callback=None):
        """Correct the text in a conversation and update the live tracker."""
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                original_text = turn.get('value', '')
                corrected_text = self.correct_text(original_text)
                turn['value'] = corrected_text
                if update_corrections_callback:
                    update_corrections_callback(original_text, corrected_text)
        return conversation

    def correct_text(self, text):
        """Correct text using grammar correction."""
        if self.enable_grammar and self.tool:
            text = self.correct_with_grammar(text)
        return text.strip()

    def correct_with_grammar(self, text):
        """Correct grammar using LanguageTool."""
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text


def correct_grammar_jsonl(input_path, output_path=None, update_callback=None):
    """
    Correct grammar in a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Optional output path (auto-generated if not provided)
        update_callback: Optional callback for progress updates (original_text, corrected_text)
    
    Returns:
        str: Path to the output file
    """
    corrector = GrammarCorrector(input_path, enable_grammar=True)
    
    if not corrector.validate_file():
        raise ValueError(f"Invalid input file: {input_path}")
    
    if output_path is None:
        output_path = corrector.prepare_output_file()
    
    corrector.process_file(output_path, update_callback)
    return output_path
