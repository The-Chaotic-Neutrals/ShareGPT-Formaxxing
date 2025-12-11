import os
from pathlib import Path
from contextlib import nullcontext, redirect_stderr
from multiprocessing import Pool, cpu_count
import json
import fasttext
import urllib.request
from tqdm import tqdm
import regex as re
import io
import sys

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
        filtered_dir = outputs_dir / "englishmaxxer" / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        output_path = filtered_dir / f"{input_path.stem}_filtered.jsonl"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            filtered_dir = outputs_dir / "englishmaxxer" / "filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            output_path = filtered_dir / output_path.name

    # Setup default rejected path if provided and relative
    if rejected_path is not None:
        rejected_path = Path(rejected_path)
        if not rejected_path.is_absolute():
            rejected_dir = outputs_dir / "englishmaxxer" / "rejected"
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

    return {
        "total_lines": total_lines,
        "english_total": english_total,
        "non_english_total": non_english_total,
        "json_error_total": json_error_total
    }
