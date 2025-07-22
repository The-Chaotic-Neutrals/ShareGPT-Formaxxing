import spacy
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from datasets import load_dataset
from pathlib import Path
import numpy as np

# Silence TorchDynamo warnings
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# Filter mode constants
FILTER_MODE_RP = "rp"
FILTER_MODE_NORMAL = "normal"

# Model names
MODEL_NAMES = {
    FILTER_MODE_RP: "Dans-DiscountModels/Dans-Classifier-RP-Validity-V1.0.0-396m",
    FILTER_MODE_NORMAL: "protectai/distilroberta-base-rejection-v1"
}

# Global state
nlp = None
tokenizers = {}
models = {}
filter_mode = FILTER_MODE_RP

total_input_count = 0
total_kept_count = 0
total_positive_count = 0
total_negative_count = 0


def get_model_name():
    return MODEL_NAMES[filter_mode]


def set_filter_mode(mode):
    global filter_mode
    if mode not in MODEL_NAMES:
        raise ValueError(f"Invalid filter mode: {mode}")
    filter_mode = mode
    initialize_models()


def get_class_indices():
    if filter_mode == FILTER_MODE_RP:
        return 0, 1  # 0 = refusal, 1 = safe
    else:
        return 1, 0  # 1 = refusal, 0 = safe


def initialize_models(status_update_callback=None):
    import torch._dynamo
    global nlp, tokenizers, models

    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")

    # Force device to GPU if available, no CPU fallback or ONNX
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable flash attention if supported (faster inference)
    if device == 'cuda' and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            if status_update_callback:
                status_update_callback("Flash attention enabled for faster inference.")
        except Exception:
            if status_update_callback:
                status_update_callback("Flash attention not available, using standard attention.")

    model_name = get_model_name()

    if model_name not in tokenizers:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    if model_name not in models:
        if "RP-Validity" in model_name or filter_mode == FILTER_MODE_RP:
            torch._dynamo.config.suppress_errors = True
        models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(torch.device(device))


def update_device_preference(gpu_var, status_update_callback=None):
    if status_update_callback:
        status_update_callback("Using GPU only (CPU/ONNX support removed)")
    initialize_models(status_update_callback=status_update_callback)


def filter_conversations(input_file_entry, threshold_entry, batch_size_entry,
                         status_update_callback=None, counts_update_callback=None):
    input_file = input_file_entry.get()
    if not input_file.endswith('.jsonl'):
        if status_update_callback:
            status_update_callback("Invalid file type. Please select a .jsonl file.")
        return

    try:
        threshold = float(threshold_entry.get())
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0.")
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    try:
        batch_size = int(batch_size_entry.get())
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
    except ValueError as e:
        if status_update_callback:
            status_update_callback(f"Error: {e}")
        return

    output_dir = "classified"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

    run_filter_streaming(input_file, output_file, threshold, batch_size,
                         status_update_callback, counts_update_callback)


def run_filter_streaming(input_file, output_file, threshold, batch_size,
                         status_update_callback=None, counts_update_callback=None):
    try:
        # Get total lines for progress tracking
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        dataset = load_dataset("json", data_files=input_file, split="train", streaming=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = open(output_file, 'w', encoding='utf-8')

        global total_positive_count, total_negative_count, total_input_count, total_kept_count
        total_positive_count = 0
        total_negative_count = 0
        total_input_count = 0
        total_kept_count = 0

        if status_update_callback:
            status_update_callback(f"Streaming and filtering started... Total lines: {total_lines}")

        batch = []
        microbatch_size = max(1, batch_size // 4)

        for item in dataset:
            total_input_count += 1
            item_cleaned = validate_json(item)
            if item_cleaned:
                batch.append(item_cleaned)
                if len(batch) >= batch_size:
                    _process_streaming_batch(batch, threshold, microbatch_size, writer,
                                             status_update_callback, counts_update_callback)
                    batch = []

            if status_update_callback and total_input_count % 100 == 0:
                percent_done = (total_input_count / total_lines) * 100
                remaining = total_lines - total_input_count
                status_update_callback(
                    f"Processed {total_input_count}/{total_lines} lines ({percent_done:.1f}%). Remaining: {remaining}"
                )

        if batch:
            _process_streaming_batch(batch, threshold, microbatch_size, writer,
                                     status_update_callback, counts_update_callback)

        writer.close()

        removed = total_input_count - total_kept_count
        if status_update_callback:
            status_update_callback(
                f"Filtering complete. Kept: {total_kept_count} | Removed: {removed} | Total: {total_input_count}. Output: {output_file}"
            )

    except Exception as e:
        if status_update_callback:
            status_update_callback(f"Streaming error: {e}")


def _process_streaming_batch(batch, threshold, microbatch_size, writer,
                             status_update_callback=None, counts_update_callback=None):
    global total_kept_count
    for conversation in batch:
        result = process_conversation(conversation, threshold, microbatch_size,
                                      status_update_callback, counts_update_callback)
        if result:
            total_kept_count += 1
            json_str = json.dumps(result, ensure_ascii=False)
            json_str = clean_text(json_str)
            if validate_utf8(json_str):
                writer.write(json_str + "\n")


def process_conversation(conversation, threshold, microbatch_size,
                         status_update_callback=None, counts_update_callback=None):
    sentences = []
    for turn in conversation.get('conversations', []):
        if turn.get('from') == 'gpt':
            value = clean_text(turn.get('value', ''))
            doc = nlp(value)
            sentences.extend(extract_sentences(doc))

    classifications = predict(sentences, microbatch_size)

    positive_count = sum(1 for classification in classifications if classification['positive'] > threshold)
    negative_count = sum(1 for classification in classifications if classification['negative'] > threshold)

    global total_positive_count, total_negative_count
    total_positive_count += positive_count
    total_negative_count += negative_count

    if counts_update_callback:
        counts_update_callback(total_positive_count, total_negative_count)

    if filter_mode == FILTER_MODE_RP:
        keep_conversation = (positive_count > 0)  # keep refusals
    else:
        keep_conversation = (positive_count == 0)  # keep safe only

    return conversation if keep_conversation else None


def extract_sentences(doc):
    return [clean_text(sent.text.strip()) for sent in doc.sents]


def update_status(message, status_update_callback=None):
    if status_update_callback:
        status_update_callback(message)


def update_counts(positive_count, negative_count, counts_update_callback=None):
    if counts_update_callback:
        counts_update_callback(positive_count, negative_count)


def predict(texts, microbatch_size=4):
    results = []
    model_name = get_model_name()
    positive_idx, negative_idx = get_class_indices()

    for i in range(0, len(texts), microbatch_size):
        chunk = texts[i:i + microbatch_size]

        inputs = tokenizers[model_name](chunk, padding=True, truncation=True, return_tensors="pt", max_length=512).to(models[model_name].device)
        with torch.no_grad():
            logits = models[model_name](**inputs).logits

        predictions = torch.sigmoid(logits).cpu().numpy()

        results.extend([
            {"positive": float(pred[positive_idx]), "negative": float(pred[negative_idx])}
            for pred in predictions
        ])

    return results


def validate_json(conversation):
    if not isinstance(conversation, dict):
        return None
    valid_keys = {'conversations'}
    cleaned_conversation = {k: v for k, v in conversation.items() if k in valid_keys}
    if 'conversations' in cleaned_conversation:
        cleaned_conversation['conversations'] = [
            turn for turn in cleaned_conversation['conversations']
            if isinstance(turn, dict) and 'from' in turn and 'value' in turn
        ]
    return cleaned_conversation if cleaned_conversation else None


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text


def validate_utf8(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False
