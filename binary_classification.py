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
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

# Initialize global variables
nlp = None
tokenizer = None
session = None
model = None
device = 'gpu'
use_onnx = False

def initialize_models():
    """Initialize Spacy, Tokenizer, and model based on device preference."""
    global nlp, tokenizer, session, model, device, use_onnx

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    use_onnx = (device == 'cpu')

    if use_onnx:
        providers = ["CPUExecutionProvider"]
        onnx_dir = Path("onnx_model")
        onnx_fp = onnx_dir / "model.onnx"
        quant_fp = onnx_dir / "model-quant.onnx"

        if not quant_fp.exists():
            os.makedirs(onnx_dir, exist_ok=True)
            model_tmp = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            model_tmp.eval()

            model_kind, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(model_tmp)
            onnx_config = onnx_config_cls(model_tmp.config)

            export(
                preprocessor=tokenizer,
                model=model_tmp,
                config=onnx_config,
                opset=14,
                output=onnx_fp
            )

            quantize_dynamic(
                model_input=str(onnx_fp),
                model_output=str(quant_fp),
                weight_type=QuantType.QInt8
            )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session = ort.InferenceSession(str(quant_fp), sess_options, providers=providers)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        model.to(torch.device("cuda"))

def update_device_preference(gpu_var, status_bar):
    global device, session, model, use_onnx
    if gpu_var.get() and torch.cuda.is_available():
        device = 'gpu'
        use_onnx = False
        update_status("Using GPU", status_bar)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        model.to(torch.device("cuda"))
    else:
        device = 'cpu'
        use_onnx = True
        update_status("Using CPU", status_bar)
        initialize_models()

def filter_conversations(input_file_entry, threshold_entry, batch_size_entry, status_bar, positive_count_label, negative_count_label):
    input_file = input_file_entry.get()
    if not input_file.endswith('.jsonl'):
        update_status("Invalid file type. Please select a .jsonl file.", status_bar)
        return

    try:
        threshold = float(threshold_entry.get())
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0.")
    except ValueError as e:
        update_status(f"Error: {e}", status_bar)
        return

    try:
        batch_size = int(batch_size_entry.get())
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
    except ValueError as e:
        update_status(f"Error: {e}", status_bar)
        return

    output_dir = "classified"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

    run_filter_streaming(input_file, output_file, threshold, batch_size, status_bar, positive_count_label, negative_count_label)

def run_filter_streaming(input_file, output_file, threshold, batch_size, status_bar, positive_count_label, negative_count_label):
    try:
        dataset = load_dataset("json", data_files=input_file, split="train", streaming=True)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = open(output_file, 'w', encoding='utf-8')

        global total_positive_count, total_negative_count
        total_positive_count = 0
        total_negative_count = 0

        update_status("Streaming and filtering started...", status_bar)

        batch = []
        microbatch_size = max(1, batch_size // 4)

        for item in dataset:
            item_cleaned = validate_json(item)
            if item_cleaned:
                batch.append(item_cleaned)
                if len(batch) >= batch_size:
                    _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label)
                    batch = []

        if batch:
            _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label)

        writer.close()
        update_status(f"Filtering complete. Output file: {output_file}", status_bar)

    except Exception as e:
        update_status(f"Streaming error: {e}", status_bar)

def _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label):
    for conversation in batch:
        result = process_conversation(conversation, threshold, microbatch_size, status_bar, positive_count_label, negative_count_label)
        if result:
            json_str = json.dumps(result, ensure_ascii=False)
            json_str = clean_text(json_str)
            if validate_utf8(json_str):
                writer.write(json_str + "\n")

def process_conversation(conversation, threshold, microbatch_size, status_bar, positive_count_label, negative_count_label):
    keep_conversation = True
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

    update_counts(total_positive_count, total_negative_count, positive_count_label, negative_count_label)

    if positive_count > 0:
        keep_conversation = False

    return conversation if keep_conversation else None

def extract_sentences(doc):
    return [clean_text(sent.text.strip()) for sent in doc.sents]

def update_status(message, status_bar):
    if status_bar:
        status_bar.after(0, lambda: status_bar.config(text=f"Status: {message}"))

def update_counts(positive_count, negative_count, positive_count_label, negative_count_label):
    if positive_count_label and negative_count_label:
        positive_count_label.after(0, lambda: positive_count_label.config(text=f"Positive Count: {positive_count}"))
        negative_count_label.after(0, lambda: negative_count_label.config(text=f"Negative Count: {negative_count}"))

def predict(texts, microbatch_size=4):
    results = []
    for i in range(0, len(texts), microbatch_size):
        chunk = texts[i:i + microbatch_size]

        if use_onnx:
            inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="np", max_length=512)
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            outputs = session.run(None, inputs)
            logits = torch.tensor(outputs[0])
        else:
            inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt", max_length=512).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits

        predictions = torch.sigmoid(logits).cpu().numpy()

        results.extend([
            {"positive": float(pred[1]), "negative": float(pred[0])}
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
