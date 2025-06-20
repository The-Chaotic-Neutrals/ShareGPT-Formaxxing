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
sessions = {}
onnx_initialized = {}
device = 'gpu'
use_onnx = False
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

def initialize_models():
    import torch._dynamo
    global nlp, tokenizers, models, sessions, device, use_onnx, onnx_initialized

    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    use_onnx = (device == 'cpu')
    model_name = get_model_name()

    if model_name not in tokenizers:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    if use_onnx:
        if model_name not in sessions and not onnx_initialized.get(model_name):
            onnx_dir = Path("onnx_model") / model_name.replace('/', '_')
            onnx_fp = onnx_dir / "model.onnx"
            quant_fp = onnx_dir / "model-quant.onnx"
            os.makedirs(onnx_dir, exist_ok=True)

            if not quant_fp.exists():
                model_tmp = AutoModelForSequenceClassification.from_pretrained(model_name)
                model_tmp.eval()
                model_kind, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(model_tmp)
                onnx_config = onnx_config_cls(model_tmp.config)
                export(
                    preprocessor=tokenizers[model_name],
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
            sessions[model_name] = ort.InferenceSession(str(quant_fp), sess_options, providers=["CPUExecutionProvider"])
            onnx_initialized[model_name] = True
    else:
        if model_name not in models:
            if "RP-Validity" in model_name or filter_mode == FILTER_MODE_RP:
                torch._dynamo.config.suppress_errors = True
            models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(torch.device("cuda"))

def update_device_preference(gpu_var, status_bar):
    global device, use_onnx
    if gpu_var.get() and torch.cuda.is_available():
        device = 'gpu'
        use_onnx = False
        update_status("Using GPU", status_bar)
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

        global total_positive_count, total_negative_count, total_input_count, total_kept_count
        total_positive_count = 0
        total_negative_count = 0
        total_input_count = 0
        total_kept_count = 0

        update_status("Streaming and filtering started...", status_bar)

        batch = []
        microbatch_size = max(1, batch_size // 4)

        for item in dataset:
            total_input_count += 1
            item_cleaned = validate_json(item)
            if item_cleaned:
                batch.append(item_cleaned)
                if len(batch) >= batch_size:
                    _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label)
                    batch = []

        if batch:
            _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label)

        writer.close()

        removed = total_input_count - total_kept_count
        update_status(
            f"Filtering complete. Kept: {total_kept_count} | Removed: {removed} | Total: {total_input_count}. Output: {output_file}",
            status_bar
        )

    except Exception as e:
        update_status(f"Streaming error: {e}", status_bar)

def _process_streaming_batch(batch, threshold, microbatch_size, writer, status_bar, positive_count_label, negative_count_label):
    global total_kept_count
    for conversation in batch:
        result = process_conversation(conversation, threshold, microbatch_size, status_bar, positive_count_label, negative_count_label)
        if result:
            total_kept_count += 1
            json_str = json.dumps(result, ensure_ascii=False)
            json_str = clean_text(json_str)
            if validate_utf8(json_str):
                writer.write(json_str + "\n")

def process_conversation(conversation, threshold, microbatch_size, status_bar, positive_count_label, negative_count_label):
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

    if filter_mode == FILTER_MODE_RP:
        keep_conversation = (positive_count > 0)  # keep refusals
    else:
        keep_conversation = (positive_count == 0)  # keep safe only

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
    model_name = get_model_name()
    positive_idx, negative_idx = get_class_indices()

    for i in range(0, len(texts), microbatch_size):
        chunk = texts[i:i + microbatch_size]

        if use_onnx:
            inputs = tokenizers[model_name](chunk, padding=True, truncation=True, return_tensors="np", max_length=512)
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            outputs = sessions[model_name].run(None, inputs)
            logits = torch.tensor(outputs[0])
        else:
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
