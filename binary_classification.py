import spacy
import jsonlines
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

# Initialize global variables
nlp = None
tokenizer = None
model = None
device = 'cpu'

def initialize_models():
    """Initialize Spacy and Transformer models."""
    global nlp, tokenizer, model
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def update_device_preference(gpu_var, status_bar):
    """Update the device preference based on user selection."""
    global device
    if gpu_var.get() and torch.cuda.is_available():
        device = 'cuda'
        update_status("Using GPU", status_bar)
    else:
        device = 'cpu'
        update_status("Using CPU", status_bar)
    model.to(device)

def filter_conversations(input_file_entry, threshold_entry, batch_size_entry, status_bar, positive_count_label, negative_count_label):
    """Filter conversations based on the specified criteria."""
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

    # Create output directory if it doesn't exist
    output_dir = "classified"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")

    # Perform filtering in a separate thread to keep the UI responsive
    update_status("Filtering started...", status_bar)
    try:
        global total_positive_count, total_negative_count
        total_positive_count = 0
        total_negative_count = 0
        run_filter(input_file, output_file, threshold, batch_size, status_bar, positive_count_label, negative_count_label)
        update_status(f"Filtering complete. Output file: {output_file}", status_bar)
    except Exception as e:
        update_status(f"Error: {e}", status_bar)

def run_filter(input_file, output_file, threshold, batch_size, status_bar, positive_count_label, negative_count_label):
    """Run the filtering process on the input file."""
    try:
        with jsonlines.open(input_file) as reader:
            total_lines = sum(1 for _ in jsonlines.open(input_file))
            update_status(f"Total lines: {total_lines}", status_bar)

            # Use a thread pool for parallel processing of conversations
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                with open(output_file, mode='w') as writer:
                    batch = []
                    for i, conversation in enumerate(reader):
                        batch.append(conversation)
                        if len(batch) >= batch_size:
                            futures.append(executor.submit(process_batch, batch, threshold, writer, status_bar, positive_count_label, negative_count_label))
                            batch = []

                    # Process remaining batch
                    if batch:
                        futures.append(executor.submit(process_batch, batch, threshold, writer, status_bar, positive_count_label, negative_count_label))

                    for future in futures:
                        future.result()  # Ensure all futures are completed

        update_status(f"Filtering complete. Output file: {output_file}", status_bar)

    except Exception as e:
        update_status(f"Error processing JSONL file: {e}", status_bar)

def process_batch(batch, threshold, writer, status_bar, positive_count_label, negative_count_label):
    """Process a batch of conversations and write the results to the output file."""
    try:
        for conversation in batch:
            result = process_conversation(conversation, threshold, status_bar, positive_count_label, negative_count_label)
            if result:
                # Convert result to JSON string and write to file
                writer.write(json.dumps(result) + '\n')
    except Exception as e:
        update_status(f"Error processing batch: {e}", status_bar)

def process_conversation(conversation, threshold, status_bar, positive_count_label, negative_count_label):
    """Process a single conversation and classify its sentences."""
    keep_conversation = True

    sentences = []
    for turn in conversation.get('conversations', []):
        if turn.get('from') == 'gpt':
            doc = nlp(turn.get('value', ''))
            sentences.extend(extract_sentences(doc))

    # Classify sentences and check against threshold
    classifications = predict(sentences)

    # Update counts
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
    """Extract sentences from a Spacy Doc object."""
    return [sent.text.strip() for sent in doc.sents]

def update_status(message, status_bar):
    """Update the status bar with a message."""
    if status_bar:
        status_bar.after(0, lambda: status_bar.config(text=f"Status: {message}"))

def update_counts(positive_count, negative_count, positive_count_label, negative_count_label):
    """Update the positive and negative counts in the UI."""
    if positive_count_label and negative_count_label:
        positive_count_label.after(0, lambda: positive_count_label.config(text=f"Positive Count: {positive_count}"))
        negative_count_label.after(0, lambda: negative_count_label.config(text=f"Negative Count: {negative_count}"))

def predict(texts):
    """Predict the classifications for a batch of texts."""
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    logits = outputs.logits
    predictions = torch.sigmoid(logits).cpu().numpy()
    results = [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]
    return results
