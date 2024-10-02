import spacy
import jsonlines
import json
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

MODEL_NAME = "protectai/distilroberta-base-rejection-v1"

# Initialize global variables
nlp = None
tokenizer = None
model = None
device = 'gpu'

def initialize_models():
    """Initialize Spacy and Transformer models."""
    global nlp, tokenizer, model, device
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

def update_device_preference(gpu_var, status_bar):
    """Update the device preference based on user selection."""
    global device
    if gpu_var.get() and torch.cuda.is_available():
        device = torch.device('cuda')
        update_status("Using GPU", status_bar)
    else:
        device = torch.device('cpu')
        update_status("Using CPU", status_bar)

    model.to(device)  # Ensure the model is moved to the correct device

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

    # Perform filtering in the main thread to debug the issue
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
        with open(input_file, 'r', encoding='utf-8') as infile:
            # Ensure input data is valid UTF-8 and clean non-printable characters
            lines = infile.readlines()
            valid_lines = [clean_text(line) for line in lines if validate_utf8(line)]
        
        with jsonlines.open(input_file, mode='r') as reader:
            total_lines = len(valid_lines)
            update_status(f"Total lines: {total_lines}", status_bar)

            # Sequentially process each batch
            with open(output_file, mode='w', encoding='utf-8') as writer:
                batch = []
                for line in valid_lines:
                    conversation = json.loads(line)
                    batch.append(conversation)
                    if len(batch) >= batch_size:
                        process_batch(batch, threshold, writer, status_bar, positive_count_label, negative_count_label)
                        batch = []

                # Process remaining batch
                if batch:
                    process_batch(batch, threshold, writer, status_bar, positive_count_label, negative_count_label)

        update_status(f"Filtering complete. Output file: {output_file}", status_bar)

    except Exception as e:
        update_status(f"Error processing JSONL file: {e}", status_bar)

def process_batch(batch, threshold, writer, status_bar, positive_count_label, negative_count_label):
    """Process a batch of conversations and write the results to the output file."""
    try:
        for conversation in batch:
            result = process_conversation(conversation, threshold, status_bar, positive_count_label, negative_count_label)
            if result:
                # Validate and clean up the result
                valid_result = validate_json(result)
                if valid_result:
                    # Convert result to JSON string
                    json_str = json.dumps(valid_result, ensure_ascii=False)
                    # Clean the JSON string for UTF-8 validity
                    json_str = clean_text(json_str)
                    if validate_utf8(json_str):
                        # Write to file
                        writer.write(json_str + '\n')
                    else:
                        update_status("Error: JSON string is not valid UTF-8.", status_bar)
    except Exception as e:
        update_status(f"Error processing batch: {e}", status_bar)

def process_conversation(conversation, threshold, status_bar, positive_count_label, negative_count_label):
    """Process a single conversation and classify its sentences."""
    keep_conversation = True

    sentences = []
    for turn in conversation.get('conversations', []):
        if turn.get('from') == 'gpt':
            # Clean the conversation value
            value = clean_text(turn.get('value', ''))
            doc = nlp(value)
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
    return [clean_text(sent.text.strip()) for sent in doc.sents]  # Clean sentences here

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
    # Encode and move input tensors to the same device as the model
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Ensure model is using the same device as the input tensors
        outputs = model(**encoded_inputs)

    logits = outputs.logits
    predictions = torch.sigmoid(logits).cpu().numpy()  # Move results to CPU for processing
    results = [{"positive": float(pred[1]), "negative": float(pred[0])} for pred in predictions]
    return results

def validate_json(conversation):
    """Validate and clean up the JSON object."""
    if not isinstance(conversation, dict):
        return None

    # Remove any keys that might invalidate the JSON
    valid_keys = {'conversations'}
    cleaned_conversation = {k: v for k, v in conversation.items() if k in valid_keys}
    
    if 'conversations' in cleaned_conversation:
        # Ensure 'conversations' key contains a list of valid entries
        cleaned_conversation['conversations'] = [
            turn for turn in cleaned_conversation['conversations']
            if isinstance(turn, dict) and 'from' in turn and 'value' in turn
        ]
    
    return cleaned_conversation if cleaned_conversation else None

def clean_text(text):
    """Clean up unwanted characters from the text."""
    # Remove non-printable and control characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = ''.join(c for c in text if c.isprintable())  # Remove non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # Remove ASCII control characters
    return text

def validate_utf8(text):
    """Ensure text is valid UTF-8."""
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

def validate_utf8_byte_level(file_path):
    """Ensure the file content is valid UTF-8 at the byte level."""
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            content.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
