import logging
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json
import spacy
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CharacterSlopFilter:
    def __init__(self, model_name="kubernetes-bad/character-slop-classifier", batch_size=64, confidence_margin=0.1, eager_load_model=False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_margin = confidence_margin

        self._model_lock = threading.Lock()
        self._model_loaded = False

        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_dtype = torch.float16 if self.device == 0 else torch.float32  # Use fp16 on GPU if available

        self.tokenizer = None
        self.model = None
        self.classifier = None

        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

        self.classification_cache = defaultdict(dict)

        max_workers = max(os.cpu_count() - 1, 1)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        logging.info(f"Initialized CharacterSlopFilter, model will load lazily on {'GPU' if self.device == 0 else 'CPU'}")
        logging.info(f"Using {max_workers} workers for CPU tasks")

        if eager_load_model:
            self._lazy_load_model()

    def _lazy_load_model(self):
        if self._model_loaded:
            return  # already loaded

        with self._model_lock:
            if self._model_loaded:
                return  # double check inside lock
            logging.info(f"Lazy loading model and tokenizer from {self.model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            )
            self.model.to(self.device)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            self._model_loaded = True
            logging.info("Model and tokenizer loaded.")

    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def classify_sentences(self, sentences):
        self._lazy_load_model()  # Ensure model is loaded before classify
        to_classify = [sent for sent in sentences if sent not in self.classification_cache]
        if to_classify:
            results = self.classifier(
                to_classify,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512,
            )
            for sent, result in zip(to_classify, results):
                self.classification_cache[sent] = {
                    "label": result["label"],
                    "score": result["score"],
                }
        return [self.classification_cache[sent] for sent in sentences]

    def process_single_conversation(self, conversation_data):
        conversations = conversation_data.get("conversations", [])
        gpt_sentences = []

        for conversation in conversations:
            if conversation.get("from") == "gpt":
                text = conversation.get("value", "")
                if text:
                    sentences = self.split_into_sentences(text)
                    gpt_sentences.extend(sentences)

        if not gpt_sentences:
            # No GPT sentences, keep conversation by default
            return conversation_data

        sentence_results = self.classify_sentences(gpt_sentences)
        positive_count = sum(
            1 for result in sentence_results
            if result["label"] == "positive" and result["score"] > 0.5 + self.confidence_margin
        )
        positive_ratio = positive_count / len(gpt_sentences)

        if positive_ratio <= 0.55:
            return conversation_data  # keep conversation
        else:
            return None  # filter out

    def filter_conversations(self, input_path, output_path, progress_callback=None):
        # Optionally force eager load here to avoid concurrency on first call
        if not self._model_loaded:
            self._lazy_load_model()

        dataset = load_dataset("json", data_files=[input_path], split="train")
        total = len(dataset)
        filtered_count = 0

        with open(output_path, "w", encoding="utf-8") as f_out:
            futures = {
                self.executor.submit(self.process_single_conversation, item): idx
                for idx, item in enumerate(dataset)
            }

            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    filtered_count += 1

                if progress_callback:
                    progress_callback(count, total)

        logging.info(f"Filtering complete. {filtered_count}/{total} conversations kept.")
        return filtered_count, total


def select_jsonl_file():
    from tkinter import Tk, filedialog
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JSON Lines", "*.jsonl")], title="Select a JSONL File")
    return file_path


def main():
    slop_filter = CharacterSlopFilter(batch_size=64, confidence_margin=0.1, eager_load_model=True)
    jsonl_filepath = select_jsonl_file()
    if not jsonl_filepath:
        logging.error("No file selected. Exiting.")
        return

    output_jsonl_filepath = os.path.splitext(jsonl_filepath)[0] + "_filtered.jsonl"

    def progress_callback(current, total):
        percent = int((current / total) * 100)
        print(f"Progress: {percent}% ({current}/{total})")

    filtered_count, total = slop_filter.filter_conversations(jsonl_filepath, output_jsonl_filepath, progress_callback=progress_callback)
    print(f"Done filtering. Kept {filtered_count} of {total} conversations.")


if __name__ == "__main__":
    main()
