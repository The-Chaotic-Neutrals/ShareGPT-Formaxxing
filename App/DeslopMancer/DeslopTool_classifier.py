import logging
import os
import threading

import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

if __package__:
    from .dataset_rows import assistant_texts, read_rows, removed_report, write_rows
else:
    from dataset_rows import assistant_texts, read_rows, removed_report, write_rows

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CharacterSlopFilter:
    def __init__(self, model_name="kubernetes-bad/character-slop-classifier", batch_size=64, confidence_margin=0.1, eager_load_model=False):
        self.model_name = model_name
        if batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
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
        if not sentences:
            return []
        self._lazy_load_model()
        unique = list(dict.fromkeys(sentences))
        results = self.classifier(unique, batch_size=self.batch_size, truncation=True, max_length=512)
        if len(results) != len(unique):
            raise RuntimeError("Classifier returned a different number of scores than sentences.")
        by_sentence = dict(zip(unique, results))
        return [by_sentence[sentence] for sentence in sentences]

    def _removal(self, sentence_results):
        count = len(sentence_results)
        positive = sum(
            1 for result in sentence_results
            if result["label"].casefold() == "positive" and result["score"] > 0.5 + self.confidence_margin
        )
        ratio = positive / count if count else 0
        if ratio > 0.55:
            return {
                "reason": "assistant_classifier_threshold", "positive_sentences": positive,
                "scored_sentences": count, "positive_ratio": ratio, "threshold": 0.55,
            }
        return None

    def process_single_conversation(self, conversation_data):
        sentences = [
            sentence for text in assistant_texts(conversation_data)
            for sentence in self.split_into_sentences(text)
        ]
        if self._removal(self.classify_sentences(sentences)) is None:
            return conversation_data
        return None

    def filter_conversations(self, input_path, output_path, progress_callback=None):
        rows = list(read_rows(input_path))
        total = len(rows)
        kept, removed = [], []
        for offset in range(0, total, self.batch_size):
            batch = rows[offset:offset + self.batch_size]
            sentences = []
            ends = []
            for row in batch:
                for text in row.assistant_texts():
                    sentences.extend(self.split_into_sentences(text))
                ends.append(len(sentences))
            results = self.classify_sentences(sentences)
            start = 0
            for row, end in zip(batch, ends):
                reason = self._removal(results[start:end])
                start = end
                if reason is None:
                    kept.append(row)
                else:
                    removed.append({**row.locate(), **reason})
            if progress_callback:
                progress_callback(offset + len(batch), total)
        write_rows(output_path, kept, input_path)
        write_rows(removed_report(output_path), removed, input_path)
        logging.info("Filtering complete. %s/%s conversations kept.", len(kept), total)
        return len(kept), total


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
