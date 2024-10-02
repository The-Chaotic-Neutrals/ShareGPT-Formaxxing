import logging
import os
import torch  # Added missing import for torch
from transformers import pipeline
import json
import spacy
import asyncio
import aiofiles  # For asynchronous file I/O
from tqdm import tqdm
from collections import defaultdict
from tkinter import Tk, filedialog  # For file selection dialog

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CharacterSlopFilter:
    def __init__(self, model_name="kubernetes-bad/character-slop-classifier", batch_size=58, confidence_margin=0.1):
        # Initialize a transformers pipeline for classification
        self.classifier = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        # Load spaCy model for sentence segmentation
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        # Configurable parameters
        self.batch_size = batch_size
        self.confidence_margin = confidence_margin
        # Cache for already classified sentences
        self.classification_cache = defaultdict(dict)

        logging.info(f"Pipeline loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def split_into_sentences(self, text):
        # Use spaCy to split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    def classify_sentences(self, sentences):
        # Check cache first to avoid redundant classification
        to_classify = [sent for sent in sentences if sent not in self.classification_cache]
        if to_classify:
            # Classify sentences using the pipeline with truncation to 512 tokens
            results = self.classifier(
                to_classify,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512  # Truncate sentences to a maximum of 512 tokens
            )
            # Store results in the cache
            for sentence, result in zip(to_classify, results):
                self.classification_cache[sentence] = {
                    "label": result["label"],
                    "score": result["score"]
                }

        # Retrieve all classifications from the cache
        return [self.classification_cache[sent] for sent in sentences]

    async def filter_conversations(self, filepath, output_filepath):
        # Read a JSON Lines file asynchronously and filter conversations based on sentence-level "gpt" classifications
        filtered_conversations = []
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                # Read all lines at once for async processing
                lines = await f.readlines()

                for line in tqdm(lines, desc="Processing conversations"):
                    try:
                        data = json.loads(line.strip())
                        conversations = data.get("conversations", [])
                        gpt_sentences = []

                        # Gather all "gpt" sentences for classification
                        for conversation in conversations:
                            if conversation.get("from") == "gpt":
                                text = conversation.get("value", "")
                                if text:
                                    # Split "gpt" entry into sentences
                                    sentences = self.split_into_sentences(text)
                                    gpt_sentences.extend(sentences)

                        # Classify all gpt sentences if any exist
                        if gpt_sentences:
                            sentence_results = self.classify_sentences(gpt_sentences)

                            positive_count = sum(
                                1 for result in sentence_results
                                if result["label"] == "positive" and result["score"] > 0.5 + self.confidence_margin
                            )
                            total_sentences = len(gpt_sentences)

                            # Calculate the percentage of positive sentences
                            positive_ratio = positive_count / total_sentences
                            
                            logging.debug(f"Positive ratio: {positive_ratio} for conversation: {data}")

                            # Keep the conversation if less than 55% of the sentences are positive
                            if positive_ratio <= 0.55:
                                filtered_conversations.append(data)
                    except json.JSONDecodeError as jde:
                        logging.error(f"JSON decode error in line: {line}, error: {jde}")
                    except UnicodeDecodeError as ude:
                        logging.error(f"Unicode decode error in line: {line}, error: {ude}")
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")
        
        # Save filtered results asynchronously to a new .jsonl file
        try:
            async with aiofiles.open(output_filepath, "w", encoding="utf-8") as f_out:
                for conversation in filtered_conversations:
                    await f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            logging.info(f"Filtered conversations saved to {output_filepath}")
        except Exception as e:
            logging.error(f"Error writing to file {output_filepath}: {e}")

def select_jsonl_file():
    # Open a file dialog to select the input .jsonl file
    Tk().withdraw()  # Hide Tkinter root window
    file_path = filedialog.askopenfilename(filetypes=[("JSON Lines", "*.jsonl")], title="Select a JSONL File")
    return file_path

async def main():
    # Initialize filter
    slop_filter = CharacterSlopFilter(batch_size=32, confidence_margin=0.1)

    # Select input file
    jsonl_filepath = select_jsonl_file()
    
    # If no file is selected, exit
    if not jsonl_filepath:
        logging.error("No file selected. Exiting.")
        return
    
    # Create output file path in the same directory as input
    output_jsonl_filepath = os.path.splitext(jsonl_filepath)[0] + "_filtered.jsonl"

    # Filter conversations based on "gpt" sentence-level classification and save the result
    await slop_filter.filter_conversations(jsonl_filepath, output_jsonl_filepath)

if __name__ == "__main__":
    asyncio.run(main())
