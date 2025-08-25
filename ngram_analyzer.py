import collections
import re
import json
import argparse
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import string
from difflib import SequenceMatcher
import subprocess
import platform

# Attempt to download stopwords if not already available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

# Expand stopword list
EXTRA_STOPWORDS = {'said', 'ask', 'tell', 'response', 'question'}
STOP_WORDS.update(EXTRA_STOPWORDS)

def get_line_count(filename):
    """
    Get the line count of a file using 'wc -l' on Unix-like systems or 'find /c /v ""' on Windows.
   
    Args:
        filename (str): Path to the file.
   
    Returns:
        int or None: Number of lines in the file, or None if not supported or command fails.
    """
    system = platform.system()
    try:
        if system in ["Linux", "Darwin"]:  # Unix-like systems (Linux, macOS)
            result = subprocess.run(
                ["wc", "-l", filename],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract line count from 'wc -l' output (format: "   123 filename")
            return int(result.stdout.split()[0])
        elif system == "Windows":  # Windows systems
            result = subprocess.run(
                ["find", "/c", "/v", "", filename],
                capture_output=True,
                text=True,
                check=True,
                shell=True  # Required for Windows 'find' command
            )
            # Extract line count from 'find /c /v ""' output (format: "---------- filename: 123")
            return int(result.stdout.split()[-1])
    except (subprocess.SubprocessError, ValueError):
        return None
    return None

def tokenize(string, no_punctuation=False):
    """
    Tokenizes a string into words. Optionally removes punctuation if no_punctuation is True.
   
    Args:
        string (str): The input string to tokenize.
        no_punctuation (bool): If True, punctuation will be removed from the tokens.
   
    Returns:
        list: A list of tokens (words).
    """
    # Normalize string: handle contractions and special characters
    string = re.sub(r"’", "'", string) # Normalize apostrophes
    string = re.sub(r"[\n\t]+", " ", string) # Replace newlines/tabs with space
   
    if no_punctuation:
        # Remove all punctuation if no_punctuation flag is True
        string = re.sub(r'[^\w\s\']', '', string) # Keep apostrophes for contractions
        words = re.findall(r"\b[\w']+\b", string.lower())
    else:
        # Capture words (including contractions) and punctuation as separate tokens
        words = re.findall(r"[\w']+|[^\w\s]", string.lower())
   
    return [word for word in words if word] # Filter out empty tokens

def ngram_similarity(ngram1, ngram2):
    """
    Calculate similarity between two n-grams based on word overlap.
   
    Args:
        ngram1 (tuple): First n-gram.
        ngram2 (tuple): Second n-gram.
   
    Returns:
        float: Similarity score (0 to 1).
    """
    str1, str2 = ' '.join(ngram1), ' '.join(ngram2)
    return SequenceMatcher(None, str1, str2).ratio()

def count_ngrams(lines, min_length=3, max_length=5, stopword_limit=1, punctuation_limit=1, no_punctuation=False, similarity_threshold=0.85):
    """
    Counts n-grams from a list of text lines, filtering out repetitive or similar n-grams.
   
    Args:
        lines (list): A list of text lines to process.
        min_length (int): The minimum n-gram length.
        max_length (int): The maximum n-gram length.
        stopword_limit (int): The maximum number of stopwords allowed in each n-gram.
        punctuation_limit (int): The maximum number of punctuation tokens allowed in each n-gram.
        no_punctuation (bool): Whether to filter out punctuation from the tokens.
        similarity_threshold (float): Threshold for filtering similar n-grams (0 to 1).
   
    Returns:
        dict: A dictionary of n-gram counts, categorized by n-gram length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    processed_lines = []
   
    for line in lines:
        processed_lines.append(line) # Store for debugging
        words = tokenize(line, no_punctuation=no_punctuation)
        if len(words) < min_length:
            continue
       
        for n in lengths:
            temp_ngrams = []
            for i in range(len(words) - n + 1):
                current_slice = tuple(words[i:i + n])
               
                # Filter based on stopwords and punctuation
                stopwords_in_ngram = sum(1 for word in current_slice if word in STOP_WORDS)
                punctuation_in_ngram = sum(1 for word in current_slice if word in string.punctuation)
               
                if stopwords_in_ngram > stopword_limit or punctuation_in_ngram > punctuation_limit:
                    continue
               
                temp_ngrams.append((current_slice, 1))
           
            # Deduplicate similar n-grams
            filtered_ngrams = []
            for ngram, count in temp_ngrams:
                if not any(ngram_similarity(ngram, kept_ngram[0]) > similarity_threshold for kept_ngram in filtered_ngrams):
                    filtered_ngrams.append((ngram, count))
           
            for ngram, count in filtered_ngrams:
                ngrams[n][ngram] += count
   
    return ngrams, processed_lines

def process_jsonl(filename, role_filter, no_punctuation=False):
    """
    Processes a JSONL file and yields conversation text based on the specified role filter.
   
    Args:
        filename (str): Path to the JSONL file.
        role_filter (list): List of roles to include in the output (e.g., ['gpt', 'system']).
        no_punctuation (bool): Whether to filter out punctuation in the tokenized text.
   
    Yields:
        str: The conversation text for each valid entry in the JSONL file.
    """
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            # Get line count using system-appropriate command
            total_lines = get_line_count(filename)
            # Use total_lines if available, else omit for progress bar
            tqdm_kwargs = {"desc": "Processing JSONL", "unit": "line"}
            if total_lines is not None:
                tqdm_kwargs["total"] = total_lines
            with tqdm(**tqdm_kwargs) as pbar:
                for line in f:
                    try:
                        json_obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line[:100]}...")
                        pbar.update(1)
                        continue
                   
                    for conversation in json_obj.get("conversations", []):
                        if "value" in conversation:
                            sender = conversation.get("from", "").lower()
                            if role_filter == ['all'] or sender in role_filter:
                                yield conversation["value"]
                    pbar.update(1)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError:
        print(f"Error: Unable to read the file '{filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file and extract n-grams from conversations.")
    parser.add_argument('filename', type=str, help="Path to the JSONL file")
    parser.add_argument('--role_filter', type=str, nargs='+', default=['gpt'], choices=['gpt', 'system', 'all'],
                        help="Role(s) to filter by ('gpt', 'system', or 'all'). Default is 'gpt'.")
    parser.add_argument('--min_ngram', type=int, default=3, help="Minimum n-gram length (default is 3).")
    parser.add_argument('--max_ngram', type=int, default=5, help="Maximum n-gram length (default is 5).")
    parser.add_argument('--stopword_limit', type=int, default=1, help="Maximum number of stopwords allowed in n-grams (default is 1).")
    parser.add_argument('--punctuation_limit', type=int, default=1, help="Maximum number of punctuation tokens allowed in n-grams (default is 1).")
    parser.add_argument('--no_punctuation', action='store_true', help="Remove punctuation from tokens.")
    parser.add_argument('--similarity_threshold', type=float, default=0.85, help="Similarity threshold for n-gram deduplication (0 to 1, default is 0.85).")
    parser.add_argument('--output_file', type=str, default=None, help="File to save n-gram results (optional).")
   
    args = parser.parse_args()
   
    # Process the JSONL file
    lines = list(process_jsonl(args.filename, args.role_filter, no_punctuation=args.no_punctuation))
   
    # Count n-grams
    ngrams, processed_lines = count_ngrams(
        lines,
        min_length=args.min_ngram,
        max_length=args.max_ngram,
        stopword_limit=args.stopword_limit,
        punctuation_limit=args.punctuation_limit,
        no_punctuation=args.no_punctuation,
        similarity_threshold=args.similarity_threshold
    )
   
    # Display results
    print("\nSample Processed Lines (first 5):")
    for line in processed_lines[:5]:
        print(f"- {line[:100]}{'...' if len(line) > 100 else ''}")
   
    print("\nN-gram Counts:")
    output_lines = []
    for n, counts in ngrams.items():
        print(f"\n{n}-grams:")
        output_lines.append(f"\n{n}-grams:")
        for ngram, count in counts.most_common(10):
            ngram_str = ' '.join(ngram)
            print(f"{ngram_str}: {count}")
            output_lines.append(f"{ngram_str}: {count}")
   
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()