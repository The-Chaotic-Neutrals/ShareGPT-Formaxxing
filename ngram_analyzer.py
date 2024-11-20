import collections
import re
import json
import argparse
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import string

# Attempt to download stopwords if not already available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))

# Expand stopword list (you can add more stopwords if necessary)
EXTRA_STOPWORDS = {'said', 'ask', 'tell', 'response', 'question'}  # You can add more words here
STOP_WORDS.update(EXTRA_STOPWORDS)

def tokenize(string, no_punctuation=False):
    """
    Tokenizes a string into words. Optionally removes punctuation if no_punctuation is True.
    
    Args:
        string (str): The input string to tokenize.
        no_punctuation (bool): If True, punctuation will be removed from the tokens.
    
    Returns:
        list: A list of tokens (words).
    """
    if no_punctuation:
        # Remove all punctuation if no_punctuation flag is True
        string = re.sub(r'[^\w\s]', '', string)  # Removes anything that's not a word or space
    
    # If punctuation should remain, include punctuation as separate tokens
    if not no_punctuation:
        words = re.findall(r'\w+|[^\w\s]', string.lower())  # Capture both words and punctuation
    else:
        # Otherwise, we tokenize normally, no punctuation included
        words = re.findall(r'\b\w+\b', string.lower())
    
    return words

def count_ngrams(lines, min_length=3, max_length=5, stopword_limit=1, punctuation_limit=1, no_punctuation=False):
    """
    Counts n-grams (from 3-grams to 5-grams) from a list of text lines, filtering out n-grams with more than allowed stopwords or punctuation.
    
    Args:
        lines (list): A list of text lines to process.
        min_length (int): The minimum n-gram length.
        max_length (int): The maximum n-gram length.
        stopword_limit (int): The maximum number of stopwords allowed in each n-gram (default is 1).
        punctuation_limit (int): The maximum number of punctuation tokens allowed in each n-gram (default is 1).
        no_punctuation (bool): Whether to filter out punctuation from the tokens.
    
    Returns:
        dict: A dictionary of n-gram counts, categorized by n-gram length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}

    for line in lines:
        words = tokenize(line, no_punctuation=no_punctuation)

        # Skip lines that are too short to form n-grams of the desired length
        if len(words) < min_length:
            continue

        # Generate n-grams for the current line only
        for n in lengths:
            for i in range(len(words) - n + 1):
                current_slice = tuple(words[i:i + n])
                
                # Count how many stopwords are in the current n-gram
                stopwords_in_ngram = sum(1 for word in current_slice if word in STOP_WORDS)
                
                # Count how many punctuation tokens are in the current n-gram
                punctuation_in_ngram = sum(1 for word in current_slice if word in string.punctuation)
                
                # If the number of stopwords or punctuation exceeds the allowed limit, skip the n-gram
                if stopwords_in_ngram > stopword_limit or punctuation_in_ngram > punctuation_limit:
                    continue
                
                # Only count n-grams that pass the stopword and punctuation limit
                ngrams[n][current_slice] += 1

    return ngrams

def process_jsonl(filename, role_filter, no_punctuation=False):
    """
    Processes a JSONL file and yields conversation text based on the specified role filter.
    
    Args:
        filename (str): Path to the JSONL file.
        role_filter (list): List of roles to include in the output (e.g., ['gpt', 'system']). Default is ['gpt'].
        no_punctuation (bool): Whether to filter out punctuation in the tokenized text.
    
    Yields:
        str: The conversation text for each valid entry in the JSONL file.
    """
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)  # Count total lines in the file
            f.seek(0)  # Reset file pointer after counting lines

            with tqdm(total=total_lines, desc="Processing JSONL", unit="line") as pbar:
                for line in f:
                    try:
                        json_obj = json.loads(line)  # Parse each line as JSON
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line[:100]}...")  # Skip malformed JSON
                        pbar.update(1)
                        continue
                    
                    for conversation in json_obj.get("conversations", []):
                        if "value" in conversation:
                            # Role filtering logic
                            sender = conversation.get("from", "").lower()
                            
                            # Process based on role_filter
                            if role_filter == ['all'] or sender in role_filter:
                                yield conversation["value"]
                    pbar.update(1)  # Update the progress bar for each line processed
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError:
        print(f"Error: Unable to read the file '{filename}'.")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process a JSONL file and extract n-grams from GPT conversations.")
    
    # Add arguments
    parser.add_argument('filename', type=str, help="Path to the JSONL file")
    parser.add_argument('--role_filter', type=str, nargs='+', default=['gpt'], choices=['gpt', 'system', 'all'], 
                        help="Role(s) to filter by ('gpt', 'system', or 'all'). Default is 'gpt'.")
    parser.add_argument('--min_ngram', type=int, default=3, help="Minimum n-gram length (default is 3).")
    parser.add_argument('--max_ngram', type=int, default=5, help="Maximum n-gram length (default is 5).")
    parser.add_argument('--stopword_limit', type=int, default=1, help="Maximum number of stopwords allowed in n-grams (default is 1).")
    parser.add_argument('--punctuation_limit', type=int, default=1, help="Maximum number of punctuation tokens allowed in n-grams (default is 1).")
    parser.add_argument('--no_punctuation', action='store_true', help="Toggle punctuation filtering. If set, punctuation will be removed from tokens.")
    
    # Parse arguments
    args = parser.parse_args()

    # Process the JSONL file with the specified role filter
    lines = process_jsonl(args.filename, args.role_filter, no_punctuation=args.no_punctuation)

    # Count n-grams with the provided arguments
    ngrams = count_ngrams(lines, min_length=args.min_ngram, max_length=args.max_ngram,
                          stopword_limit=args.stopword_limit, punctuation_limit=args.punctuation_limit,
                          no_punctuation=args.no_punctuation)

    # Display results (for demonstration, you could extend this to save results to a file, etc.)
    print("\nN-gram Counts:")
    for n, counts in ngrams.items():
        print(f"\n{n}-grams:")
        for ngram, count in counts.most_common(10):  # Display top 10 n-grams for each length
            print(f"{' '.join(ngram)}: {count}")

if __name__ == "__main__":
    main()
