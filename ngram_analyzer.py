import collections
import re
import json
import argparse
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import string
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

# Precompiled regex patterns for speed
RE_APOSTROPHE = re.compile(r"’|‘|`")
RE_QUOTES = re.compile(r"['\"]+")
RE_NEWLINES = re.compile(r"[\n\t\r]+")
RE_EXTRA_SPACES = re.compile(r"\s+")
RE_NO_PUNCT = re.compile(r"[^\w\s']")
RE_WORDS_NOPUNCT = re.compile(r"\b[\w']+\b")
RE_WORDS_PUNCT = re.compile(r"[\w']+|[^\w\s]")
RE_CONTRACTIONS = re.compile(r"\b(I|you|we|they|he|she|it)'(ll|ve|re|d|m)\b", re.IGNORECASE)

def preprocess_text(text):
    """Preprocess text with additional token modifications."""
    # Normalize different types of apostrophes and quotes
    text = RE_APOSTROPHE.sub("'", text)
    text = RE_QUOTES.sub('"', text)
    
    # Handle common contractions (e.g., I'll -> I will)
    contraction_map = {
        "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
        "he'll": "he will", "she'll": "she will", "it'll": "it will",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "he's": "he is", "she's": "she is", "it's": "it is",
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would"
    }
    for contraction, expanded in contraction_map.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text, flags=re.IGNORECASE)
    
    # Normalize newlines and extra spaces
    text = RE_NEWLINES.sub(" ", text)
    text = RE_EXTRA_SPACES.sub(" ", text).strip()
    
    return text

def get_line_count(filename):
    """Get the line count of a file quickly using OS tools."""
    system = platform.system()
    try:
        if system in ["Linux", "Darwin"]:
            result = subprocess.run(
                ["wc", "-l", filename],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.split()[0])
        elif system == "Windows":
            result = subprocess.run(
                ["find", "/c", "/v", "", filename],
                capture_output=True,
                text=True,
                check=True,
                shell=True
            )
            return int(result.stdout.split()[-1])
    except Exception:
        return None
    return None

def tokenize(text, no_punctuation=False):
    """Fast tokenization with optional punctuation stripping."""
    # Apply preprocessing first
    text = preprocess_text(text)
    
    if no_punctuation:
        text = RE_NO_PUNCT.sub('', text)
        return RE_WORDS_NOPUNCT.findall(text.lower())
    else:
        return RE_WORDS_PUNCT.findall(text.lower())

def count_ngrams(lines, min_length=3, max_length=5, stopword_limit=1, punctuation_limit=1, no_punctuation=False, similarity_threshold=0.85):
    """Count n-grams with filtering and deduplication."""
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    # Precompute hash buckets for deduplication
    seen = {length: [] for length in lengths}
    for line in lines:
        words = tokenize(line, no_punctuation=no_punctuation)
        if len(words) < min_length:
            continue
        for n in lengths:
            for i in range(len(words) - n + 1):
                current_slice = tuple(words[i:i + n])
                stopwords_in_ngram = sum(1 for w in current_slice if w in STOP_WORDS)
                if stopwords_in_ngram > stopword_limit:
                    continue
                punctuation_in_ngram = sum(1 for w in current_slice if w in string.punctuation)
                if punctuation_in_ngram > punctuation_limit:
                    continue
                # Fast deduplication: compare against only a few recent entries
                skip = False
                for kept in seen[n][-50:]:  # only check last 50 seen
                    if _similar_enough(current_slice, kept, similarity_threshold):
                        skip = True
                        break
                if skip:
                    continue
                seen[n].append(current_slice)
                ngrams[n][current_slice] += 1
    return ngrams

def _similar_enough(ngram1, ngram2, threshold):
    """Lightweight similarity check: Jaccard overlap on tokens."""
    set1, set2 = set(ngram1), set(ngram2)
    overlap = len(set1 & set2) / max(len(set1 | set2), 1)
    return overlap >= threshold

def process_jsonl(filename, role_filter):
    """Stream JSONL and yield conversation text."""
    try:
        with open(filename, 'r', encoding="utf-8") as f:
            total_lines = get_line_count(filename)
            tqdm_kwargs = {"desc": "Processing JSONL", "unit": "line"}
            if total_lines is not None:
                tqdm_kwargs["total"] = total_lines
            with tqdm(**tqdm_kwargs) as pbar:
                for line in f:
                    try:
                        json_obj = json.loads(line)
                    except json.JSONDecodeError:
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
    parser = argparse.ArgumentParser(description="Process JSONL file and extract n-grams.")
    parser.add_argument('filename', type=str, help="Path to the JSONL file")
    parser.add_argument('--role_filter', type=str, nargs='+', default=['gpt'], choices=['gpt', 'system', 'all'],
                        help="Role(s) to filter by")
    parser.add_argument('--min_ngram', type=int, default=3)
    parser.add_argument('--max_ngram', type=int, default=5)
    parser.add_argument('--stopword_limit', type=int, default=1)
    parser.add_argument('--punctuation_limit', type=int, default=1)
    parser.add_argument('--no_punctuation', action='store_true')
    parser.add_argument('--similarity_threshold', type=float, default=0.85)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    lines = list(process_jsonl(args.filename, args.role_filter))
    ngrams = count_ngrams(
        lines,
        min_length=args.min_ngram,
        max_length=args.max_ngram,
        stopword_limit=args.stopword_limit,
        punctuation_limit=args.punctuation_limit,
        no_punctuation=args.no_punctuation,
        similarity_threshold=args.similarity_threshold
    )

    print("\nN-gram Counts:")
    output_lines = []
    for n, counts in ngrams.items():
        print(f"\n{n}-grams:")
        output_lines.append(f"\n{n}-grams:")
        for ngram, count in counts.most_common(10):
            ngram_str = ' '.join(ngram)
            print(f"{ngram_str}: {count}")
            output_lines.append(f"{ngram_str}: {count}")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()