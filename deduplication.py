import os
import sys
import json
import hashlib
import logging
import argparse
import jsonlines
import numpy as np
import faiss
from rensa import RMinHash
from sentence_transformers import SentenceTransformer, util
import contextlib

# Suppress noisy transformers/tokenizers output BEFORE imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging: ERRORs only, file only, NO console output
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log")]
)

# Suppress stdout/stderr context manager
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class Deduplication:
    def __init__(self, threshold=0.96, num_perm=256, seed=0, shingle_k=10, semantic_threshold=0.95):
        self.duplicate_count = 0
        self.unique_conversations = set()
        self.threshold = threshold
        self.num_perm = num_perm
        self.seed = seed
        self.shingle_k = shingle_k
        self.semantic_threshold = semantic_threshold

        # Load semantic model silently
        with suppress_stdout_stderr():
            self.semantic_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    def perform_sha256_deduplication(self, input_file, output_file, update_status, update_progress):
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:
                reader = jsonlines.Reader(f_in)
                total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
                f_in.seek(0)
                for i, conversation in enumerate(reader):
                    if i % 10 == 0:
                        update_progress(i, total_lines)
                    conversation_id = self.generate_sha256_hash(conversation)
                    if conversation_id in self.unique_conversations:
                        self.duplicate_count += 1
                        continue
                    self.unique_conversations.add(conversation_id)
                    f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')

            update_status(f"SHA-256 Deduplication complete. Duplicates found: {self.duplicate_count}. Output: {output_file}")
        except Exception as e:
            logging.error(f"Error during SHA-256 deduplication: {e}", exc_info=True)

    def perform_min_hash_deduplication(self, input_file, output_file, update_status, update_progress):
        try:
            # Load conversations
            with open(input_file, 'r', encoding='utf-8') as f_in:
                reader = jsonlines.Reader(f_in)
                conversations = list(reader)
            total = len(conversations)

            update_progress(0, total)

            # Extract texts
            convo_texts = [self.extract_convo_text(c) for c in conversations]

            # Generate MinHashes
            minhashes = []
            for i, text in enumerate(convo_texts):
                shingles = self.shingle_text(text, k=self.shingle_k)
                m = self.generate_min_hash(shingles)
                minhashes.append(m)
                if i % 100 == 0:
                    update_progress(i, total)

            # Batch encode all texts (fast!)
            with suppress_stdout_stderr():
                embeddings = self.semantic_model.encode(convo_texts, convert_to_tensor=False, batch_size=128, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')

            # Normalize embeddings for cosine similarity using inner product
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            unique_indices = []
            duplicate_count = 0
            visited = set()

            for i, m in enumerate(minhashes):
                if i in visited:
                    continue

                # Find candidates by MinHash jaccard threshold
                candidates = [j for j in range(i+1, total) if j not in visited and m.jaccard(minhashes[j]) >= self.threshold]

                if candidates:
                    # Search semantic similarity for candidate embeddings
                    D, I = index.search(embeddings[i:i+1], len(candidates))
                    for idx, sim in zip(I[0], D[0]):
                        if idx in candidates and sim >= self.semantic_threshold:
                            visited.add(idx)
                            duplicate_count += 1

                unique_indices.append(i)

            # Write out unique conversations
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for idx in unique_indices:
                    f_out.write(json.dumps(conversations[idx], ensure_ascii=False) + '\n')

            update_status(f"MinHash + Semantic Deduplication complete. Duplicates found: {duplicate_count}. Output: {output_file}")

        except Exception as e:
            logging.error(f"Error during MinHash deduplication: {e}", exc_info=True)

    @staticmethod
    def extract_convo_text(conversation):
        return ''.join(turn.get('value', '') for turn in conversation.get('conversations', []) if turn.get('value') is not None)

    def generate_min_hash(self, shingles):
        m = RMinHash(self.num_perm, seed=self.seed)
        for shingle in shingles:
            m.update(list(shingle))
        return m

    @staticmethod
    def shingle_text(text, k=10):
        return set(text[i:i + k] for i in range(len(text) - k + 1))

    @staticmethod
    def generate_sha256_hash(conversation):
        conversation_text = ''.join(turn.get('value', '') for turn in conversation.get('conversations', []) if turn.get('value') is not None)
        return hashlib.sha256(conversation_text.encode('utf-8')).hexdigest()

# Silent callbacks for UI integration (no print or logging here)
def update_status(message):
    pass

def update_progress(current, total):
    pass

def main():
    parser = argparse.ArgumentParser(description="Deduplication tool for conversations")
    parser.add_argument('input_file', type=str, help="Input JSONL file with conversations")
    parser.add_argument('output_file', type=str, help="Output JSONL file for deduplicated conversations")
    parser.add_argument('--method', choices=['sha256', 'minhash'], default='sha256', help="Deduplication method (default: sha256)")
    parser.add_argument('--threshold', type=float, default=0.96, help="MinHash similarity threshold (default: 0.96)")
    parser.add_argument('--num_perm', type=int, default=256, help="Number of permutations for MinHash (default: 256)")
    parser.add_argument('--seed', type=int, default=0, help="Seed for MinHash (default: 0)")
    parser.add_argument('--shingle_k', type=int, default=10, help="Shingle length for MinHash (default: 10)")
    parser.add_argument('--semantic_threshold', type=float, default=0.95, help="Cosine similarity threshold for semantic dedup (default: 0.95)")

    args = parser.parse_args()

    dedup = Deduplication(threshold=args.threshold, num_perm=args.num_perm,
                          seed=args.seed, shingle_k=args.shingle_k,
                          semantic_threshold=args.semantic_threshold)

    if args.method == 'sha256':
        dedup.perform_sha256_deduplication(args.input_file, args.output_file, update_status, update_progress)
    else:
        dedup.perform_min_hash_deduplication(args.input_file, args.output_file, update_status, update_progress)

if __name__ == "__main__":
    main()