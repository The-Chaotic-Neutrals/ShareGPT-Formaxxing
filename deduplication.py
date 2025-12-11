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
from sentence_transformers import SentenceTransformer
import contextlib
from collections import Counter

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
    def __init__(
        self,
        threshold=0.96,
        num_perm=256,
        seed=0,
        shingle_k=10,
        semantic_threshold=0.95,
        prefix_bits=16,  # 👈 configurable SimHash prefix bits
    ):
        self.duplicate_count = 0
        # Note: keep this across files if you want cross-file exact dedup.
        self.unique_conversations = set()
        self.threshold = threshold
        self.num_perm = num_perm
        self.seed = seed
        self.shingle_k = shingle_k
        self.semantic_threshold = semantic_threshold
        self.prefix_bits = prefix_bits

        # Load semantic model silently
        with suppress_stdout_stderr():
            self.semantic_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # ---------- SHA-256 DEDUP (single pass, byte-based progress, no tell()) ----------

    def perform_sha256_deduplication(self, input_file, output_file, update_status, update_progress):
        try:
            file_size = os.path.getsize(input_file)
            if file_size == 0:
                update_status(f"Input file is empty. Nothing to deduplicate.")
                return

            # Reset per-run duplicate counter
            self.duplicate_count = 0

            bytes_read = 0

            with open(input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:

                for i, line in enumerate(f_in):
                    # Track bytes read for progress (avoid tell())
                    bytes_read += len(line.encode('utf-8'))

                    if not line.strip():
                        continue

                    try:
                        conversation = json.loads(line)
                    except json.JSONDecodeError:
                        logging.error("Invalid JSON line encountered, skipping.", exc_info=True)
                        continue

                    conversation_id = self.generate_sha256_hash(conversation)
                    if conversation_id in self.unique_conversations:
                        self.duplicate_count += 1
                        continue

                    self.unique_conversations.add(conversation_id)
                    f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')

                    # Byte-based progress
                    if i % 10 == 0:
                        update_progress(bytes_read, file_size)

            # ensure final 100%
            update_progress(file_size, file_size)
            update_status(
                f"SHA-256 Deduplication complete. Duplicates found: {self.duplicate_count}. "
                f"Output: {output_file}"
            )
        except Exception as e:
            logging.error(f"Error during SHA-256 deduplication: {e}", exc_info=True)

    # ---------- MINHASH + SIMHASH BUCKETING + SEMANTIC COSINE ----------

    def perform_min_hash_deduplication(self, input_file, output_file, update_status, update_progress):
        try:
            # Load conversations
            with open(input_file, 'r', encoding='utf-8') as f_in:
                reader = jsonlines.Reader(f_in)
                conversations = list(reader)
            num_items = len(conversations)

            if num_items == 0:
                update_status(f"No conversations found in {input_file}.")
                return

            # Two-phase virtual progress:
            #   total_steps = 2 * num_items
            #   Phase 1: feature computation      -> current in [0, num_items]
            #   Phase 2: dedup + streaming writes -> current in [num_items, 2 * num_items]
            total_steps = num_items * 2
            update_progress(0, total_steps)

            # Extract texts
            convo_texts = [self.extract_convo_text(c) for c in conversations]

            # Generate MinHashes + SimHashes
            minhashes = []
            simhashes = []
            for i, text in enumerate(convo_texts):
                shingles = self.shingle_text(text, k=self.shingle_k)
                m = self.generate_min_hash(shingles)
                minhashes.append(m)

                sh = self.simhash(text)
                simhashes.append(sh)

                if i % 100 == 0:
                    # Phase 1 progress: feature computation
                    update_progress(i, total_steps)

            # Ensure we at least mark the end of phase 1
            update_progress(num_items, total_steps)

            # Batch encode all texts (semantic embeddings)
            with suppress_stdout_stderr():
                embeddings = self.semantic_model.encode(
                    convo_texts,
                    convert_to_tensor=False,
                    batch_size=128,
                    show_progress_bar=False,
                )
            embeddings = np.array(embeddings, dtype='float32')

            # Normalize embeddings for cosine similarity using inner product
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            # Bucket indices by SimHash prefix for cheaper candidate search
            PREFIX_BITS = self.prefix_bits
            prefix_mask = (1 << PREFIX_BITS) - 1
            buckets = {}
            for idx, sh in enumerate(simhashes):
                key = (sh >> (64 - PREFIX_BITS)) & prefix_mask
                buckets.setdefault(key, []).append(idx)

            duplicate_count = 0
            visited = set()
            processed = 0  # number of items we've decided on in phase 2

            # Stream write unique conversations as we finalize them
            with open(output_file, 'w', encoding='utf-8') as f_out:
                # Near-duplicate detection within each SimHash bucket
                for key, idxs in buckets.items():
                    if len(idxs) == 1:
                        # Single-item bucket: if not visited, it's unique
                        i = idxs[0]
                        if i not in visited:
                            visited.add(i)
                            f_out.write(json.dumps(conversations[i], ensure_ascii=False) + '\n')
                        processed += 1

                        if processed % 100 == 0:
                            # Phase 2 progress: offset by num_items
                            update_progress(num_items + processed, total_steps)
                        continue

                    for local_pos, i in enumerate(idxs):
                        if i in visited:
                            # Already decided as duplicate or unique
                            processed += 1
                            if processed % 100 == 0:
                                update_progress(num_items + processed, total_steps)
                            continue

                        m_i = minhashes[i]
                        # MinHash Jaccard candidates within this bucket
                        candidates = []
                        for j in idxs[local_pos + 1:]:
                            if j in visited:
                                continue
                            if m_i.jaccard(minhashes[j]) >= self.threshold:
                                candidates.append(j)

                        if candidates:
                            # Semantic cosine filter over MinHash candidates
                            D, I = index.search(embeddings[i:i+1], len(candidates))
                            for idx_found, sim in zip(I[0], D[0]):
                                if idx_found in candidates and sim >= self.semantic_threshold:
                                    visited.add(idx_found)
                                    duplicate_count += 1

                        # i is now finalized as unique
                        visited.add(i)
                        f_out.write(json.dumps(conversations[i], ensure_ascii=False) + '\n')

                        processed += 1
                        if processed % 100 == 0:
                            update_progress(num_items + processed, total_steps)

            # Push progress to 100% at the end of phase 2
            update_progress(total_steps, total_steps)
            update_status(
                f"MinHash + SimHash + Semantic Deduplication complete. "
                f"Duplicates found: {duplicate_count}. Output: {output_file}"
            )
        except Exception as e:
            logging.error(f"Error during MinHash deduplication: {e}", exc_info=True)

    # ---------- HELPERS ----------

    @staticmethod
    def extract_convo_text(conversation):
        # Assumes ShareGPT-like {"conversations": [{"value": "..."}]}
        return ''.join(
            turn.get('value', '')
            for turn in conversation.get('conversations', [])
            if turn.get('value') is not None
        )

    def generate_min_hash(self, shingles):
        m = RMinHash(self.num_perm, seed=self.seed)
        for shingle in shingles:
            m.update(list(shingle))
        return m

    @staticmethod
    def shingle_text(text, k=10):
        if not text or len(text) < k:
            return {text} if text else set()
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    @staticmethod
    def generate_sha256_hash(conversation):
        conversation_text = ''.join(
            turn.get('value', '')
            for turn in conversation.get('conversations', [])
            if turn.get('value') is not None
        )
        return hashlib.sha256(conversation_text.encode('utf-8')).hexdigest()

    @staticmethod
    def simple_tokenize(text):
        return text.lower().split() if text else []

    def simhash(self, text, hash_bits=64):
        """
        Simple SimHash over token counts.
        Used only for bucketing, not as the final similarity decision.
        """
        tokens = self.simple_tokenize(text)
        if not tokens:
            return 0

        v = [0] * hash_bits
        counts = Counter(tokens)
        for token, count in counts.items():
            # stable token hash
            h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            for i in range(hash_bits):
                bit = (h >> i) & 1
                v[i] += count if bit else -count

        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint


# Silent callbacks for UI integration (no print or logging here)
def update_status(message):
    pass


def update_progress(current, total):
    pass


def main():
    parser = argparse.ArgumentParser(description="Deduplication tool for conversations")
    parser.add_argument('input_file', type=str, help="Input JSONL file with conversations")
    parser.add_argument('output_file', type=str, help="Output JSONL file for deduplicated conversations")
    parser.add_argument('--method', choices=['sha256', 'minhash'], default='sha256',
                        help="Deduplication method (default: sha256)")
    parser.add_argument('--threshold', type=float, default=0.96,
                        help="MinHash similarity threshold (default: 0.96)")
    parser.add_argument('--num_perm', type=int, default=256,
                        help="Number of permutations for MinHash (default: 256)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for MinHash (default: 0)")
    parser.add_argument('--shingle_k', type=int, default=10,
                        help="Shingle length for MinHash (default: 10)")
    parser.add_argument('--semantic_threshold', type=float, default=0.95,
                        help="Cosine similarity threshold for semantic dedup (default: 0.95)")
    parser.add_argument('--prefix_bits', type=int, default=16,
                        help="SimHash prefix bits for bucketing (default: 16)")

    args = parser.parse_args()

    dedup = Deduplication(
        threshold=args.threshold,
        num_perm=args.num_perm,
        seed=args.seed,
        shingle_k=args.shingle_k,
        semantic_threshold=args.semantic_threshold,
        prefix_bits=args.prefix_bits,
    )

    # Optional CLI tqdm integration; UI can still pass its own callbacks
    try:
        from tqdm.auto import tqdm
        use_tqdm = sys.stdout.isatty()
    except Exception:
        tqdm = None
        use_tqdm = False

    if use_tqdm:
        state = {"pbar": None, "last_current": 0}

        def cli_update_progress(current, total):
            if state["pbar"] is None:
                # Use bytes for sha256, "units" for minhash
                unit = "B" if args.method == "sha256" else "it"
                unit_scale = True if args.method == "sha256" else False
                state["pbar"] = tqdm(total=total, unit=unit, unit_scale=unit_scale)
                state["last_current"] = 0
            pbar = state["pbar"]
            if pbar.total != total and total is not None:
                pbar.total = total
            delta = current - state["last_current"]
            if delta > 0:
                pbar.update(delta)
                state["last_current"] = current

        def cli_update_status(msg):
            if state["pbar"] is not None:
                state["pbar"].write(msg)
            else:
                print(msg, file=sys.stderr)

        if args.method == 'sha256':
            dedup.perform_sha256_deduplication(
                args.input_file, args.output_file,
                cli_update_status, cli_update_progress
            )
        else:
            dedup.perform_min_hash_deduplication(
                args.input_file, args.output_file,
                cli_update_status, cli_update_progress
            )

        if state["pbar"] is not None:
            state["pbar"].close()
    else:
        if args.method == 'sha256':
            dedup.perform_sha256_deduplication(
                args.input_file, args.output_file,
                update_status, update_progress
            )
        else:
            dedup.perform_min_hash_deduplication(
                args.input_file, args.output_file,
                update_status, update_progress
            )


if __name__ == "__main__":
    main()
