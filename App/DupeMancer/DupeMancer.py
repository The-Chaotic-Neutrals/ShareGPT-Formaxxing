import os
import sys
import json
import hashlib
import argparse
import random
import numpy as np
from rensa import RMinHash
import contextlib
import tempfile
import unicodedata
from pathlib import Path
from collections import Counter

# Suppress noisy transformers/tokenizers output BEFORE imports
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def dataset_output_paths(input_files, output_dir):
    directory = Path(output_dir)
    reserved = {Path(name).resolve() for name in input_files}
    outputs = []
    for name in input_files:
        stem = Path(name).stem
        output = directory / f"{stem}-deduplicated.jsonl"
        suffix = 2
        while output.resolve() in reserved:
            output = directory / f"{stem}-{suffix}-deduplicated.jsonl"
            suffix += 1
        reserved.add(output.resolve())
        outputs.append(output)
    return outputs


class Deduplication:
    def __init__(
        self,
        threshold=0.90, # 👈 dropped by 0.06
        num_perm=256,
        seed=0,
        shingle_k=10,
        semantic_threshold=0.89, # 👈 dropped by 0.06
        prefix_bits=16,  
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

        self.semantic_model = None

    def _get_semantic_model(self):
        if self.semantic_model is None:
            from sentence_transformers import SentenceTransformer

            with suppress_stdout_stderr():
                self.semantic_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        return self.semantic_model

    @staticmethod
    def _read_rows(input_file, on_bytes_read=None):
        def unique_keys(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError(f"Duplicate JSON key {key!r}.")
                obj[key] = value
            return obj

        bytes_read = 0
        with open(input_file, "rb") as source:
            for number, raw in enumerate(source, 1):
                bytes_read += len(raw)
                if on_bytes_read is not None and number % 10 == 0:
                    on_bytes_read(bytes_read)
                if number == 1:
                    raw = raw.removeprefix(b"\xef\xbb\xbf")
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a JSON object.")
                except ValueError as exc:
                    raise ValueError(f"{input_file}, line {number}: {exc}") from exc
                yield raw, row

    @staticmethod
    def _write_rows(input_file, output_file, rows):
        output = Path(output_file)
        if output.resolve() == Path(input_file).resolve():
            raise ValueError("Choose an output path different from the input dataset.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as out:
                temporary = Path(out.name)
                for raw in rows:
                    out.write(raw)
                    if not raw.endswith(b"\n"):
                        out.write(b"\n")
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def perform_sha256_deduplication(self, input_file, output_file, update_status, update_progress, randomize=False):
        file_size = os.path.getsize(input_file)
        new_hashes = set()
        retained = []
        duplicate_count = 0
        for raw, row in self._read_rows(input_file, lambda size: update_progress(size, file_size)):
            digest = self.generate_sha256_hash(row)
            if digest in self.unique_conversations or digest in new_hashes:
                duplicate_count += 1
            else:
                new_hashes.add(digest)
                retained.append(raw)

        if randomize:
            random.shuffle(retained)
        self._write_rows(input_file, output_file, retained)
        self.unique_conversations.update(new_hashes)
        self.duplicate_count = duplicate_count
        update_progress(file_size, file_size)
        update_status(f"SHA-256 complete. Duplicates found: {duplicate_count}. Output: {output_file}")

    def perform_min_hash_deduplication(self, input_file, output_file, update_status, update_progress, randomize=False):
        rows = list(self._read_rows(input_file))
        total = len(rows) * 2
        update_progress(0, total)
        seen_hashes = set()
        dropped = set()
        text_indices = []
        texts = []
        features = {}
        buckets = {}

        for index, (_, row) in enumerate(rows):
            digest = self.generate_sha256_hash(row)
            if digest in seen_hashes:
                dropped.add(index)
            else:
                seen_hashes.add(digest)
                turns = self._comparison_turns(row)
                if turns is not None:
                    roles, values = zip(*turns)
                    text = "\n".join(f"{role}: {value}" for role, value in turns)
                    prefix = self.simhash(text) >> (64 - self.prefix_bits)
                    chat_key = "conversations" if "conversations" in row else "messages"
                    context = json.dumps(
                        {key: value for key, value in row.items() if key != chat_key},
                        sort_keys=True, separators=(",", ":"),
                    )
                    bucket = (chat_key, roles, context, prefix)
                    buckets.setdefault(bucket, []).append(index)
                    features[index] = (
                        self.generate_min_hash(self.shingle_text(text, self.shingle_k)),
                        values,
                    )
                    text_indices.append(index)
                    texts.append(text)
            if index % 100 == 0:
                update_progress(index, total)
        update_progress(len(rows), total)

        if any(len(indices) > 1 for indices in buckets.values()):
            import faiss

            with suppress_stdout_stderr():
                embeddings = self._get_semantic_model().encode(
                    texts, convert_to_tensor=False, batch_size=128, show_progress_bar=False,
                )
            embeddings = np.asarray(embeddings, dtype="float32", order="C")
            faiss.normalize_L2(embeddings)
            embedding_index = {index: position for position, index in enumerate(text_indices)}
            processed = 0
            for indices in buckets.values():
                for position, index in enumerate(indices):
                    processed += 1
                    if processed % 100 == 0:
                        update_progress(len(rows) + processed, total)
                    if index in dropped:
                        continue
                    minhash, turns = features[index]
                    candidates = []
                    for other in indices[position + 1:]:
                        if other in dropped:
                            continue
                        other_minhash, other_turns = features[other]
                        if minhash.jaccard(other_minhash) < self.threshold:
                            continue
                        if not all(self._turns_match(left, right) for left, right in zip(turns, other_turns)):
                            continue
                        candidates.append(other)
                    if candidates:
                        positions = [embedding_index[other] for other in candidates]
                        scores = embeddings[embedding_index[index]] @ embeddings[positions].T
                        dropped.update(other for other, score in zip(candidates, scores) if score >= self.semantic_threshold)

        retained = [raw for index, (raw, _) in enumerate(rows) if index not in dropped]
        if randomize:
            random.shuffle(retained)
        self._write_rows(input_file, output_file, retained)
        self.duplicate_count = len(dropped)
        update_progress(total, total)
        update_status(f"MinHash + Semantic complete. Duplicates found: {len(dropped)}. Output: {output_file}")

    def _turns_match(self, left, right):
        if left == right:
            return True
        first = self.shingle_text(left, self.shingle_k)
        second = self.shingle_text(right, self.shingle_k)
        return bool(first and second) and len(first & second) / len(first | second) >= self.threshold

    @staticmethod
    def _comparison_turns(row):
        turns = row.get("conversations", row.get("messages"))
        if not isinstance(turns, list) or not turns:
            return None
        result = []
        for turn in turns:
            if not isinstance(turn, dict):
                return None
            if set(turn) == {"from", "value"}:
                role, value = turn["from"], turn["value"]
            elif set(turn) == {"role", "content"}:
                role, value = turn["role"], turn["content"]
            else:
                # Structured turns can carry fields that a text comparison does not see.
                return None
            if not isinstance(role, str) or not isinstance(value, str):
                return None
            result.append((role, unicodedata.normalize("NFKC", value.strip())))
        return tuple(result)

    def generate_min_hash(self, shingles):
        signature = RMinHash(self.num_perm, seed=self.seed)
        signature.update(list(shingles))
        return signature

    @staticmethod
    def shingle_text(text, k=10):
        if not text or len(text) < k:
            return {text} if text else set()
        return {text[i:i + k] for i in range(len(text) - k + 1)}

    @staticmethod
    def generate_sha256_hash(row):
        canonical_row = json.dumps(row, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_row.encode("utf-8")).hexdigest()

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
    parser = argparse.ArgumentParser(description="DupeMancer dataset deduplication")
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
