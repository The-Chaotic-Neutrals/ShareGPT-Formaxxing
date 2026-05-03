import os
import json
import secrets
import re
from collections import deque

class LineMancerCore:
    def __init__(self):
        # Base directory of this script
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Repo root (parent of App directory)
        self.repo_root = os.path.dirname(os.path.dirname(self.base_dir))
        # Unified output directory
        self.outputs_dir = os.path.join(self.repo_root, "Outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

    def _cryptographic_shuffle(self, items):
        rng = secrets.SystemRandom()
        for i in range(len(items) - 1, 0, -1):
            j = rng.randrange(i + 1)
            items[i], items[j] = items[j], items[i]

    def _normalize_text(self, value):
        text = re.sub(r"\s+", " ", value.lower())
        return re.sub(r"[^a-z0-9 ]", "", text).strip()

    def _collect_text_fragments(self, value, fragments, max_fragments=3):
        if len(fragments) >= max_fragments:
            return
        if isinstance(value, str):
            normalized = self._normalize_text(value)
            if normalized:
                fragments.append(" ".join(normalized.split()[:12]))
            return
        if isinstance(value, list):
            for item in value:
                if len(fragments) >= max_fragments:
                    break
                self._collect_text_fragments(item, fragments, max_fragments)
            return
        if isinstance(value, dict):
            for key in sorted(value.keys()):
                if len(fragments) >= max_fragments:
                    break
                self._collect_text_fragments(value[key], fragments, max_fragments)

    def _line_signature(self, obj):
        if isinstance(obj, dict):
            keys = tuple(sorted(obj.keys())[:10])
            fragments = []
            for preferred_key in ("instruction", "prompt", "input", "question", "text", "messages", "conversations"):
                if preferred_key in obj:
                    self._collect_text_fragments(obj[preferred_key], fragments)
                    if fragments:
                        break
            if not fragments:
                self._collect_text_fragments(obj, fragments)
            prefix = "|".join(fragments)[:160]
            return ("dict", keys, prefix)

        if isinstance(obj, list):
            fragments = []
            self._collect_text_fragments(obj, fragments)
            return ("list", len(obj), "|".join(fragments)[:160])

        if isinstance(obj, str):
            normalized = self._normalize_text(obj)
            return ("str", normalized[:160])

        return (type(obj).__name__, str(obj)[:80])

    def _line_tokens(self, obj):
        fragments = []
        self._collect_text_fragments(obj, fragments, max_fragments=8)
        if not fragments:
            return set()
        text = " ".join(fragments)
        tokens = [tok for tok in text.split() if len(tok) > 2]
        return set(tokens[:64])

    def _adjacent_clump_penalty(self, left_entry, right_entry):
        if left_entry is None or right_entry is None:
            return 0

        penalty = 0
        if left_entry["signature"] == right_entry["signature"]:
            penalty += 1000

        token_overlap = len(left_entry["tokens"] & right_entry["tokens"])
        penalty += min(token_overlap, 16)
        return penalty

    def _swap_improvement(self, entries, i, j):
        affected_pair_indices = {
            i - 1, i, j - 1, j
        }
        affected_pair_indices = [idx for idx in affected_pair_indices if 0 <= idx < len(entries) - 1]

        before = 0
        for idx in affected_pair_indices:
            before += self._adjacent_clump_penalty(entries[idx], entries[idx + 1])

        def at(index):
            if index == i:
                return entries[j]
            if index == j:
                return entries[i]
            return entries[index]

        after = 0
        for idx in affected_pair_indices:
            after += self._adjacent_clump_penalty(at(idx), at(idx + 1))

        return before - after

    def _declump_by_swaps(self, entries, search_span=220):
        if len(entries) < 3:
            return

        for i in range(1, len(entries) - 1):
            if self._adjacent_clump_penalty(entries[i - 1], entries[i]) == 0:
                continue

            best_j = None
            best_gain = 0
            end = min(len(entries), i + search_span)
            for j in range(i + 1, end):
                gain = self._swap_improvement(entries, i, j)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j is not None:
                entries[i], entries[best_j] = entries[best_j], entries[i]

        for i in range(len(entries) - 1, 0, -1):
            if self._adjacent_clump_penalty(entries[i - 1], entries[i]) == 0:
                continue

            best_j = None
            best_gain = 0
            start = max(0, i - search_span)
            for j in range(start, i):
                gain = self._swap_improvement(entries, i, j)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j is not None:
                entries[i], entries[best_j] = entries[best_j], entries[i]

    def _anti_clump_order(self, entries):
        rng = secrets.SystemRandom()
        pool = []
        for raw_line, obj in entries:
            pool.append({
                "line": raw_line,
                "signature": self._line_signature(obj),
                "tokens": self._line_tokens(obj),
            })

        self._cryptographic_shuffle(pool)

        output = []
        recent_window = deque()
        recent_token_counts = {}
        recent_signature_counts = {}
        last_signature = None

        window_size = 8
        sample_size = 48

        while pool:
            k = min(sample_size, len(pool))
            if k == len(pool):
                candidate_indices = range(len(pool))
            else:
                candidate_indices = rng.sample(range(len(pool)), k)

            best_idx = None
            best_score = None

            for idx in candidate_indices:
                candidate = pool[idx]
                signature = candidate["signature"]

                same_as_last = 1 if signature == last_signature else 0
                seen_recently = recent_signature_counts.get(signature, 0)
                token_overlap = 0
                for token in candidate["tokens"]:
                    token_overlap += recent_token_counts.get(token, 0)

                score = (same_as_last, seen_recently, token_overlap, rng.random())
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx

            chosen = pool[best_idx]
            pool[best_idx] = pool[-1]
            pool.pop()

            output.append(chosen)
            last_signature = chosen["signature"]

            recent_window.append((chosen["signature"], chosen["tokens"]))
            recent_signature_counts[chosen["signature"]] = recent_signature_counts.get(chosen["signature"], 0) + 1
            for token in chosen["tokens"]:
                recent_token_counts[token] = recent_token_counts.get(token, 0) + 1

            if len(recent_window) > window_size:
                old_signature, old_tokens = recent_window.popleft()
                recent_signature_counts[old_signature] -= 1
                if recent_signature_counts[old_signature] <= 0:
                    del recent_signature_counts[old_signature]
                for token in old_tokens:
                    recent_token_counts[token] -= 1
                    if recent_token_counts[token] <= 0:
                        del recent_token_counts[token]

        self._declump_by_swaps(output)
        return [entry["line"] for entry in output]

    def split_jsonl(self, input_path, lines_per_file):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if lines_per_file <= 0:
            raise ValueError("Lines per file must be > 0")

        prefix = os.path.splitext(os.path.basename(input_path))[0]
        index = 1
        buffer = []

        with open(input_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    buffer.append(line)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSON line {i}")
                if len(buffer) >= lines_per_file:
                    self._write_split_file(self.outputs_dir, prefix, index, buffer)
                    index += 1
                    buffer = []
            if buffer:
                self._write_split_file(self.outputs_dir, prefix, index, buffer)

        print(f"[LineMancer] Split into {index} files with prefix '{prefix}' in '{self.outputs_dir}'")
        return index

    def _write_split_file(self, directory, prefix, index, buffer):
        out_path = os.path.join(directory, f"{prefix}_split_{index}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(buffer) + "\n")
        print(f"[LineMancer] Wrote {len(buffer)} lines to {out_path}")

    def merge_jsonl(self, input_files=None, input_dir=None, prefix=None, output_filename=None):
        """
        Merge multiple JSONL files into one.
        
        Parameters:
        - input_files: list of explicit file paths to merge (if given, overrides input_dir/prefix)
        - input_dir: directory to look for files (default: self.split_dir)
        - prefix: prefix filter for files in input_dir
        - output_filename: name of output merged file (in self.merge_dir)
        
        Returns:
        - output_path, total_lines
        """
        if input_files:
            # Validate files exist
            files = []
            for f in input_files:
                if not os.path.isfile(f):
                    print(f"[Warning] Input file not found: {f}, skipping")
                else:
                    files.append(f)
            if not files:
                raise FileNotFoundError("No valid input files provided in input_files list.")
            # Sort files by extracted index if possible, else lex order
            def extract_index(filepath):
                fname = os.path.basename(filepath)
                m = re.search(r"_split_(\d+)\.jsonl$", fname)
                return int(m.group(1)) if m else 0
            files.sort(key=extract_index)
            input_paths = files
            # For output filename prefix extraction
            if prefix is None and len(files) > 0:
                prefix = os.path.splitext(os.path.basename(files[0]))[0].rsplit('_', 1)[0]
        else:
            input_dir = input_dir or self.outputs_dir

            if prefix is None and output_filename:
                prefix = os.path.splitext(output_filename)[0]

            if prefix:
                pattern = re.compile(re.escape(prefix) + r"_split_\d+\.jsonl$")
                files = [f for f in os.listdir(input_dir) if pattern.match(f)]
                if not files:
                    raise FileNotFoundError(f"No split files with prefix '{prefix}' found in {input_dir}")
            else:
                files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
                if not files:
                    raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

            def extract_index(filename):
                m = re.search(r"_split_(\d+)\.jsonl$", filename)
                return int(m.group(1)) if m else 0

            files.sort(key=extract_index)
            input_paths = [os.path.join(input_dir, f) for f in files]

        if output_filename is None:
            output_filename = f"{prefix}_merged.jsonl" if prefix else "merged.jsonl"
        if not output_filename.endswith(".jsonl"):
            output_filename += ".jsonl"
        output_path = os.path.join(self.outputs_dir, output_filename)

        total_lines = 0
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for path in input_paths:
                if not os.path.isfile(path):
                    print(f"[Warning] File not found: {path}, skipping")
                    continue
                with open(path, 'r', encoding='utf-8') as infile:
                    for i, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            outfile.write(line + "\n")
                            total_lines += 1
                        except json.JSONDecodeError:
                            print(f"[Warning] Skipping invalid JSON line {i} in {path}")

        print(f"[LineMancer] Merged {total_lines} lines into {output_path}")
        return output_path, total_lines

    def shuffle_jsonl(self, input_path, anti_clump=False):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        prefix = os.path.splitext(os.path.basename(input_path))[0]
        suffix = "_shuffled_anticlump.jsonl" if anti_clump else "_shuffled.jsonl"
        output_path = os.path.join(self.outputs_dir, f"{prefix}{suffix}")

        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        self._cryptographic_shuffle(lines)

        ordered_lines = []
        parsed_entries = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                parsed_entries.append((stripped, obj))
            except json.JSONDecodeError:
                print(f"[Warning] Skipping invalid JSON line {i} during shuffle.")

        if anti_clump:
            ordered_lines = self._anti_clump_order(parsed_entries)
        else:
            ordered_lines = [line for line, _ in parsed_entries]

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in ordered_lines:
                outfile.write(line + "\n")

        mode_label = "with anti-clump" if anti_clump else "without anti-clump"
        print(f"[LineMancer] Shuffled {len(ordered_lines)} lines {mode_label} into {output_path}")
        return output_path
