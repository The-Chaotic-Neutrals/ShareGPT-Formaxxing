import json
import os
import random
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer


class TokenMaxxerCore:
    CONFIG_FILE = "tokenmaxxer_config.json"

    def __init__(self):
        self.tokenizer = None
        self.recent_models = []
        self.model_name = ""
        self.file_path = ""

    # ---------------- Config ----------------
    def load_config(self):
        default_model = "meta-llama/Llama-2-7b-hf"
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                last = config.get("last_model", default_model)
                recent = config.get("recent_models", [])
                if last not in recent:
                    recent.insert(0, last)
                return last, recent
        return default_model, []

    def save_config(self, model):
        if not model:
            return

        recent = []
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    recent = data.get("recent_models", [])
                except Exception:
                    pass

        if model in recent:
            recent.remove(model)
        recent.insert(0, model)
        recent = recent[:10]

        with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_model": model, "recent_models": recent}, f, indent=2)

    # ---------------- Tokenizer ----------------
    def load_tokenizer(self, model_repo: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model_name = model_repo
        self.save_config(model_repo)

    def has_chat_template(self):
        return bool(self.tokenizer and getattr(self.tokenizer, "chat_template", None))

    @staticmethod
    def _normalize_role(role):
        role = str(role or "").strip().lower()
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "model": "assistant",
            "bot": "assistant",
            "system": "system",
            "developer": "developer",
            "tool": "tool",
        }
        return role_map.get(role)

    def _entry_to_messages(self, entry):
        """Convert common ShareGPT-style rows into HF chat-template messages."""
        conversations = entry.get("conversations")
        if conversations is None:
            conversations = entry.get("messages", [])

        messages = []
        for message in conversations or []:
            role = self._normalize_role(
                message.get("from")
                or message.get("role")
                or message.get("speaker")
            )
            if role is None:
                continue

            content = message.get("value")
            if content is None:
                content = message.get("content")
            if content is None:
                content = message.get("text", "")

            # Most ShareGPT datasets use strings. Preserve non-string content
            # in a readable form rather than silently dropping it.
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            messages.append({"role": role, "content": content})

        return messages

    def _raw_flat_text(self, entry):
        conversations = entry.get("conversations")
        if conversations is None:
            conversations = entry.get("messages", [])

        values = []
        for message in conversations or []:
            content = message.get("value")
            if content is None:
                content = message.get("content")
            if content is None:
                content = message.get("text")
            if content is None:
                continue
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            values.append(content)
        return "\n".join(values)

    def encode_entry(self, entry):
        """
        Tokenize an entry as the loaded model would see it during chat SFT.

        Uses the tokenizer's own chat template when available. If the tokenizer
        has no chat template, falls back to the old raw-text counting method.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        if self.has_chat_template():
            messages = self._entry_to_messages(entry)
            if not messages:
                return []

            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            # Normally this is list[int], but tolerate tokenizer variants.
            # Some tokenizers return a BatchEncoding/dict (e.g. {'input_ids': [...],
            # 'attention_mask': [...]}). BatchEncoding is a UserDict, not a plain
            # dict, so isinstance(encoded, dict) is False.
            if not isinstance(encoded, (list, tuple)):
                encoded = encoded.get("input_ids", [])
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            return list(encoded)

        flat = self._raw_flat_text(entry)
        return self.tokenizer.encode(flat, add_special_tokens=False)

    def get_token_count(self, entry):
        return len(self.encode_entry(entry))

    @staticmethod
    def _approximate_length_sort(entries, bucket_tokens=128, longest_first=False, seed=42):
        """
        Approximate length ordering.

        Entries are grouped into token-width buckets (e.g. 128-token ranges),
        shuffled inside each bucket, then buckets are emitted shortest->longest
        or longest->shortest. This preserves length locality without creating a
        perfectly monotonic sample-length curriculum.
        """
        if not entries:
            return entries

        bucket_tokens = max(1, int(bucket_tokens))
        buckets = defaultdict(list)
        for item in entries:
            token_count = item[1]
            bucket_id = token_count // bucket_tokens
            buckets[bucket_id].append(item)

        rng = random.Random(seed)
        ordered = []
        for bucket_id in sorted(buckets.keys(), reverse=longest_first):
            bucket = buckets[bucket_id]
            rng.shuffle(bucket)
            ordered.extend(bucket)
        return ordered

    # ---------------- Core operations ----------------
    def analyze_file(self, file_path: str):
        raw_entries = []
        lengths = []
        errors = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    count = self.get_token_count(entry)
                    raw_entries.append(entry)
                    lengths.append(count)
                except Exception:
                    errors += 1

        if not lengths:
            return "⚠️ No valid entries to analyze."

        percentiles = np.percentile(lengths, [10, 25, 50, 75, 90, 95, 99, 100])
        longest_idx = lengths.index(max(lengths))
        mode = "HF chat template" if self.has_chat_template() else "raw-text fallback"

        return (
            f"🧮 Counting mode: {mode}\n"
            f"🔢 Percentiles: {percentiles.tolist()}\n"
            f"📏 Min: {min(lengths)} | Max: {max(lengths)}\n"
            f"📍 Longest Entry Index: {longest_idx}\n"
            f"🧪 Tokens: {lengths[longest_idx]}\n"
            f"⚠️ Skipped/invalid entries: {errors}\n"
            f"🕵️ Entry:\n{json.dumps(raw_entries[longest_idx], indent=2, ensure_ascii=False)}"
        )

    def clean_file(
        self,
        file_path: str,
        max_tokens: int,
        sort_by_length: bool = False,
        longest_first: bool = False,
        bucket_tokens: int = 128,
    ):
        # Default to outputs folder in repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        outputs_dir = os.path.join(repo_root, "Outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cleaned = os.path.join(outputs_dir, f"{base_name}_cleaned.jsonl")
        long_file = os.path.join(outputs_dir, f"{base_name}_long.jsonl")

        cleaned_entries = []
        long_entries = []
        long_count = 0
        error_count = 0
        longest_entry_pre = None
        longest_count_pre = 0

        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    count = self.get_token_count(entry)

                    if count > longest_count_pre:
                        longest_count_pre = count
                        longest_entry_pre = entry

                    if count <= max_tokens:
                        cleaned_entries.append((entry, count))
                    else:
                        long_entries.append((entry, count))
                        long_count += 1
                except Exception:
                    error_count += 1

        if sort_by_length:
            cleaned_entries = self._approximate_length_sort(
                cleaned_entries,
                bucket_tokens=bucket_tokens,
                longest_first=longest_first,
            )
            long_entries = self._approximate_length_sort(
                long_entries,
                bucket_tokens=bucket_tokens,
                longest_first=longest_first,
            )

        with open(cleaned, "w", encoding="utf-8") as fclean:
            for entry, _ in cleaned_entries:
                json.dump(entry, fclean, ensure_ascii=False)
                fclean.write("\n")

        with open(long_file, "w", encoding="utf-8") as flong:
            for entry, _ in long_entries:
                json.dump(entry, flong, ensure_ascii=False)
                flong.write("\n")

        if cleaned_entries:
            longest_entry = max(cleaned_entries, key=lambda x: x[1])
            post_clean_log = (
                f"🥵️ Longest Entry After Cleaning ({longest_entry[1]} tokens):\n"
                f"{json.dumps(longest_entry[0], indent=2, ensure_ascii=False)}"
            )
        else:
            post_clean_log = "⚠️ No entries were retained after cleaning."

        pre_clean_log = (
            f"🧌 Longest Entry Before Cleaning ({longest_count_pre} tokens):\n"
            f"{json.dumps(longest_entry_pre, indent=2, ensure_ascii=False)}"
        ) if longest_entry_pre else "⚠️ Failed to determine longest entry before cleaning."

        mode = "HF chat template" if self.has_chat_template() else "raw-text fallback"
        if sort_by_length:
            sort_description = (
                f"Approximate {'longest' if longest_first else 'shortest'} first "
                f"({bucket_tokens}-token buckets, shuffled within each bucket)"
            )
        else:
            sort_description = "Original order"

        return (
            f"✅ Finished processing: {os.path.basename(file_path)}\n"
            f"🧮 Counting mode: {mode}\n"
            f"🪼 Cleaned: {len(cleaned_entries)}\n"
            f"🐉 Long: {long_count}\n"
            f"⚠️ Skipped/invalid: {error_count}\n"
            f"📁 Cleaned File: {cleaned}\n"
            f"📁 Long File: {long_file}\n"
            f"📏 Sort order: {sort_description}\n"
            f"{pre_clean_log}\n\n{post_clean_log}"
        )

    def tokenize_only(self, file_path: str):
        # Default to outputs folder in repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        outputs_dir = os.path.join(repo_root, "Outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(outputs_dir, f"{base_name}_tokenized.jsonl")
        error_count = 0
        written = 0

        with open(file_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    input_ids = self.encode_entry(entry)
                    out = {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                    }
                    json.dump(out, fout, ensure_ascii=False)
                    fout.write("\n")
                    written += 1
                except Exception:
                    error_count += 1

        mode = "HF chat template" if self.has_chat_template() else "raw-text fallback"
        return (
            f"✅ Tokenized file written to: {output_path}\n"
            f"🧮 Tokenization mode: {mode}\n"
            f"🧵 Entries written: {written}\n"
            f"⚠️ Skipped/invalid: {error_count}"
        )
