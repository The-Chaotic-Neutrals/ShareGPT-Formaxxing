import json
import os
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
        # update recent models
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

    def get_token_count(self, entry):
        try:
            flat = "\n".join([m["value"] for m in entry.get("conversations", []) if "value" in m])
            return len(self.tokenizer.encode(flat, add_special_tokens=False))
        except Exception:
            return 0

    # ---------------- Core operations ----------------
    def analyze_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_entries = [json.loads(line) for line in f]
        lengths = [self.get_token_count(e) for e in raw_entries]
        if not lengths:
            return "⚠️ No entries to analyze."
        percentiles = np.percentile(lengths, [10, 25, 50, 75, 90, 95, 99, 100])
        longest_idx = lengths.index(max(lengths))
        return (
            f"🔢 Percentiles: {percentiles.tolist()}\n"
            f"📏 Min: {min(lengths)} | Max: {max(lengths)}\n"
            f"📍 Longest Entry Index: {longest_idx}\n"
            f"🧪 Tokens: {lengths[longest_idx]}\n"
            f"🕵️ Entry:\n{json.dumps(raw_entries[longest_idx], indent=2, ensure_ascii=False)}"
        )

    def clean_file(self, file_path: str, max_tokens: int):
        base = os.path.splitext(file_path)[0]
        cleaned = f"{base}_cleaned.jsonl"
        long_file = f"{base}_long.jsonl"
        cleaned_entries = []
        long_count = 0
        longest_entry_pre = None
        longest_count_pre = 0

        with open(file_path, "r", encoding="utf-8") as fin:
            raw_lines = list(fin)

        for line in raw_lines:
            try:
                entry = json.loads(line)
                count = self.get_token_count(entry)
                if count > longest_count_pre:
                    longest_count_pre = count
                    longest_entry_pre = entry
            except Exception:
                pass

        with open(file_path, "r", encoding="utf-8") as fin, \
             open(cleaned, "w", encoding="utf-8") as fclean, \
             open(long_file, "w", encoding="utf-8") as flong:
            for line in fin:
                try:
                    entry = json.loads(line)
                    count = self.get_token_count(entry)
                    if count <= max_tokens:
                        cleaned_entries.append((entry, count))
                        json.dump(entry, fclean, ensure_ascii=False)
                        fclean.write("\n")
                    else:
                        json.dump(entry, flong, ensure_ascii=False)
                        flong.write("\n")
                        long_count += 1
                except Exception:
                    pass

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

        return (
            f"✅ Finished processing: {os.path.basename(file_path)}\n"
            f"🪼 Cleaned: {len(cleaned_entries)}\n🐉 Long: {long_count}\n"
            f"📁 Cleaned File: {cleaned}\n📁 Long File: {long_file}\n"
            f"{pre_clean_log}\n\n{post_clean_log}"
        )

    def tokenize_only(self, file_path: str):
        output_path = os.path.splitext(file_path)[0] + "_tokenized.jsonl"
        with open(file_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    entry = json.loads(line)
                    convs = entry.get("conversations", [])
                    flat = "\n".join([m["value"] for m in convs if "value" in m])
                    encoded = self.tokenizer(
                        flat,
                        truncation=False,
                        padding=False,
                        return_attention_mask=True,
                        return_tensors=None,
                        add_special_tokens=False
                    )
                    out = {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"]
                    }
                    json.dump(out, fout, ensure_ascii=False)
                    fout.write("\n")
                except Exception:
                    pass
        return f"✅ Tokenized file written to: {output_path}"
