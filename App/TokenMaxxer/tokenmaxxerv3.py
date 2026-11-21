import ast
import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def _read_studio_gemma_template(source, _mtime_ns):
    # Read the literal template without importing Studio or initializing Unsloth.
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "gemma4_thinking_template"
            for target in statement.targets
        ):
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                return statement.value.value
            break
    raise ValueError(f"Studio's Gemma 4 training template could not be read from {source}.")


def _studio_gemma_template():
    repo = Path(os.environ.get("UNSLOTH_STUDIO_REPO") or Path(__file__).resolve().parents[3] / "unsloth")
    source = repo / "unsloth" / "chat_templates.py"
    try:
        return _read_studio_gemma_template(source, source.stat().st_mtime_ns)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Studio checkout not found at {repo}. Set UNSLOTH_STUDIO_REPO to its folder "
            "to count Gemma 4 rows using Studio's training template."
        ) from exc


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
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        previous = self.tokenizer
        self.tokenizer = tokenizer
        try:
            if self._uses_gemma_reasoning():
                _studio_gemma_template()
        except Exception:
            self.tokenizer = previous
            raise
        self.model_name = model_repo
        self.save_config(model_repo)

    def has_chat_template(self):
        return bool(self.tokenizer and getattr(self.tokenizer, "chat_template", None))

    def _uses_gemma_reasoning(self):
        template = getattr(self.tokenizer, "chat_template", None)
        return isinstance(template, str) and "<|channel>thought" in template and "<turn|>" in template

    def _counting_mode(self):
        if self._uses_gemma_reasoning():
            return "Studio Gemma 4 reasoning SFT"
        return "HF chat template" if self.has_chat_template() else "raw-text fallback"

    @staticmethod
    def _entry_to_messages(entry):
        if not isinstance(entry, dict):
            raise ValueError("Expected a JSON object for the dataset row.")
        conversations = entry.get("conversations")
        if conversations is None:
            conversations = entry.get("messages")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("Expected a non-empty conversations or messages list.")

        roles = {
            "human": "user", "user": "user", "instruction": "user",
            "gpt": "assistant", "assistant": "assistant", "model": "assistant",
            "bot": "assistant", "output": "assistant", "system": "system",
            "input": "system", "developer": "developer", "tool": "tool",
        }
        messages = []
        for number, message in enumerate(conversations, 1):
            if not isinstance(message, dict):
                raise ValueError(f"Turn {number}: expected a message object.")
            source_role = message.get("from") or message.get("role") or message.get("speaker")
            role = roles.get(str(source_role or "").strip().lower())
            if role is None:
                raise ValueError(f"Turn {number}: unsupported or missing role {source_role!r}.")

            # Keep inline Gemma thoughts in content, as Studio does. Converting
            # them to reasoning_content changes the rendered message boundaries.
            rendered = {
                key: value for key, value in message.items()
                if key not in {"from", "speaker", "value", "text"} and value is not None
            }
            content = next(
                (message[key] for key in ("value", "content", "text") if message.get(key) is not None),
                "",
            )
            rendered.update(role=role, content=content)
            if "reasoning_content" not in rendered and rendered.get("thinking"):
                rendered["reasoning_content"] = rendered["thinking"]
            messages.append(rendered)

        system = entry.get("system")
        if isinstance(system, str) and system.strip() and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def encode_entry(self, entry, *, studio_template=None):
        """Encode the rendered text without adding another BOS or EOS."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        messages = self._entry_to_messages(entry)
        if self.has_chat_template():
            options = {}
            if self._uses_gemma_reasoning():
                options.update(
                    chat_template=studio_template or _studio_gemma_template(),
                    enable_thinking=True,
                    preserve_thinking=True,
                )
            tools = entry.get("tools")
            if tools:
                if isinstance(tools, str):
                    tools = json.loads(tools)
                if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
                    raise ValueError("Expected tools to be a list of tool definitions.")
                options["tools"] = []
                for tool in tools:
                    if "function" not in tool and "name" in tool:
                        tool = {"type": "function", "function": tool}
                    elif not tool.get("type") and "function" in tool:
                        tool = {**tool, "type": "function"}
                    options["tools"].append(tool)
            if not any(
                (m["content"].strip() if isinstance(m["content"], str) else m["content"])
                or m.get("tool_calls") or m.get("reasoning_content")
                for m in messages
            ):
                raise ValueError("No message content to tokenize.")
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **options,
            )
        else:
            text = "\n".join(
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                for content in (message["content"] for message in messages)
            )
        if not text.strip():
            raise ValueError("Rendered conversation is empty.")
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if not len(encoded):
            raise ValueError("Tokenizer returned zero tokens.")
        return encoded

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
    @staticmethod
    def _output_base(file_path):
        output = Path(__file__).resolve().parents[2] / "Outputs"
        output.mkdir(parents=True, exist_ok=True)
        return output / Path(file_path).stem

    def _encoded_rows(self, file_path, issues, *, studio_template):
        with open(file_path, "rb") as source:
            for number, raw in enumerate(source, 1):
                try:
                    line = raw.decode("utf-8-sig" if number == 1 else "utf-8")
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    ids = self.encode_entry(entry, studio_template=studio_template)
                except Exception as exc:
                    issues.append({
                        "line": number, "status": "invalid",
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                yield number, entry, ids

    @staticmethod
    def _write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _issue_summary(self, output_base, operation, issues):
        report = output_base.with_name(f"{output_base.name}_{operation}_issues.jsonl")
        issues = sorted(issues, key=lambda issue: issue["line"])
        self._write_jsonl(report, issues)
        preview = "\n".join(
            f"Line {issue['line']}: {issue['reason']}" for issue in issues[:10]
        )
        return f"Row issues: {len(issues)}\nIssue report: {report}" + (f"\n{preview}" if preview else "")

    def _prepare_template(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        return _studio_gemma_template() if self._uses_gemma_reasoning() else None

    def analyze_file(self, file_path: str):
        template = self._prepare_template()
        issues = []
        lengths = []
        longest_entry = None
        longest_tokens = 0
        longest_line = None
        for number, entry, ids in self._encoded_rows(file_path, issues, studio_template=template):
            count = len(ids)
            lengths.append(count)
            if count > longest_tokens:
                longest_entry, longest_tokens, longest_line = entry, count, number

        report = self._issue_summary(self._output_base(file_path), "analyze", issues)
        if not lengths:
            return f"No valid entries to analyze.\n{report}"

        percentiles = np.percentile(lengths, [10, 25, 50, 75, 90, 95, 99, 100])
        return (
            f"Counting mode: {self._counting_mode()}\n"
            f"Percentiles: {percentiles.tolist()}\n"
            f"Min: {min(lengths)} | Max: {max(lengths)}\n"
            f"Longest entry: line {longest_line} ({longest_tokens} tokens)\n"
            f"{report}\n"
            f"{json.dumps(longest_entry, indent=2, ensure_ascii=False)}"
        )

    def clean_file(
        self, file_path: str, max_tokens: int, sort_by_length: bool = False,
        longest_first: bool = False, bucket_tokens: int = 128,
    ):
        if max_tokens < 1:
            raise ValueError("Max token length must be a positive integer.")
        template = self._prepare_template()
        output_base = self._output_base(file_path)
        cleaned_path = output_base.with_name(f"{output_base.name}_cleaned.jsonl")
        long_path = output_base.with_name(f"{output_base.name}_long.jsonl")
        issues = []
        cleaned = []
        long = []
        longest_input = None
        # Studio final-turn-only SFT excludes rows at the cap because the
        # trainer cannot tell them apart from truncated conversations.
        usable_limit = max_tokens - 1 if template is not None else max_tokens
        for number, entry, ids in self._encoded_rows(file_path, issues, studio_template=template):
            count = len(ids)
            if longest_input is None or count > longest_input[1]:
                longest_input = (entry, count)
            if count <= usable_limit:
                cleaned.append((entry, count))
            else:
                long.append((entry, count))
                reason = (
                    f"{count} tokens reaches the {max_tokens}-token Studio final-turn limit."
                    if count == max_tokens else f"{count} tokens exceeds the {max_tokens}-token limit."
                )
                issues.append({"line": number, "status": "limit", "tokens": count, "reason": reason})

        if sort_by_length:
            cleaned = self._approximate_length_sort(cleaned, bucket_tokens, longest_first)
            long = self._approximate_length_sort(long, bucket_tokens, longest_first)
        self._write_jsonl(cleaned_path, (entry for entry, _ in cleaned))
        self._write_jsonl(long_path, (entry for entry, _ in long))
        report = self._issue_summary(output_base, "clean", issues)
        longest_retained = max(cleaned, key=lambda row: row[1], default=None)
        previews = []
        for name, row in (("Longest input", longest_input), ("Longest retained entry", longest_retained)):
            if row is not None:
                entry, count = row
                previews.append(f"{name} ({count} tokens):\n{json.dumps(entry, indent=2, ensure_ascii=False)}")
        sort_order = (
            f"{'Longest' if longest_first else 'Shortest'} first in {bucket_tokens}-token buckets"
            if sort_by_length else "Original order"
        )
        preview = "\n\n".join(previews)
        return (
            f"Finished processing: {Path(file_path).name}\n"
            f"Counting mode: {self._counting_mode()}\n"
            f"Cleaned: {len(cleaned)} | Outside token limit: {len(long)}\n"
            f"Cleaned file: {cleaned_path}\nLong file: {long_path}\n"
            f"Sort order: {sort_order}\n{report}\n\n{preview}"
        )

    def tokenize_only(self, file_path: str):
        template = self._prepare_template()
        output_base = self._output_base(file_path)
        output = output_base.with_name(f"{output_base.name}_tokenized.jsonl")
        issues = []
        written = 0

        def rows():
            nonlocal written
            for _, _, ids in self._encoded_rows(file_path, issues, studio_template=template):
                written += 1
                yield {"input_ids": ids, "attention_mask": [1] * len(ids)}

        self._write_jsonl(output, rows())
        report = self._issue_summary(output_base, "tokenize", issues)
        return (
            f"Tokenized file: {output}\n"
            f"Tokenization mode: {self._counting_mode()}\n"
            f"Entries written: {written}\n{report}"
        )
