"""
SpaceMancer - Conservative spacing restoration for ShareGPT-style datasets.

This module applies high-confidence spacing fixes to conversation text while
preserving dataset structure. It supports local JSON/JSONL files and
Hugging Face datasets.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None


CONTRACTION_MAP = {
    "cant": "can't",
    "wont": "won't",
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
    "id": "I'd",
    "youre": "you're",
    "theyre": "they're",
    "thats": "that's",
    "theres": "there's",
    "whats": "what's",
}

ROLE_ALIASES = {
    "human": "human",
    "user": "human",
    "gpt": "gpt",
    "assistant": "gpt",
    "system": "system",
}


@dataclass
class SpaceMancerStats:
    total_rows: int = 0
    changed_rows: int = 0
    changed_turns: int = 0
    skipped_rows: int = 0
    output_path: str = ""
    audit_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "changed_rows": self.changed_rows,
            "changed_turns": self.changed_turns,
            "skipped_rows": self.skipped_rows,
            "output_path": self.output_path,
            "audit_path": self.audit_path,
        }


class SpaceMancer:
    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        min_token_len: int = 8,
        min_score_delta: float = 4.5,
        allowed_roles: Optional[set[str]] = None,
        enable_glued_token_split: bool = True,
        llm_refiner: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.log_callback = log_callback or print
        self.min_token_len = min_token_len
        self.min_score_delta = min_score_delta
        self.allowed_roles = allowed_roles
        self.enable_glued_token_split = enable_glued_token_split
        self.llm_refiner = llm_refiner
        self._spell = self._init_spellchecker()

        self._protected_pattern = re.compile(
            r"```.*?```|`[^`]*`|https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[A-Za-z]:\\[^\s]+|(?<!\w)/(?:[\w.\-]+/)*[\w.\-]+",
            re.DOTALL,
        )
        self._long_token_pattern = re.compile(r"\b[a-z]{8,}\b")
        self._short_whitelist = {
            "and", "for", "the", "that", "with", "from", "this", "your", "you",
            "are", "was", "were", "can", "not", "but", "all", "any", "out",
        }

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def _init_spellchecker(self):
        if SpellChecker is None:
            self.log("Warning: pyspellchecker unavailable. Glued-token splitting disabled.")
            return None
        try:
            return SpellChecker(language="en", case_sensitive=False)
        except Exception as exc:
            self.log(f"Warning: SpellChecker init failed: {exc}")
            return None

    def _protect_spans(self, text: str) -> tuple[str, dict[str, str]]:
        replacements: dict[str, str] = {}

        def _repl(match: re.Match) -> str:
            key = f"__SM_PROTECTED_{len(replacements)}__"
            replacements[key] = match.group(0)
            return key

        return self._protected_pattern.sub(_repl, text), replacements

    def _restore_spans(self, text: str, replacements: dict[str, str]) -> str:
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text

    def _word_frequency(self, word: str) -> int:
        if not self._spell:
            return 0
        try:
            return int(self._spell.word_frequency.dictionary.get(word.lower(), 0))
        except Exception:
            return 0

    def _is_known_word(self, word: str) -> bool:
        if not self._spell:
            return False
        w = word.lower()
        if w in ("a", "i"):
            return True
        freq = self._word_frequency(w)
        return freq > 0

    def _apply_contractions(self, text: str) -> tuple[str, bool]:
        changed = False

        def replace_match(match: re.Match, replacement: str) -> str:
            nonlocal changed
            original = match.group(0)
            changed = True
            if original.isupper():
                return replacement.upper()
            if original[:1].isupper():
                return replacement[:1].upper() + replacement[1:]
            return replacement

        for raw, replacement in CONTRACTION_MAP.items():
            pattern = re.compile(rf"\b{re.escape(raw)}\b", re.IGNORECASE)
            text = pattern.sub(lambda m, rep=replacement: replace_match(m, rep), text)

        return text, changed

    def _segment_token(self, token: str) -> Optional[str]:
        if not self._spell:
            return None
        if len(token) < self.min_token_len:
            return None
        if self._is_known_word(token):
            return None

        lower = token.lower()
        n = len(lower)
        max_word_len = 20
        neg_inf = -1e12
        dp = [neg_inf] * (n + 1)
        parent = [-1] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] <= neg_inf / 2:
                continue
            end_max = min(n, i + max_word_len)
            for j in range(i + 1, end_max + 1):
                piece = lower[i:j]
                if len(piece) == 1 and piece not in {"a", "i"}:
                    continue
                if not self._is_known_word(piece):
                    continue

                freq = self._word_frequency(piece)
                score = dp[i] + math.log(freq + 1.0) - 0.35
                if score > dp[j]:
                    dp[j] = score
                    parent[j] = i

        if parent[n] == -1:
            return None

        parts = []
        idx = n
        while idx > 0 and parent[idx] != -1:
            start = parent[idx]
            parts.append(lower[start:idx])
            idx = start
        parts.reverse()

        if len(parts) < 2:
            return None
        if len(parts) > 3:
            return None
        for p in parts:
            if len(p) <= 2 and p not in self._short_whitelist:
                return None
            if p not in self._short_whitelist and self._word_frequency(p) < 30:
                return None

        # Keep conservative behavior: avoid over-fragmenting unknown technical words.
        longest_part = max(len(p) for p in parts)
        if longest_part < 4:
            return None

        unsplit_score = math.log(self._word_frequency(lower) + 1.0)
        split_score = dp[n]
        if split_score - unsplit_score < self.min_score_delta:
            return None

        return " ".join(parts)

    def _split_glued_tokens(self, text: str) -> tuple[str, bool]:
        if not self.enable_glued_token_split:
            return text, False

        changed = False

        def repl(match: re.Match) -> str:
            nonlocal changed
            token = match.group(0)
            split = self._segment_token(token)
            if split and split != token:
                changed = True
                return split
            return token

        return self._long_token_pattern.sub(repl, text), changed

    def _apply_spacing_rules(self, text: str) -> tuple[str, bool]:
        before = text
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
        text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
        text = re.sub(r"(?<=[!?;:])(?=[A-Za-z])", " ", text)
        text = re.sub(r"(?<=[A-Za-z],)(?=[A-Za-z])", " ", text)
        text = re.sub(r"(?<=[A-Za-z]\.)(?=[A-Za-z])", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text, text != before

    def _has_spacing_suspicion(self, text: str) -> bool:
        return bool(
            re.search(r"(?<=[a-z])(?=[A-Z])", text)
            or re.search(r"\b[a-z]{10,}\b", text)
            or re.search(r"(?<=[!?;:,.])(?=[A-Za-z])", text)
            or re.search(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", text)
        )

    @staticmethod
    def _same_nonspace_text(a: str, b: str) -> bool:
        a_ns = re.sub(r"\s+", "", a)
        b_ns = re.sub(r"\s+", "", b)
        return a_ns == b_ns

    def _apply_llm_refinement(self, text: str) -> tuple[str, bool]:
        if not self.llm_refiner:
            return text, False
        if not self._has_spacing_suspicion(text):
            return text, False

        protected, replacements = self._protect_spans(text)
        try:
            candidate = self.llm_refiner(protected)
        except Exception:
            return text, False

        if not candidate or not isinstance(candidate, str):
            return text, False
        if not self._same_nonspace_text(protected, candidate):
            return text, False

        restored = self._restore_spans(candidate, replacements)
        if "__SM_PROTECTED_" in restored:
            return text, False
        return restored, restored != text

    def clean_text(self, text: str) -> tuple[str, list[str]]:
        if not isinstance(text, str) or not text.strip():
            return text, []

        rules_applied: list[str] = []
        protected, replacements = self._protect_spans(text)

        protected, contractions_changed = self._apply_contractions(protected)
        if contractions_changed:
            rules_applied.append("contractions")

        protected, spacing_changed = self._apply_spacing_rules(protected)
        if spacing_changed:
            rules_applied.append("spacing")

        protected, split_changed = self._split_glued_tokens(protected)
        if split_changed:
            rules_applied.append("split_glued_tokens")

        restored = self._restore_spans(protected, replacements)
        restored, llm_changed = self._apply_llm_refinement(restored)
        if llm_changed:
            rules_applied.append("llm_refine")
        return restored, rules_applied

    def _normalize_role(self, role: str) -> str:
        return ROLE_ALIASES.get((role or "").strip().lower(), (role or "").strip().lower())

    def process_record(self, record: dict[str, Any], row_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(record, dict):
            return record, []

        turns: Optional[list[Any]] = None
        role_key = "from"
        value_key = "value"

        if isinstance(record.get("conversations"), list):
            turns = record["conversations"]
            role_key = "from"
            value_key = "value"
        elif isinstance(record.get("messages"), list):
            turns = record["messages"]
            role_key = "role"
            value_key = "content"

        if turns is None:
            return record, []

        audit_entries: list[dict[str, Any]] = []

        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            if value_key not in turn or not isinstance(turn.get(value_key), str):
                continue

            role = self._normalize_role(turn.get(role_key, ""))
            if self.allowed_roles is not None and role not in self.allowed_roles:
                continue

            before = turn[value_key]
            after, rules = self.clean_text(before)
            if after != before:
                turn[value_key] = after
                audit_entries.append(
                    {
                        "row_index": row_index,
                        "turn_index": turn_index,
                        "role": turn.get(role_key, ""),
                        "container": "messages" if value_key == "content" else "conversations",
                        "rules": rules,
                        "before": before,
                        "after": after,
                    }
                )

        return record, audit_entries

    def _default_output_paths(
        self,
        base_name: str,
        output_filename: Optional[str] = None,
        write_audit: bool = True,
    ) -> tuple[Path, Optional[Path]]:
        repo_root = Path(__file__).parent.parent.parent
        output_dir = repo_root / "Outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename:
            output_name = output_filename if output_filename.endswith(".jsonl") else f"{output_filename}.jsonl"
        else:
            output_name = f"{base_name}_spaced.jsonl"
        output_path = output_dir / output_name

        audit_path = None
        if write_audit:
            audit_path = output_dir / f"{Path(output_name).stem}_audit.jsonl"

        return output_path, audit_path

    def process_local_file(
        self,
        input_path: str,
        output_filename: Optional[str] = None,
        write_audit: bool = True,
        dry_run: bool = False,
        preview_limit: int = 20,
    ) -> dict[str, Any]:
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_path: Optional[Path] = None
        audit_path: Optional[Path] = None
        if not dry_run:
            output_path, audit_path = self._default_output_paths(
                base_name=input_file.stem,
                output_filename=output_filename,
                write_audit=write_audit,
            )

        stats = SpaceMancerStats()
        preview_entries: list[dict[str, Any]] = []
        audit_fh = open(audit_path, "w", encoding="utf-8") if audit_path else None

        try:
            out_f = open(output_path, "w", encoding="utf-8") if output_path else None
            try:
                ext = input_file.suffix.lower()
                if ext == ".jsonl":
                    with open(input_file, "r", encoding="utf-8") as in_f:
                        for idx, line in enumerate(in_f):
                            line = line.strip()
                            if not line:
                                continue
                            stats.total_rows += 1
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                stats.skipped_rows += 1
                                continue

                            record, audit_entries = self.process_record(record, idx)
                            if audit_entries:
                                stats.changed_rows += 1
                                stats.changed_turns += len(audit_entries)
                                if len(preview_entries) < preview_limit:
                                    remaining = preview_limit - len(preview_entries)
                                    preview_entries.extend(audit_entries[:remaining])
                                if audit_fh:
                                    for entry in audit_entries:
                                        audit_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

                            if out_f:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif ext == ".json":
                    with open(input_file, "r", encoding="utf-8") as in_f:
                        payload = json.load(in_f)

                    records = payload if isinstance(payload, list) else [payload]
                    for idx, record in enumerate(records):
                        stats.total_rows += 1
                        if not isinstance(record, dict):
                            stats.skipped_rows += 1
                            continue

                        record, audit_entries = self.process_record(record, idx)
                        if audit_entries:
                            stats.changed_rows += 1
                            stats.changed_turns += len(audit_entries)
                            if len(preview_entries) < preview_limit:
                                remaining = preview_limit - len(preview_entries)
                                preview_entries.extend(audit_entries[:remaining])
                            if audit_fh:
                                for entry in audit_entries:
                                    audit_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

                        if out_f:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    raise ValueError("Unsupported file type. Use .jsonl or .json")
            finally:
                if out_f:
                    out_f.close()
        finally:
            if audit_fh:
                audit_fh.close()

        stats.output_path = str(output_path) if output_path else ""
        stats.audit_path = str(audit_path) if audit_path else ""
        result = stats.to_dict()
        result["dry_run"] = dry_run
        result["preview"] = preview_entries
        return result

    def process_hf_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        config_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        output_filename: Optional[str] = None,
        write_audit: bool = True,
        dry_run: bool = False,
        preview_limit: int = 20,
    ) -> dict[str, Any]:
        if load_dataset is None:
            raise RuntimeError("datasets package is unavailable. Install dependencies first.")

        slug = dataset_id.replace("/", "_")
        output_path: Optional[Path] = None
        audit_path: Optional[Path] = None
        if not dry_run:
            output_path, audit_path = self._default_output_paths(
                base_name=f"{slug}_{split}",
                output_filename=output_filename,
                write_audit=write_audit,
            )

        kwargs: dict[str, Any] = {}
        if hf_token:
            kwargs["token"] = hf_token

        dataset = load_dataset(dataset_id, name=(config_name or None), split=split, **kwargs)

        stats = SpaceMancerStats()
        preview_entries: list[dict[str, Any]] = []
        audit_fh = open(audit_path, "w", encoding="utf-8") if audit_path else None

        try:
            out_f = open(output_path, "w", encoding="utf-8") if output_path else None
            try:
                for idx, row in enumerate(dataset):
                    stats.total_rows += 1
                    if not isinstance(row, dict):
                        stats.skipped_rows += 1
                        continue

                    record = dict(row)
                    record, audit_entries = self.process_record(record, idx)

                    if audit_entries:
                        stats.changed_rows += 1
                        stats.changed_turns += len(audit_entries)
                        if len(preview_entries) < preview_limit:
                            remaining = preview_limit - len(preview_entries)
                            preview_entries.extend(audit_entries[:remaining])
                        if audit_fh:
                            for entry in audit_entries:
                                audit_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

                    if out_f:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                if out_f:
                    out_f.close()
        finally:
            if audit_fh:
                audit_fh.close()

        stats.output_path = str(output_path) if output_path else ""
        stats.audit_path = str(audit_path) if audit_path else ""
        result = stats.to_dict()
        result["dry_run"] = dry_run
        result["preview"] = preview_entries
        return result
