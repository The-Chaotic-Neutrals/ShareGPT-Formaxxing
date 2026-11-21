"""Read and write DeslopMancer rows without a schema conversion."""
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


_LEADING_THOUGHT = re.compile(r"^\s*(<\|channel>\s*thought\b|<think\b[^>]*>)", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetRow:
    record: dict
    position: int
    unit: str
    raw: bytes | None = None

    def locate(self):
        return {self.unit: self.position}

    def assistant_texts(self):
        try:
            return assistant_texts(self.record)
        except ValueError as exc:
            raise ValueError(f"{self.unit.capitalize()} {self.position}: {exc}") from exc


def assistant_texts(record):
    turns = record.get("conversations", record.get("messages"))
    if not isinstance(turns, list):
        raise ValueError("Expected a conversations or messages list.")
    replies = []
    for number, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {number} is not a message object.")
        role = turn.get("from") or turn.get("role")
        if str(role or "").strip().lower() not in {"gpt", "assistant", "model"}:
            continue
        value = turn.get("value", turn.get("content"))
        if isinstance(value, list):
            value = "\n".join(
                part["text"] for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Turn {number} has non-text assistant content.")
        # Only remove a leading, real reasoning block. Literal examples in
        # the answer and the saved record itself remain untouched.
        while match := _LEADING_THOUGHT.match(value):
            closer = "<channel|>" if match[1].lower().startswith("<|channel>") else "</think>"
            tail = value[match.end():]
            end = tail.lower().find(closer)
            if end < 0:
                raise ValueError(f"Turn {number} has an unclosed reasoning block.")
            value = tail[end + len(closer):]
        if value.strip():
            replies.append(value)
    return replies


def read_rows(input_path):
    path = Path(input_path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as source:
            records = json.load(source)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array of conversation objects.")
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError(f"{path}, entry {index}: expected a JSON object.")
            yield DatasetRow(record, index, "entry")
        return
    with path.open("rb") as source:
        for number, raw in enumerate(source, 1):
            if number == 1:
                raw = raw.removeprefix(b"\xef\xbb\xbf")
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}, line {number}: {exc}") from exc
            yield DatasetRow(record, number, "line", raw)


def write_rows(output_path, rows, input_path=None):
    path = Path(output_path)
    if input_path is not None and path.resolve() == Path(input_path).resolve():
        raise ValueError("Choose an output path separate from the input dataset.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            temp = Path(out.name)
            for row in rows:
                if isinstance(row, DatasetRow):
                    raw = row.raw or json.dumps(row.record, ensure_ascii=False).encode("utf-8")
                else:
                    raw = json.dumps(row, ensure_ascii=False).encode("utf-8")
                out.write(raw)
                if not raw.endswith(b"\n"):
                    out.write(b"\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def output_paths(input_paths, directory):
    reserved = {Path(path).resolve() for path in input_paths}
    paths = []
    for input_path in input_paths:
        stem = Path(input_path).stem
        number = 1
        while True:
            suffix = "" if number == 1 else f"_{number}"
            output = Path(directory) / f"{stem}{suffix}_deslopped.jsonl"
            audit = removed_report(output)
            if output.resolve() not in reserved and audit.resolve() not in reserved:
                reserved.update((output.resolve(), audit.resolve()))
                paths.append(output)
                break
            number += 1
    return paths


def removed_report(output_path):
    path = Path(output_path)
    return path.with_name(f"{path.stem}_removed.jsonl")
