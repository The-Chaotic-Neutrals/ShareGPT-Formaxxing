import math
import re
from pathlib import Path

from App.DeslopMancer.dataset_rows import (
    DatasetRow, output_paths, read_rows, removed_report, write_rows,
)


def load_filter_criteria(filter_files):
    criteria = []
    for path in filter_files:
        with open(path, encoding="utf-8-sig") as source:
            criteria.extend(line.strip() for line in source if line.strip())
    return list(dict.fromkeys(criteria))


def phrase_patterns(phrases):
    patterns = {}
    for phrase in dict.fromkeys(phrases):
        if not phrase:
            continue
        left = r"(?<!\w)" if re.match(r"\w", phrase[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", phrase[-1]) else ""
        patterns[phrase] = re.compile(left + re.escape(phrase) + right)
    return patterns


def phrase_matches(row, patterns):
    return [
        phrase for text in row.assistant_texts()
        for phrase, pattern in patterns.items() if phrase in text and pattern.search(text)
    ]


def _partition_rows(rows, patterns, threshold_multiplier, progress_callback):
    if threshold_multiplier is not None and (
        not math.isfinite(threshold_multiplier) or threshold_multiplier <= 0
    ):
        raise ValueError("Threshold multiplier must be finite and greater than zero.")
    matches = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        matches.append(phrase_matches(row, patterns))
        if progress_callback:
            progress_callback(index, total, 1)

    hits = sum(map(len, matches))
    threshold = hits / total * threshold_multiplier if total and threshold_multiplier is not None else None
    kept, removed = [], []
    for index, (row, phrases) in enumerate(zip(rows, matches), 1):
        if threshold is not None and phrases and len(phrases) >= threshold:
            removed.append({
                **row.locate(), "reason": "assistant_phrase_threshold",
                "matched_phrases": phrases, "match_count": len(phrases), "threshold": threshold,
            })
        else:
            kept.append(row)
        if progress_callback:
            progress_callback(index, total, 2)
    return kept, removed, hits


def filter_conversations(conversations, filter_criteria, threshold_multiplier=None, progress_callback=None):
    rows = [DatasetRow(record, index, "entry") for index, record in enumerate(conversations, 1)]
    kept, removed, hits = _partition_rows(rows, phrase_patterns(filter_criteria), threshold_multiplier, progress_callback)
    return [row.record for row in kept], len(removed), hits, len(removed)


def write_filtered_jsonl(filtered_data, output_file_path):
    write_rows(output_file_path, filtered_data)


def filter_dataset(dataset_file, filter_files, threshold_multiplier=None, progress_callback=None, *, output_path=None):
    patterns = phrase_patterns(load_filter_criteria(filter_files))
    data = list(read_rows(dataset_file))
    last_percent = -1

    def progress(current, total, phase):
        nonlocal last_percent
        percent = (phase - 1) * 50 + current * 50 // total
        if progress_callback and percent != last_percent:
            last_percent = percent
            progress_callback(percent, 100)

    kept, removed, hits = _partition_rows(data, patterns, threshold_multiplier, progress)
    if output_path is None:
        output_path = output_paths([dataset_file], Path(__file__).resolve().parents[2] / "Outputs")[0]
    report = removed_report(output_path)
    write_rows(output_path, kept, dataset_file)
    write_rows(report, removed, dataset_file)
    if progress_callback and last_percent != 100:
        progress_callback(100, 100)
    return (
        f"Kept {len(kept)} of {len(data)} conversations. Removed {len(removed)}.\n"
        f"Total matched phrases: {hits}\n"
        f"Filtered output: {output_path}\nRemoved-row report: {report}\n"
    )
