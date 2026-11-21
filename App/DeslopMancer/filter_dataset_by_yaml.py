from pathlib import Path

import yaml

from App.DeslopMancer.dataset_rows import read_rows, removed_report, write_rows
from App.DeslopMancer.DeslopTool import phrase_matches, phrase_patterns


def filter_dataset_by_yaml(input_path, yaml_path, output_dir):
    with open(yaml_path, encoding="utf-8-sig") as source:
        phrases = yaml.safe_load(source)
    if not isinstance(phrases, list) or not all(isinstance(phrase, str) and phrase.strip() for phrase in phrases):
        raise ValueError("Expected a YAML list of nonempty filter phrases.")
    patterns = phrase_patterns(phrases)
    kept, removed = [], []
    for row in read_rows(input_path):
        matches = phrase_matches(row, patterns)
        if matches:
            removed.append({**row.locate(), "reason": "assistant_yaml_phrase", "matched_phrases": matches})
        else:
            kept.append(row)
    output = Path(output_dir) / "Filtered" / f"{Path(input_path).stem}_filtered.jsonl"
    report = removed_report(output)
    write_rows(output, kept, input_path)
    write_rows(report, removed, input_path)
    return (
        f"Filtered data saved to {output}\nOriginal size: {len(kept) + len(removed)}\n"
        f"Filtered size: {len(kept)}\nRemoved-row report: {report}"
    )
