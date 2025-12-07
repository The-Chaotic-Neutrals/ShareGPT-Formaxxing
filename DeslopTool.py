# DeslopTool.py
import json
from pathlib import Path
from datasets import load_dataset, Dataset
import math
from typing import Callable, List, Tuple, Any


def load_filter_criteria(filter_files: List[str]) -> List[str]:
    filter_criteria = []
    for filter_file in filter_files:
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    # Deduplicate to speed up simple substring scans
    return list(dict.fromkeys(filter_criteria))


def filter_conversations(
    conversations: List[dict],
    filter_criteria: List[str],
    threshold_multiplier: float | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> Tuple[List[dict], int, int, int]:
    """
    threshold_multiplier:
        - None: keep everything (no removals) but still compute metrics
        - float: remove conversations whose matched count >= (average * threshold_multiplier)
    """
    filtered_conversations: List[dict] = []
    total_matched_phrases = 0
    removed_conversation_count = 0
    matched_counts: List[int] = []

    total = len(conversations)

    # Phase 1: Calculate matched counts per conversation
    for idx, conversation in enumerate(conversations):
        total_phrases_in_conversation = 0
        conv_list = conversation.get("conversations", [])
        if isinstance(conv_list, list):
            for msg in conv_list:
                if isinstance(msg, dict) and msg.get("from") == "gpt":
                    val = msg.get("value") or ""
                    if val:
                        # simple substring match for each phrase
                        matched_count = sum(1 for phrase in filter_criteria if phrase in val)
                        total_phrases_in_conversation += matched_count

        matched_counts.append(total_phrases_in_conversation)
        total_matched_phrases += total_phrases_in_conversation

        if progress_callback:
            # Report progress scaled to first half (0-50%)
            progress_callback(idx + 1, total, 1)

    if total == 0:
        # Nothing to do
        return [], 0, 0, 0

    if total_matched_phrases == 0:
        # No filtering needed/possible; keep all
        return conversations, 0, total_matched_phrases, 0

    average_matched_phrases = total_matched_phrases / total

    # Phase 2: Filter conversations
    for idx, conversation in enumerate(conversations):
        remove = False
        if threshold_multiplier is not None:
            # IMPORTANT: use multiplier directly against the *average*
            remove = matched_counts[idx] >= (average_matched_phrases * threshold_multiplier)

        if remove:
            removed_conversation_count += 1
        else:
            filtered_conversations.append(conversation)

        if progress_callback:
            # Report progress scaled to second half (50-100%)
            progress_callback(idx + 1, total, 2)

    filtered_count = removed_conversation_count

    return filtered_conversations, filtered_count, total_matched_phrases, removed_conversation_count


def write_filtered_jsonl(filtered_data: List[dict], output_file_path: Path) -> None:
    """
    Use HuggingFace Datasets to reliably write JSONL.
    """
    # Ensure parent exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # If empty, still emit a valid (empty) JSONL file
    if not filtered_data:
        with open(output_file_path, "w", encoding="utf-8") as f:
            pass
        return

    dataset = Dataset.from_list(filtered_data)
    dataset.to_json(str(output_file_path), force_ascii=False)


def filter_dataset(
    dataset_file: str,
    filter_files: List[str],
    threshold_multiplier: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    """
    Orchestrates loading, filtering, and writing a *_deslopped.jsonl.
    progress_callback expects (current, total) where current is a percent (0..100), total=100.
    """
    filter_criteria = load_filter_criteria(filter_files)

    # Load dataset (supports .jsonl or .json where .json is a list of objects)
    dataset = load_dataset("json", data_files=dataset_file, split="train", streaming=False)
    data = list(dataset)  # materialize

    last_reported_percent = [-1]  # mutable capture

    def wrapped_progress_callback(current: int, total: int, phase: int) -> None:
        if progress_callback:
            # Phase 1 maps to 0..50, Phase 2 maps to 50..100
            if total <= 0:
                percent = 0
            else:
                if phase == 1:
                    percent = math.floor((current / total) * 50)
                else:
                    percent = 50 + math.floor((current / total) * 50)
            if percent != last_reported_percent[0]:
                last_reported_percent[0] = percent
                progress_callback(percent, 100)

    filtered_data, filtered_count, total_matched_phrases, removed_conversation_count = filter_conversations(
        data,
        filter_criteria,
        threshold_multiplier,
        progress_callback=wrapped_progress_callback
    )

    output_folder = Path(dataset_file).parent / "deslopped"
    output_folder.mkdir(exist_ok=True)

    dataset_name = Path(dataset_file).stem
    output_file_path = output_folder / f"{dataset_name}_deslopped.jsonl"

    write_filtered_jsonl(filtered_data, output_file_path)

    output_message = (
        f"Total original conversations: {len(data)}\n"
        f"Total filtered conversations: {filtered_count}\n"
        f"Remaining conversations after filtering: {len(filtered_data)}\n"
        f"Total matched phrases: {total_matched_phrases}\n"
        f"Total conversations removed: {removed_conversation_count}\n"
        f"Filtered output written to: {output_file_path}\n"
    )

    return output_message
