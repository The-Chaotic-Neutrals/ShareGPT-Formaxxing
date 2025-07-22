import json
from pathlib import Path
from datasets import load_dataset, Dataset
import math

def load_filter_criteria(filter_files):
    filter_criteria = []
    for filter_file in filter_files:
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    return filter_criteria

def filter_conversations(conversations, filter_criteria, threshold=None, progress_callback=None):
    filtered_conversations = []
    total_matched_phrases = 0
    removed_conversation_count = 0

    matched_counts = []

    total = len(conversations)
    # Phase 1: Calculate matched counts per conversation
    for idx, conversation in enumerate(conversations):
        total_phrases_in_conversation = 0
        for msg in conversation.get("conversations", []):
            if msg.get("from") == "gpt" and msg.get("value"):
                matched_phrases = [phrase for phrase in filter_criteria if phrase in msg["value"]]
                matched_count = len(matched_phrases)
                total_phrases_in_conversation += matched_count

        matched_counts.append(total_phrases_in_conversation)
        total_matched_phrases += total_phrases_in_conversation

        if progress_callback:
            # Report progress scaled to first half (0-50%)
            progress_callback(idx + 1, total, phase=1)

    if total_matched_phrases == 0:
        # No filtering needed, just return original dataset
        return conversations, 0, total_matched_phrases, 0

    average_matched_phrases = total_matched_phrases / len(conversations) if conversations else 0

    # Phase 2: Filter conversations
    for idx, conversation in enumerate(conversations):
        if threshold is not None and matched_counts[idx] >= average_matched_phrases * threshold:
            removed_conversation_count += 1
        else:
            filtered_conversations.append(conversation)

        if progress_callback:
            # Report progress scaled to second half (50-100%)
            progress_callback(idx + 1, total, phase=2)

    filtered_count = removed_conversation_count

    return filtered_conversations, filtered_count, total_matched_phrases, removed_conversation_count

def write_filtered_jsonl(filtered_data, output_file_path):
    # Using datasets to save
    dataset = Dataset.from_list(filtered_data)
    dataset.to_json(str(output_file_path), force_ascii=False)

def filter_dataset(dataset_file, filter_files, threshold=None, progress_callback=None):
    filter_criteria = load_filter_criteria(filter_files)
    # Load dataset using datasets library
    dataset = load_dataset("json", data_files=dataset_file, split="train", streaming=False)
    data = list(dataset)  # Convert to list for processing

    last_reported_percent = [-1]  # Mutable to persist across calls

    def wrapped_progress_callback(current, total, phase):
        if progress_callback:
            # Progress reporting with phase info and scaled progress
            if phase == 1:
                percent = math.floor((current / total) * 50)
            else:
                percent = 50 + math.floor((current / total) * 50)

            # Only update if percent actually changed
            if percent != last_reported_percent[0]:
                last_reported_percent[0] = percent
                progress_callback(percent, 100)

    filtered_data, filtered_count, total_matched_phrases, removed_conversation_count = filter_conversations(
        data,
        filter_criteria,
        threshold,
        progress_callback=wrapped_progress_callback
    )

    output_folder = Path(__file__).parent / "deslopped"
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
