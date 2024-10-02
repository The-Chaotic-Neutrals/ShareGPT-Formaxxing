import json
from pathlib import Path

def load_jsonl(file_path):
    data = []
    # Open file with UTF-8 encoding and error handling
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}. Error: {e}")
    return data

def load_filter_criteria(filter_files):
    filter_criteria = []
    for filter_file in filter_files:
        # Open filter files with UTF-8 encoding
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    return filter_criteria

def filter_conversations(conversations, filter_criteria, threshold=None):
    filtered_conversations = []
    total_matched_phrases = 0  # Total number of matched phrases
    removed_conversation_count = 0  # Count of removed conversation keys

    # Calculate total matched phrases in each conversation
    matched_counts = []
    
    for conversation in conversations:
        total_phrases_in_conversation = 0  # Total matched phrases in this conversation
        for msg in conversation.get("conversations", []):
            if msg["from"] == "gpt":
                # Count matched phrases
                matched_phrases = [phrase for phrase in filter_criteria if phrase in msg["value"]]
                matched_count = len(matched_phrases)  # Count matched phrases in the current message
                total_phrases_in_conversation += matched_count  # Accumulate total matched phrases

        matched_counts.append(total_phrases_in_conversation)  # Store matched count for the current conversation
        total_matched_phrases += total_phrases_in_conversation  # Count matched phrases overall

    # Check if total matched phrases is zero; if so, return original conversations
    if total_matched_phrases == 0:
        return conversations, 0, total_matched_phrases, 0

    # Calculate average matched phrases per conversation
    average_matched_phrases = total_matched_phrases / len(conversations) if conversations else 0

    # Apply threshold filtering based on the average if a threshold is provided
    for idx, conversation in enumerate(conversations):
        if threshold is not None and matched_counts[idx] >= average_matched_phrases * threshold:
            removed_conversation_count += 1  # Increment removed conversation count
        else:
            filtered_conversations.append(conversation)

    filtered_count = removed_conversation_count  # Total filtered conversations

    return filtered_conversations, filtered_count, total_matched_phrases, removed_conversation_count

def write_filtered_jsonl(filtered_data, output_file_path):
    # Write to file with UTF-8 encoding
    with open(output_file_path, 'w', encoding='utf-8', errors='replace') as file:
        for conversation in filtered_data:
            json.dump(conversation, file, ensure_ascii=False)
            file.write('\n')  # Write a newline after each JSON object

def filter_dataset(dataset_file, filter_files, threshold=None):
    # Load filter criteria
    filter_criteria = load_filter_criteria(filter_files)

    # Load dataset
    data = load_jsonl(dataset_file)

    # Filter conversations
    filtered_data, filtered_count, total_matched_phrases, removed_conversation_count = filter_conversations(data, filter_criteria, threshold)

    # Create output folder in the same directory as the main app
    output_folder = Path(__file__).parent / "deslopped"  # Use __file__ to get the current script's directory
    output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist

    # Prepare output file path
    dataset_name = Path(dataset_file).stem  # Get the stem (name without extension)
    output_file_path = output_folder / f"{dataset_name}_deslopped.jsonl"  # Create new output file name

    # Write filtered conversations to the output file
    write_filtered_jsonl(filtered_data, output_file_path)

    # Prepare output message
    output_message = (
        f"Total original conversations: {len(data)}\n"
        f"Total filtered conversations: {filtered_count}\n"
        f"Remaining conversations after filtering: {len(filtered_data)}\n"
        f"Total matched phrases: {total_matched_phrases}\n"
        f"Total conversations removed: {removed_conversation_count}\n"
        f"Filtered output written to: {output_file_path}"
    )

    return output_message
