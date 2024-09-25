import json
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_filter_criteria(filter_files):
    filter_criteria = []
    for filter_file in filter_files:
        with open(filter_file, 'r') as f:
            filter_criteria.extend(line.strip() for line in f if line.strip())
    return filter_criteria

def filter_conversations(conversations, filter_criteria):
    filtered_conversations = []
    filtered_count = 0  # Count of filtered conversations
    total_matched_phrases = 0  # Total number of matched phrases
    removed_conversation_count = 0  # Count of removed conversation keys

    for conversation in conversations:
        should_filter = False
        for msg in conversation.get("conversations", []):
            if msg["from"] == "gpt":
                # Check for matched phrases
                matched_phrases = [phrase for phrase in filter_criteria if phrase in msg["value"]]
                total_matched_phrases += len(matched_phrases)  # Count matched phrases

                if matched_phrases:
                    should_filter = True
                    break
            
        if should_filter:
            filtered_count += 1  # Increment the filtered count
            removed_conversation_count += 1  # Increment removed conversation count
        else:
            filtered_conversations.append(conversation)

    return filtered_conversations, filtered_count, total_matched_phrases, removed_conversation_count

def write_filtered_jsonl(filtered_data, output_file_path):
    with open(output_file_path, 'w') as file:
        for conversation in filtered_data:
            json.dump(conversation, file)
            file.write('\n')  # Write a newline after each JSON object

def filter_dataset(dataset_file, filter_files):
    # Load filter criteria
    filter_criteria = load_filter_criteria(filter_files)

    # Load dataset
    data = load_jsonl(dataset_file)

    # Filter conversations
    filtered_data, filtered_count, total_matched_phrases, removed_conversation_count = filter_conversations(data, filter_criteria)

    # Create output folder if it doesn't exist
    output_folder = Path(dataset_file).parent / "deslopped"
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
