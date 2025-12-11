import json
import re
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def ends_with_letter_number_comma(text):
    """Check if a text ends with a letter, number, or comma."""
    if isinstance(text, str):
        return bool(re.search(r'[a-zA-Z0-9,]$', text.strip()))
    return False

def normalize_text(text):
    """
    Normalize text for comparison: strip, lowercase, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def has_human_gpt_duplicate(conversations):
    """
    Return True if any human turn is exactly duplicated by the following gpt turn
    (after normalization).
    """
    for i in range(len(conversations) - 1):
        cur_msg = conversations[i]
        next_msg = conversations[i + 1]

        if cur_msg.get("from") == "human" and next_msg.get("from") == "gpt":
            cur_val = normalize_text(cur_msg.get("value", ""))
            next_val = normalize_text(next_msg.get("value", ""))
            if cur_val and cur_val == next_val:
                return True
    return False

def filter_dataset(
    input_path,
    output_dir,
    check_blank_turns=True,
    check_invalid_endings=True,
    check_null_gpt=True,
    check_duplicate_system=True,
    allow_empty_system_role=True,
    check_duplicate_turns=True,  # NEW
):
    """
    Filters a dataset of conversations based on specified criteria.

    Parameters:
        input_path (str): Path to the input JSONL file.
        output_dir (str): Directory to save the filtered dataset.
        check_blank_turns (bool): Remove conversations with blank turns.
        check_invalid_endings (bool): Remove conversations with invalid endings.
        check_null_gpt (bool): Remove conversations with null GPT responses.
        check_duplicate_system (bool): Remove duplicate system messages.
        allow_empty_system_role (bool): Allow conversations with empty system role.
        check_duplicate_turns (bool): Remove conversations with duplicate human→gpt turns.

    Returns:
        str: Summary of filtering results.
    """
    try:
        # Prepare paths
        input_path = Path(input_path)
        # If output_dir not provided or relative, default to outputs folder
        if output_dir is None or (isinstance(output_dir, str) and not os.path.isabs(output_dir)):
            script_dir = Path(__file__).parent.absolute()
            repo_root = script_dir.parent.absolute()
            output_dir = repo_root / "outputs" / "datamaxxer"
        output_dir = Path(output_dir)
        filtered_dir = output_dir / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        output_file = filtered_dir / f"{input_path.stem}_filtered.jsonl"
        
        filtered_data_count = 0
        original_data_count = 0
        json_error_count = 0
        duplicate_turn_conv_count = 0

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             output_file.open('w', encoding='utf-8') as outfile:
            
            for line in infile:
                original_data_count += 1
                line = line.strip()

                if not line:
                    continue  # Skip empty lines

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    json_error_count += 1
                    logging.error(f"JSON decode error at line {original_data_count}: {e}")
                    continue
                
                conversations = item.get("conversations", [])
                has_blank_turn = False
                has_invalid_ending = False
                has_null_gpt_value = False
                
                filtered_conversations = []
                for i, msg in enumerate(conversations):
                    value = msg.get('value')
                    role = msg.get('from')

                    # Check blank turns
                    if check_blank_turns:
                        if role == "system":
                            if value is not None and not isinstance(value, str):
                                has_blank_turn = True
                                break
                        else:
                            if not (isinstance(value, str) and value.strip()):
                                has_blank_turn = True
                                break

                    # Check invalid endings
                    if check_invalid_endings and value and ends_with_letter_number_comma(value):
                        has_invalid_ending = True
                        break

                    # Check null GPT responses
                    if check_null_gpt and role == 'gpt' and value is None:
                        has_null_gpt_value = True
                        break

                    # Check for duplicate system messages
                    if check_duplicate_system and role == 'system' and i < len(conversations) - 1:
                        next_msg = conversations[i + 1]
                        next_value = next_msg.get('value')
                        if (
                            next_msg.get('from') == 'human'
                            and value
                            and next_value
                            and value.strip().lower() == next_value.strip().lower()
                        ):
                            # Skip this system message (do not append)
                            continue

                    # Check for empty system role if not allowed
                    if role == "system" and not allow_empty_system_role and not value:
                        has_blank_turn = True
                        break

                    filtered_conversations.append(msg)

                # Skip conversations that fail checks
                if has_blank_turn or has_invalid_ending or has_null_gpt_value:
                    continue

                # NEW: optionally drop conversations with human→gpt duplicate turns
                if check_duplicate_turns and has_human_gpt_duplicate(filtered_conversations):
                    duplicate_turn_conv_count += 1
                    continue

                # Ensure valid roles exist in the conversation
                roles = set(msg.get('from') for msg in filtered_conversations)
                if 'human' in roles and 'gpt' in roles:
                    # Remove the last human message if it's the last one
                    if filtered_conversations and filtered_conversations[-1].get('from') == 'human':
                        filtered_conversations = filtered_conversations[:-1]
                    
                    # Write valid conversations
                    if filtered_conversations:
                        filtered_item = {"conversations": filtered_conversations}
                        json.dump(filtered_item, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        filtered_data_count += 1
        
        logging.info(f"Filtered data saved to {output_file}")
        logging.info(f"Original lines read             : {original_data_count}")
        logging.info(f"Lines dropped (JSON errors)     : {json_error_count}")
        logging.info(f"Conversations dropped (dups)    : {duplicate_turn_conv_count}")
        logging.info(f"Filtered size (written)         : {filtered_data_count}")

        summary = (
            f"Filtered data saved to {output_file}\n"
            f"Original lines read             : {original_data_count}\n"
            f"Lines dropped (JSON errors)     : {json_error_count}\n"
            f"Conversations dropped (dups)    : {duplicate_turn_conv_count}\n"
            f"Filtered size (written)         : {filtered_data_count}"
        )
        return summary

    except Exception as e:
        logging.error("Unexpected error during filtering", exc_info=True)
        raise ValueError(f"Error during filtering: {str(e)}")

def update_status(message):
    """Update status message."""
    print(message)

def update_progress(current, total):
    """Update progress."""
    progress = (current / total) * 100
    print(f"Progress: {progress:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Dataset filtering tool for conversations")
    parser.add_argument('input_file', type=str, help="Input JSONL file with conversations")
    parser.add_argument('output_dir', type=str, help="Output directory for filtered conversations")
    parser.add_argument('--check_blank_turns', action='store_true', default=True, help="Enable blank turn filtering")
    parser.add_argument('--check_invalid_endings', action='store_true', default=True, help="Enable invalid ending filtering")
    parser.add_argument('--check_null_gpt', action='store_true', default=True, help="Enable null GPT response filtering")
    parser.add_argument('--check_duplicate_system', action='store_true', default=True, help="Enable duplicate system message filtering")
    parser.add_argument('--allow_empty_system_role', action='store_true', default=True, help="Allow empty system role filtering")
    parser.add_argument('--check_duplicate_turns', action='store_true', default=True,
                        help="Enable duplicate human→gpt turn filtering")  # NEW

    args = parser.parse_args()

    # Perform filtering
    result = filter_dataset(
        input_path=args.input_file,
        output_dir=args.output_dir,
        check_blank_turns=args.check_blank_turns,
        check_invalid_endings=args.check_invalid_endings,
        check_null_gpt=args.check_null_gpt,
        check_duplicate_system=args.check_duplicate_system,
        allow_empty_system_role=args.allow_empty_system_role,
        check_duplicate_turns=args.check_duplicate_turns,
    )
    print(result)

if __name__ == "__main__":
    main()
