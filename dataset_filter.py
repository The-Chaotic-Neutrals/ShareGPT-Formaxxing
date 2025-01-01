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

def filter_dataset(input_path, output_dir, 
                   check_blank_turns=True, 
                   check_invalid_endings=True, 
                   check_null_gpt=True, 
                   check_duplicate_system=True, 
                   allow_empty_system_role=True):
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

    Returns:
        str: Summary of filtering results.
    """
    try:
        # Prepare paths
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        filtered_dir = output_dir / "filtered"
        filtered_dir.mkdir(exist_ok=True)
        output_file = filtered_dir / f"{input_path.stem}_filtered.jsonl"
        
        filtered_data_count = 0
        original_data_count = 0

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
                        if next_msg.get('from') == 'human' and value and next_value and value.strip().lower() == next_value.strip().lower():
                            continue

                    # Check for empty system role if allowed
                    if role == "system" and not allow_empty_system_role and not value:
                        has_blank_turn = True
                        break

                    filtered_conversations.append(msg)

                # Skip conversations that fail checks
                if has_blank_turn or has_invalid_ending or has_null_gpt_value:
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
        logging.info(f"Original size: {original_data_count}")
        logging.info(f"Filtered size: {filtered_data_count}")

        return f"Filtered data saved to {output_file}\nOriginal size: {original_data_count}\nFiltered size: {filtered_data_count}"

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

    args = parser.parse_args()

    # Perform filtering
    result = filter_dataset(
        input_path=args.input_file,
        output_dir=args.output_dir,
        check_blank_turns=args.check_blank_turns,
        check_invalid_endings=args.check_invalid_endings,
        check_null_gpt=args.check_null_gpt,
        check_duplicate_system=args.check_duplicate_system,
        allow_empty_system_role=args.allow_empty_system_role
    )
    print(result)

if __name__ == "__main__":
    main()
