import json
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Adjusted regular expression to match exactly two letters inside brackets, e.g., [xx]
tag_pattern = re.compile(r'\[[a-zA-Z]{2}\]')

def ends_with_letter_number_comma(text):
    # Check if text is a valid string before processing
    if isinstance(text, str):
        return bool(re.search(r'[a-zA-Z0-9,]$', text.strip()))
    return False  # Return False if text is None or not a string

def remove_two_letter_tags(text):
    # Use regular expression to replace two-letter tags inside square brackets
    if isinstance(text, str):
        return tag_pattern.sub('', text)
    return text

def filter_dataset(input_path, output_dir, 
                   check_blank_turns=True, 
                   check_invalid_endings=True, 
                   check_null_gpt=True, 
                   check_deleted_by_user=True, 
                   check_duplicate_system=True, 
                   remove_two_letter_tags_flag=True):
    
    try:
        # Prepare paths
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        filtered_dir = output_dir / "filtered"
        filtered_dir.mkdir(exist_ok=True)
        output_file = filtered_dir / f"{input_path.stem}_filtered.jsonl"
        
        filtered_data_count = 0
        original_data_count = 0

        # Use 'utf-8' with error ignoring to handle any non-decodable characters
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
                    # Log the error and skip the problematic line
                    logging.error(f"JSON decode error at line {original_data_count}: {e}")
                    continue
                
                # Process conversations
                conversations = item.get("conversations", [])

                # Track reasons for filtering out conversations
                has_blank_turn = False
                has_invalid_ending = False
                has_null_gpt_value = False
                has_deleted_by_user = False
                
                filtered_conversations = []
                for i, msg in enumerate(conversations):
                    value = msg.get('value')
                    role = msg.get('from')

                    # Remove two-letter tags like [xx] from the message (if enabled)
                    if remove_two_letter_tags_flag and value:
                        value = remove_two_letter_tags(value)
                        msg['value'] = value

                    # Check if the message contains "[deleted by user]" (if enabled)
                    if check_deleted_by_user and value and "[deleted by user]" in value:
                        has_deleted_by_user = True
                        break

                    # Check for blank or non-string values (if enabled)
                    if check_blank_turns and (value is None or not isinstance(value, str) or not value.strip()):
                        has_blank_turn = True
                        break

                    # Check for invalid endings (if enabled)
                    if check_invalid_endings and value and ends_with_letter_number_comma(value):
                        has_invalid_ending = True
                        break

                    # Check for null 'gpt' value (if enabled)
                    if check_null_gpt and role == 'gpt' and value is None:
                        has_null_gpt_value = True
                        break

                    # Remove duplicate system turns if the next human turn has the same value (if enabled)
                    if check_duplicate_system and role == 'system' and i < len(conversations) - 1:
                        next_msg = conversations[i + 1]
                        next_value = next_msg.get('value')

                        # Ensure both values are cleaned (strip and lower) and check if they are identical
                        if next_msg.get('from') == 'human' and value and next_value and value.strip().lower() == next_value.strip().lower():
                            continue  # Skip this system message
                        
                    # Add non-duplicate, valid messages to the filtered list
                    filtered_conversations.append(msg)

                # Skip conversations that fail any of the checks
                if has_blank_turn or has_invalid_ending or has_null_gpt_value or has_deleted_by_user:
                    continue

                # Check if both roles are present
                roles = set(msg.get('from') for msg in filtered_conversations)
                if 'human' in roles and 'gpt' in roles:
                    # Remove the last human message if it's the last one
                    if filtered_conversations and filtered_conversations[-1].get('from') == 'human':
                        filtered_conversations = filtered_conversations[:-1]
                    
                    # Write filtered conversation to output file
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
