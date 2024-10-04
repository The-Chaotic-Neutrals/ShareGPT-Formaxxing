import json
import re
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)

def ends_with_letter_number_comma(text):
    return bool(re.search(r'[a-zA-Z0-9,]$', text.strip()))

def filter_dataset(input_path, output_dir):
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

                # Check for blank turns and invalid endings
                has_blank_turn = False
                has_invalid_ending = False
                for msg in conversations:
                    value = msg.get('value')
                    if value is None or not isinstance(value, str) or not value.strip():
                        has_blank_turn = True
                        break
                    if ends_with_letter_number_comma(value):
                        has_invalid_ending = True
                        break

                if has_blank_turn or has_invalid_ending:
                    continue  # Skip this conversation

                
                # Check if both roles are present
                roles = set(msg.get('from') for msg in conversations)
                if 'human' in roles and 'gpt' in roles:
                    # Remove the last human message if it's the last one
                    if conversations and conversations[-1].get('from') == 'human':
                        conversations = conversations[:-1]
                    
                    # Write filtered conversation to output file
                    filtered_item = {"conversations": conversations}
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
