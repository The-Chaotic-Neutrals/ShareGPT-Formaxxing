import json
import os
import re
from typing import Optional, List, Dict, Any

try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz

# Format identifiers
FORMAT_SHAREGPT = "sharegpt"
FORMAT_HUGGINGFACE = "huggingface"
FORMAT_VICUNA = "vicuna"
FORMAT_ALPACA = "alpaca"
FORMAT_CHATML = "chatml"
FORMAT_UNKNOWN = "unknown"


def value_with_prefix(message: Dict[str, Any]) -> str:
    value = message.get('value', '')
    if not isinstance(value, str):
        return ""

    prefix = message.get('prefix')
    value = value.strip()
    if not isinstance(prefix, str) or not prefix.strip():
        return value

    prefix = prefix.strip()
    if not value:
        return prefix
    if value.startswith(prefix):
        return value
    return f"{prefix} {value}"


class DatasetConverter:
    @staticmethod
    def load_data(input_path: str) -> list:
        """
        Load data from a file based on its extension.
        """
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.json':
            return DatasetConverter.load_json_data(input_path)
        elif ext == '.jsonl':
            return DatasetConverter.load_jsonl_data(input_path)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def load_json_data(input_path: str) -> list:
        """
        Load data from a JSON file, handling arrays of JSON objects and ignoring extra data.
        """
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                try:
                    data = json.loads(file_content)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    print("JSON Decode Error. Attempting to process line by line.")
                    lines = file_content.splitlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                json_object = json.loads(line)
                                if isinstance(json_object, dict):
                                    data.append(json_object)
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON line: {line}")
                                data.extend(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Ensure file is encoded in UTF-8.")
        return data

    @staticmethod
    def load_jsonl_data(input_path: str) -> list:
        """
        Load data from a JSONL file, handling each line as a separate JSON object.
        """
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line: {line}")
                            data.extend(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Ensure file is encoded in UTF-8.")
        return data

    @staticmethod
    def detect_format(data: List[Dict[str, Any]], sample_size: int = 5) -> str:
        """
        Automatically detect the format of the dataset by analyzing sample entries.
        Returns one of: sharegpt, huggingface, vicuna, alpaca, chatml, unknown
        """
        if not data:
            return FORMAT_UNKNOWN
        
        sample = data[:min(sample_size, len(data))]
        
        for entry in sample:
            if not isinstance(entry, dict):
                continue
            
            # Check for ShareGPT format
            if 'conversations' in entry:
                convs = entry.get('conversations', [])
                if isinstance(convs, list) and len(convs) > 0:
                    first_msg = convs[0] if isinstance(convs[0], dict) else {}
                    if 'from' in first_msg and 'value' in first_msg:
                        return FORMAT_SHAREGPT
            
            # Check for HuggingFace format (has "messages" with "role" and "content")
            if 'messages' in entry:
                msgs = entry.get('messages', [])
                if isinstance(msgs, list) and len(msgs) > 0:
                    first_msg = msgs[0] if isinstance(msgs[0], dict) else {}
                    if 'role' in first_msg and 'content' in first_msg:
                        return FORMAT_HUGGINGFACE
            
            # Check for Alpaca format (has "instruction", "input", "output")
            if 'instruction' in entry or ('input' in entry and 'output' in entry):
                return FORMAT_ALPACA
            
            # Check for Vicuna format (similar to ShareGPT but might have different structure)
            # Vicuna often has "conversations" but might use different role names
            if 'conversations' in entry:
                convs = entry.get('conversations', [])
                if isinstance(convs, list) and len(convs) > 0:
                    # Check if it looks like Vicuna (might have "from" with different values)
                    first_msg = convs[0] if isinstance(convs[0], dict) else {}
                    if 'from' in first_msg:
                        from_val = first_msg.get('from', '').lower()
                        if from_val in ['user', 'assistant', 'system', 'human', 'gpt']:
                            return FORMAT_VICUNA
            
            # Check for ChatML format (structured with special tokens or text format)
            # ChatML can be in messages format or as text with <|im_start|> tokens
            if 'messages' in entry:
                msgs = entry.get('messages', [])
                if isinstance(msgs, list):
                    # Check if any message has ChatML structure
                    for msg in msgs:
                        if isinstance(msg, dict):
                            content = str(msg.get('content', ''))
                            if '<|im_start|>' in content or '<|im_end|>' in content:
                                return FORMAT_CHATML
                            # ChatML might also be structured differently
                            if 'role' in msg and msg.get('role') in ['system', 'user', 'assistant']:
                                # Could be ChatML or HuggingFace, prefer HuggingFace if already detected
                                pass
            
            # Check for ChatML in text format (entire entry might be a string with tokens)
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if isinstance(value, str):
                        if '<|im_start|>' in value or '<|im_end|>' in value:
                            return FORMAT_CHATML
        
        # If we have messages but couldn't identify, default to HuggingFace-like
        for entry in sample:
            if isinstance(entry, dict) and 'messages' in entry:
                return FORMAT_HUGGINGFACE
        
        return FORMAT_UNKNOWN

    @staticmethod
    def convert_huggingface_to_sharegpt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert HuggingFace format to ShareGPT format.
        HuggingFace format: {"messages": [{"role": "user/assistant/system", "content": "..."}]}
        """
        conversations = []
        messages = entry.get('messages', [])
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            # Map roles to ShareGPT format
            if role == 'user':
                role = 'human'
            elif role == 'assistant':
                role = 'gpt'
            elif role == 'system':
                role = 'system'
            else:
                # Unknown role, try to infer
                if role in ['human', 'gpt']:
                    pass  # Already correct
                else:
                    continue  # Skip unknown roles
            
            if content and isinstance(content, str):
                conversations.append({
                    "from": role,
                    "value": content.strip()
                })
        
        return conversations

    @staticmethod
    def convert_vicuna_to_sharegpt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert Vicuna format to ShareGPT format.
        Vicuna format is similar to ShareGPT but might use different role names.
        """
        conversations = []
        convs = entry.get('conversations', [])
        
        for msg in convs:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get('from', '').lower()
            value = value_with_prefix(msg)
            
            # Normalize role names
            if role == 'user':
                role = 'human'
            elif role == 'assistant':
                role = 'gpt'
            elif role not in ['system', 'human', 'gpt']:
                # Try to infer from common variations
                if 'user' in role or 'human' in role:
                    role = 'human'
                elif 'assistant' in role or 'gpt' in role or 'bot' in role:
                    role = 'gpt'
                elif 'system' in role:
                    role = 'system'
                else:
                    continue
            
            if value:
                conversations.append({"from": role, "value": value})
        
        return conversations

    @staticmethod
    def convert_alpaca_to_sharegpt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert Alpaca format to ShareGPT format.
        Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
        """
        conversations = []
        
        instruction = entry.get('instruction', '').strip()
        input_text = entry.get('input', '').strip()
        output = entry.get('output', '').strip()
        
        # Combine instruction and input as the human message
        human_parts = []
        if instruction:
            human_parts.append(instruction)
        if input_text:
            human_parts.append(input_text)
        
        human_message = '\n'.join(human_parts).strip()
        
        if human_message:
            conversations.append({
                "from": "human",
                "value": human_message
            })
        
        if output:
            conversations.append({
                "from": "gpt",
                "value": output
            })
        
        return conversations

    @staticmethod
    def convert_chatml_to_sharegpt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert ChatML format to ShareGPT format.
        ChatML can be in multiple forms:
        1. Text with tokens: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>"
        2. Structured messages: {"messages": [{"role": "...", "content": "..."}]}
        """
        conversations = []
        
        # Check if it's text-based ChatML
        text_content = None
        if isinstance(entry, dict):
            # Look for text content in common fields
            for key in ['text', 'content', 'prompt', 'messages']:
                if key in entry:
                    val = entry[key]
                    if isinstance(val, str) and ('<|im_start|>' in val or '<|im_end|>' in val):
                        text_content = val
                        break
                    elif isinstance(val, list):
                        # Check if messages contain ChatML tokens
                        for item in val:
                            if isinstance(item, dict):
                                content = item.get('content', '')
                                if isinstance(content, str) and '<|im_start|>' in content:
                                    text_content = content
                                    break
        
        if text_content:
            # Parse ChatML text format
            # Pattern: <|im_start|>role\ncontent<|im_end|>
            pattern = r'<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>'
            matches = re.findall(pattern, text_content, re.DOTALL)
            
            for role, content in matches:
                role = role.strip().lower()
                content = content.strip()
                
                # Map roles
                if role == 'user':
                    role = 'human'
                elif role == 'assistant':
                    role = 'gpt'
                elif role not in ['system', 'human', 'gpt']:
                    continue
                
                if content:
                    conversations.append({
                        "from": role,
                        "value": content
                    })
        else:
            # Try structured format (similar to HuggingFace)
            conversations = DatasetConverter.convert_huggingface_to_sharegpt(entry)
        
        return conversations

    @staticmethod
    def extract_conversations(entry: dict, detected_format: Optional[str] = None) -> list:
        """
        Extract conversations from an entry based on detected format.
        Ensures all outputs follow ShareGPT format: {"from": "system/human/gpt", "value": "..."}
        """
        # Auto-detect format if not provided
        if detected_format is None:
            sample_data = [entry] if isinstance(entry, dict) else []
            detected_format = DatasetConverter.detect_format(sample_data, sample_size=1)
        
        conversations = []
        
        # Use format-specific converter
        if detected_format == FORMAT_SHAREGPT:
            # Already in ShareGPT format, just normalize
            if 'conversations' in entry:
                for message in entry['conversations']:
                    if isinstance(message, dict):
                        role = message.get('from', '').lower()
                        if role == 'user':
                            role = 'human'
                        elif role == 'assistant':
                            role = 'gpt'
                        value = value_with_prefix(message)
                        if value:
                            conversations.append({"from": role, "value": value})
        
        elif detected_format == FORMAT_HUGGINGFACE:
            conversations = DatasetConverter.convert_huggingface_to_sharegpt(entry)
        
        elif detected_format == FORMAT_VICUNA:
            conversations = DatasetConverter.convert_vicuna_to_sharegpt(entry)
        
        elif detected_format == FORMAT_ALPACA:
            conversations = DatasetConverter.convert_alpaca_to_sharegpt(entry)
        
        elif detected_format == FORMAT_CHATML:
            conversations = DatasetConverter.convert_chatml_to_sharegpt(entry)
        
        else:
            # Fallback to original extraction logic
            if 'conversations' in entry:
                for message in entry['conversations']:
                    role = message.get('from')
                    if role == 'user':
                        role = 'human'
                    value = value_with_prefix(message)
                    conversations.append({"from": role if role != 'assistant' else 'gpt', "value": value})
            else:
                if 'system' in entry:
                    conversations.append({"from": "system", "value": entry['system'].strip()})
                if 'completion' in entry:
                    DatasetConverter.process_completion(entry['completion'], conversations)
                elif 'messages' in entry:
                    for message in entry.get('messages', []):
                        if isinstance(message, dict):
                            role = message.get('role')
                            if role == 'user':
                                role = 'human'
                            elif role == 'assistant':
                                role = 'gpt'
                            conversations.append({"from": role, "value": message.get('content', '').strip()})
        
        # Ensure the output follows the specified format
        if not conversations:
            return [{"from": "system", "value": "No conversations found."}]
        
        return conversations

    @staticmethod
    def process_completion(completion: dict, conversations: list):
        """
        Process completion data and add it to the list of conversations.
        """
        if isinstance(completion, list):
            for message in completion:
                DatasetConverter.add_conversation(message, conversations)
        elif isinstance(completion, str):
            try:
                completion_json = json.loads(completion)
                if isinstance(completion_json, list):
                    for message in completion_json:
                        DatasetConverter.add_conversation(message, conversations)
            except json.JSONDecodeError:
                pass

    @staticmethod
    def add_conversation(message: dict, conversations: list):
        """
        Add a conversation message to the list of conversations.
        Ensures the output format is consistently:
        {"from": "system/human/gpt", "value": "text"}
        """
        role = message.get('role')
        if role == 'user':
            role = 'human'
        elif role == 'assistant':
            role = 'gpt'
        conversations.append({"from": role, "value": message.get('content', '').strip()})

    @staticmethod
    def fallback_parse_line(line: str) -> list:
        """
        Fallback method to handle lines that cannot be parsed as JSON.
        This method tries to infer a structured format from raw lines using fuzzy matching and string searches.
        """
        conversations = []
        # Example of simple keyword-based parsing
        keywords = {
            'system': 'system:',
            'user': 'user:',
            'assistant': 'assistant:',
        }
        
        for role, keyword in keywords.items():
            if keyword in line:
                value = line.split(keyword, 1)[1].strip()
                conversations.append({"from": role if role != 'assistant' else 'gpt', "value": value})
        
        # Fuzzy matching fallback
        if not conversations:
            potential_roles = ['system', 'user', 'assistant']
            for role in potential_roles:
                # Look for a close match to known roles
                ratio = fuzz.ratio(line.lower(), role)
                if ratio > 70:  # Adjust threshold as needed
                    conversations.append({"from": role if role != 'assistant' else 'gpt', "value": line.strip()})
                    break
        
        if not conversations:
            # Default case if no structured information found
            conversations.append({"from": "unknown", "value": line.strip()})  # Ensure format consistency
        
        return conversations

    @staticmethod
    def validate_jsonl(output_path: str):
        """
        Validate the final output to ensure it is proper JSONL.
        """
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON at line {i}: {line}")
                        raise ValueError(f"Invalid JSONL format detected at line {i}.")
        print("Validation completed: The output is proper JSONL.")

    @staticmethod
    def write_jsonl_rows(rows, output_path: str):
        """Write JSONL with the same spacing as the original conversion path."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def process_data(data: list, output_path: str, detected_format: Optional[str] = None) -> tuple:
        """
        Process data and write conversations to an output file.
        Each line will start with the format {"conversations": [...]}.
        Returns: (preview_entries, detected_format)
        """
        # Detect format from data if not provided
        if detected_format is None:
            detected_format = DatasetConverter.detect_format(data, sample_size=min(10, len(data)))
            print(f"Detected format: {detected_format}")
        
        preview_entries = []
        conversations_found = False

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry, detected_format)
                # Create the formatted entry with the required prefix
                formatted_entry = {"conversations": conversations}
                # Write it to the file with the required prefix
                f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')  # Write without escaping non-ASCII
                conversations_found = True
                if len(preview_entries) < 3:
                    preview_entries.append(formatted_entry)  # Store formatted entry for preview

        status_message = (
            "Conversations completed successfully."
            if conversations_found
            else "No conversations found for this dataset."
        )
        print(status_message)

        DatasetConverter.validate_jsonl(output_path)

        return preview_entries, detected_format

    @staticmethod
    def process_multiple_files(input_paths: list, output_dir: str) -> dict:
        """
        Process multiple files separately and output them as separate JSONL files.
        Returns dict with format: {filename: (preview_entries, detected_format)}
        """
        preview_entries = {}
        for input_path in input_paths:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jsonl")
            print(f"Processing file: {filename}")
            if os.path.splitext(input_path)[1].lower() == ".parquet":
                preview, detected_format = DatasetConverter.process_parquet_data(input_path, output_path)
            else:
                data = DatasetConverter.load_data(input_path)
                preview, detected_format = DatasetConverter.process_data(data, output_path)
            preview_entries[filename] = (preview, detected_format)
        return preview_entries

    @staticmethod
    def process_parquet_data(input_path: str, output_path: str) -> tuple:
        """
        Convert a Parquet dataset to ShareGPT JSONL through HF Datasets/Arrow.
        """
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=input_path, split="train")
        sample = [dataset[i] for i in range(min(10, len(dataset)))]
        detected_format = DatasetConverter.detect_format(sample, sample_size=len(sample))
        print(f"Detected format: {detected_format}")

        def convert_row(entry):
            return {"conversations": DatasetConverter.extract_conversations(entry, detected_format)}

        converted = dataset.map(convert_row, remove_columns=dataset.column_names, desc="Converting Parquet to ShareGPT")
        preview_entries = [converted[i] for i in range(min(3, len(converted)))]
        DatasetConverter.write_jsonl_rows(converted, output_path)
        DatasetConverter.validate_jsonl(output_path)
        return preview_entries, detected_format

# =============================================================================
# Dataset Filtering (formerly DataMaxxer)
# =============================================================================

FILTER_STATS_TEMPLATE = {
    "input_size_bytes": 0,
    "output_size_bytes": 0,
    "kept_source_bytes": 0,
    "json_error_drop_bytes": 0,
    "blank_turn_drop_bytes": 0,
    "invalid_ending_drop_bytes": 0,
    "null_gpt_drop_bytes": 0,
    "missing_role_drop_bytes": 0,
    "empty_after_cleanup_drop_bytes": 0,
    "duplicate_drop_bytes": 0,
    "original_data_count": 0,
    "json_error_count": 0,
    "blank_turn_drop_count": 0,
    "invalid_ending_drop_count": 0,
    "null_gpt_drop_count": 0,
    "missing_role_drop_count": 0,
    "empty_after_cleanup_drop_count": 0,
    "duplicate_turn_conv_count": 0,
    "duplicate_exact_conv_count": 0,
    "duplicate_near_conv_count": 0,
    "filtered_data_count": 0,
}


def format_bytes(size_bytes):
    size = float(size_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


def _jsonl_line_sizes(input_path):
    sizes = []
    with open(input_path, 'rb') as f:
        for line in f:
            if line.strip():
                sizes.append(len(line))
    return sizes

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


def normalize_for_similarity(text):
    """
    Normalize text for fuzzy/near-duplicate checks.
    """
    text = normalize_text(text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_human_gpt_duplicate_type(conversations, similarity_threshold=92):
    """
    Return the duplicate type for a human->gpt pair:
    - "exact": normalized texts are identical
    - "near": highly similar copy-with-small-edits
    - None: no duplicate match found
    """
    threshold = max(0, min(100, int(similarity_threshold)))
    partial_threshold = min(100, threshold + 4)

    for i in range(len(conversations) - 1):
        cur_msg = conversations[i]
        next_msg = conversations[i + 1]

        if cur_msg.get("from") == "human" and next_msg.get("from") == "gpt":
            cur_val = normalize_for_similarity(cur_msg.get("value", ""))
            next_val = normalize_for_similarity(next_msg.get("value", ""))
            if not cur_val or not next_val:
                continue

            if cur_val == next_val:
                return "exact"

            shorter_len = min(len(cur_val), len(next_val))
            longer_len = max(len(cur_val), len(next_val))
            length_ratio = shorter_len / longer_len if longer_len else 0

            token_ratio = fuzz.token_sort_ratio(cur_val, next_val)
            partial_ratio = fuzz.partial_ratio(cur_val, next_val)

            # Catch "copy with tiny tweaks" patterns while avoiding big rewrites.
            if length_ratio >= 0.7 and (token_ratio >= threshold or partial_ratio >= partial_threshold):
                return "near"

            if length_ratio >= 0.8 and (next_val.startswith(cur_val) or cur_val.startswith(next_val)):
                return "near"
    return None


def has_human_gpt_duplicate(conversations, similarity_threshold=92):
    """
    Return True if any human turn is duplicated by the following gpt turn.
    Includes exact and near-duplicate checks.
    """
    return get_human_gpt_duplicate_type(conversations, similarity_threshold) is not None


def clean_conversation_item(
    item,
    check_blank_turns=True,
    check_invalid_endings=True,
    check_null_gpt=True,
    check_duplicate_system=True,
    allow_empty_system_role=True,
    check_duplicate_turns=True,
    duplicate_similarity_threshold=92,
):
    conversations = item.get("conversations", []) if hasattr(item, "get") else []
    has_blank_turn = False
    has_invalid_ending = False
    has_null_gpt_value = False

    filtered_conversations = []
    for i, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            has_blank_turn = True
            break

        value = value_with_prefix(msg)
        role = msg.get('from')

        if check_blank_turns:
            if role == "system":
                if value is not None and not isinstance(value, str):
                    has_blank_turn = True
                    break
            else:
                if not (isinstance(value, str) and value.strip()):
                    has_blank_turn = True
                    break

        if check_invalid_endings and value and ends_with_letter_number_comma(value):
            has_invalid_ending = True
            break

        if check_null_gpt and role == 'gpt' and value is None:
            has_null_gpt_value = True
            break

        if check_duplicate_system and role == 'system' and i < len(conversations) - 1:
            next_msg = conversations[i + 1]
            next_value = next_msg.get('value') if isinstance(next_msg, dict) else None
            if (
                isinstance(next_msg, dict)
                and next_msg.get('from') == 'human'
                and value
                and next_value
                and value.strip().lower() == next_value.strip().lower()
            ):
                continue

        if role == "system" and not allow_empty_system_role and not value:
            has_blank_turn = True
            break

        filtered_conversations.append({"from": role, "value": value})

    if has_blank_turn:
        return None, "blank_turn"
    if has_invalid_ending:
        return None, "invalid_ending"
    if has_null_gpt_value:
        return None, "null_gpt"

    if check_duplicate_turns:
        duplicate_type = get_human_gpt_duplicate_type(
            filtered_conversations,
            similarity_threshold=duplicate_similarity_threshold,
        )
        if duplicate_type:
            return None, f"duplicate_{duplicate_type}"

    roles = set(msg.get('from') for msg in filtered_conversations if isinstance(msg, dict))
    if 'human' not in roles or 'gpt' not in roles:
        return None, "missing_roles"

    if filtered_conversations and filtered_conversations[-1].get('from') == 'human':
        filtered_conversations = filtered_conversations[:-1]

    if not filtered_conversations:
        return None, "empty_after_cleanup"

    return {"conversations": filtered_conversations}, None


def _record_drop_reason(stats, reason, source_bytes=0):
    if reason == "duplicate_exact":
        stats["duplicate_turn_conv_count"] += 1
        stats["duplicate_exact_conv_count"] += 1
        stats["duplicate_drop_bytes"] += source_bytes
    elif reason == "duplicate_near":
        stats["duplicate_turn_conv_count"] += 1
        stats["duplicate_near_conv_count"] += 1
        stats["duplicate_drop_bytes"] += source_bytes
    elif reason == "blank_turn":
        stats["blank_turn_drop_count"] += 1
        stats["blank_turn_drop_bytes"] += source_bytes
    elif reason == "invalid_ending":
        stats["invalid_ending_drop_count"] += 1
        stats["invalid_ending_drop_bytes"] += source_bytes
    elif reason == "null_gpt":
        stats["null_gpt_drop_count"] += 1
        stats["null_gpt_drop_bytes"] += source_bytes
    elif reason == "missing_roles":
        stats["missing_role_drop_count"] += 1
        stats["missing_role_drop_bytes"] += source_bytes
    elif reason == "empty_after_cleanup":
        stats["empty_after_cleanup_drop_count"] += 1
        stats["empty_after_cleanup_drop_bytes"] += source_bytes


def _filter_dataset_with_hf_datasets(
    input_path,
    output_dir,
    check_blank_turns=True,
    check_invalid_endings=True,
    check_null_gpt=True,
    check_duplicate_system=True,
    allow_empty_system_role=True,
    check_duplicate_turns=True,
    duplicate_similarity_threshold=92,
):
    from pathlib import Path
    import tempfile
    from datasets import load_dataset

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=str(input_path), split="train")
        output_file = output_dir / input_path.name
        write_format = "parquet"
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(input_path), split="train")
        if suffix == ".jsonl":
            line_sizes = _jsonl_line_sizes(input_path)
            if len(line_sizes) == len(dataset):
                dataset = dataset.add_column("__formaxxer_source_size_bytes", line_sizes)
        output_file = output_dir / input_path.name
        write_format = "jsonl"
    else:
        raise ValueError(f"Unsupported filter input format: {suffix}")

    stats = dict(FILTER_STATS_TEMPLATE)
    stats["input_size_bytes"] = input_path.stat().st_size if input_path.exists() else 0
    stats["original_data_count"] = len(dataset)

    def map_row(item):
        source_bytes = item.get("__formaxxer_source_size_bytes") if hasattr(item, "get") else None
        if source_bytes is None:
            source_bytes = len(json.dumps(dict(item), ensure_ascii=False).encode("utf-8")) if hasattr(item, "items") else 0
        cleaned, drop_reason = clean_conversation_item(
            item,
            check_blank_turns=check_blank_turns,
            check_invalid_endings=check_invalid_endings,
            check_null_gpt=check_null_gpt,
            check_duplicate_system=check_duplicate_system,
            allow_empty_system_role=allow_empty_system_role,
            check_duplicate_turns=check_duplicate_turns,
            duplicate_similarity_threshold=duplicate_similarity_threshold,
        )

        return {
            "conversations": cleaned["conversations"] if cleaned else [],
            "__drop_reason": drop_reason or "",
            "__formaxxer_source_size_bytes": int(source_bytes or 0),
        }

    mapped = dataset.map(map_row, desc="Filtering conversations")
    drop_reasons = mapped["__drop_reason"]
    source_sizes = mapped["__formaxxer_source_size_bytes"]
    for drop_reason, source_bytes in zip(drop_reasons, source_sizes):
        if drop_reason:
            _record_drop_reason(stats, drop_reason, source_bytes)
        else:
            stats["kept_source_bytes"] += source_bytes

    filtered = mapped.filter(lambda row: row["__drop_reason"] == "", desc="Keeping valid conversations")
    columns_to_remove = [column for column in filtered.column_names if column != "conversations"]
    if columns_to_remove:
        filtered = filtered.remove_columns(columns_to_remove)
    stats["filtered_data_count"] = len(filtered)

    temp_output_file = None
    writing_in_place = output_file.resolve() == input_path.resolve()
    write_target = output_file
    if writing_in_place:
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f"{input_path.stem}_filtered_",
            suffix=input_path.suffix,
            dir=str(output_dir),
        )
        os.close(temp_fd)
        temp_output_file = Path(temp_path)
        write_target = temp_output_file

    try:
        if write_format == "parquet":
            filtered.to_parquet(str(write_target))
        else:
            DatasetConverter.write_jsonl_rows(filtered, str(write_target))

        if writing_in_place and temp_output_file is not None:
            os.replace(str(temp_output_file), str(output_file))
        stats["output_size_bytes"] = output_file.stat().st_size if output_file.exists() else 0
    except Exception:
        if temp_output_file is not None and temp_output_file.exists():
            try:
                temp_output_file.unlink()
            except Exception:
                pass
        raise

    return _format_filter_summary(output_file, stats, duplicate_similarity_threshold), str(output_file)


def _format_filter_summary(output_file, stats, duplicate_similarity_threshold):
    rule_drop_count = (
        stats['blank_turn_drop_count']
        + stats['invalid_ending_drop_count']
        + stats['null_gpt_drop_count']
        + stats['missing_role_drop_count']
        + stats['empty_after_cleanup_drop_count']
    )
    rule_drop_bytes = (
        stats['blank_turn_drop_bytes']
        + stats['invalid_ending_drop_bytes']
        + stats['null_gpt_drop_bytes']
        + stats['missing_role_drop_bytes']
        + stats['empty_after_cleanup_drop_bytes']
    )
    known_source_bytes = stats['kept_source_bytes'] + rule_drop_bytes + stats['duplicate_drop_bytes'] + stats['json_error_drop_bytes']
    source_accounting = ""
    if known_source_bytes:
        kept_pct = (stats['kept_source_bytes'] / known_source_bytes) * 100
        dropped_pct = ((rule_drop_bytes + stats['duplicate_drop_bytes'] + stats['json_error_drop_bytes']) / known_source_bytes) * 100
        source_accounting = (
            f"Source bytes kept               : {format_bytes(stats['kept_source_bytes'])} ({kept_pct:.1f}%)\n"
            f"Source bytes dropped            : {format_bytes(rule_drop_bytes + stats['duplicate_drop_bytes'] + stats['json_error_drop_bytes'])} ({dropped_pct:.1f}%)\n"
            f"  Dropped by rules              : {format_bytes(rule_drop_bytes)}\n"
            f"  Dropped by duplicates         : {format_bytes(stats['duplicate_drop_bytes'])}\n"
            f"  Dropped by JSON errors        : {format_bytes(stats['json_error_drop_bytes'])}\n"
        )
    summary = (
        f"Filtered data saved to {output_file}\n"
        f"Input size                      : {format_bytes(stats['input_size_bytes'])}\n"
        f"Output size                     : {format_bytes(stats['output_size_bytes'])}\n"
        f"{source_accounting}"
        f"Original lines read             : {stats['original_data_count']}\n"
        f"Lines dropped (JSON errors)     : {stats['json_error_count']}\n"
        f"Conversations dropped (rules)   : {rule_drop_count}\n"
        f"  Blank/invalid turns           : {stats['blank_turn_drop_count']}\n"
        f"  Invalid endings               : {stats['invalid_ending_drop_count']}\n"
        f"  Null GPT responses            : {stats['null_gpt_drop_count']}\n"
        f"  Missing human/gpt roles       : {stats['missing_role_drop_count']}\n"
        f"  Empty after cleanup           : {stats['empty_after_cleanup_drop_count']}\n"
        f"Conversations dropped (dups)    : {stats['duplicate_turn_conv_count']}\n"
        f"  Exact duplicate drops         : {stats['duplicate_exact_conv_count']}\n"
        f"  Near duplicate drops          : {stats['duplicate_near_conv_count']}\n"
        f"  Similarity threshold used     : {max(0, min(100, int(duplicate_similarity_threshold)))}\n"
        f"Filtered size (written)         : {stats['filtered_data_count']}"
    )
    print(summary)
    return summary


def filter_dataset(
    input_path,
    output_dir,
    check_blank_turns=True,
    check_invalid_endings=True,
    check_null_gpt=True,
    check_duplicate_system=True,
    allow_empty_system_role=True,
    check_duplicate_turns=True,
    duplicate_similarity_threshold=92,
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
        duplicate_similarity_threshold (int): Similarity threshold for duplicate human→gpt checks (0-100).

    Returns:
        tuple: (summary string, output_file_path)
    """
    from pathlib import Path
    import tempfile
    
    try:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        if input_path.suffix.lower() in {".json", ".jsonl", ".parquet"}:
            try:
                return _filter_dataset_with_hf_datasets(
                    input_path,
                    output_dir,
                    check_blank_turns=check_blank_turns,
                    check_invalid_endings=check_invalid_endings,
                    check_null_gpt=check_null_gpt,
                    check_duplicate_system=check_duplicate_system,
                    allow_empty_system_role=allow_empty_system_role,
                    check_duplicate_turns=check_duplicate_turns,
                    duplicate_similarity_threshold=duplicate_similarity_threshold,
                )
            except Exception as hf_error:
                if input_path.suffix.lower() == ".parquet":
                    raise
                print(f"HF Datasets filtering unavailable; falling back to streaming JSONL: {hf_error}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / input_path.name
        writing_in_place = output_file.resolve() == input_path.resolve()

        temp_output_file = None
        if writing_in_place:
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=f"{input_path.stem}_filtered_",
                suffix=".jsonl",
                dir=str(output_dir),
            )
            os.close(temp_fd)
            temp_output_file = Path(temp_path)
            write_target = temp_output_file
        else:
            write_target = output_file
        
        stats = dict(FILTER_STATS_TEMPLATE)
        stats["input_size_bytes"] = input_path.stat().st_size if input_path.exists() else 0

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             write_target.open('w', encoding='utf-8') as outfile:
            
            for line in infile:
                stats["original_data_count"] += 1
                source_bytes = len(line.encode('utf-8'))
                line = line.strip()

                if not line:
                    continue  # Skip empty lines

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    stats["json_error_count"] += 1
                    stats["json_error_drop_bytes"] += source_bytes
                    print(f"JSON decode error at line {stats['original_data_count']}: {e}")
                    continue

                filtered_item, drop_reason = clean_conversation_item(
                    item,
                    check_blank_turns=check_blank_turns,
                    check_invalid_endings=check_invalid_endings,
                    check_null_gpt=check_null_gpt,
                    check_duplicate_system=check_duplicate_system,
                    allow_empty_system_role=allow_empty_system_role,
                    check_duplicate_turns=check_duplicate_turns,
                    duplicate_similarity_threshold=duplicate_similarity_threshold,
                )
                if drop_reason:
                    _record_drop_reason(stats, drop_reason, source_bytes)
                    continue
                if filtered_item:
                    json.dump(filtered_item, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    stats["filtered_data_count"] += 1
                    stats["kept_source_bytes"] += source_bytes
        
        if writing_in_place and temp_output_file is not None:
            os.replace(str(temp_output_file), str(output_file))
        stats["output_size_bytes"] = output_file.stat().st_size if output_file.exists() else 0

        return _format_filter_summary(output_file, stats, duplicate_similarity_threshold), str(output_file)

    except Exception as e:
        if 'temp_output_file' in locals() and temp_output_file is not None and temp_output_file.exists():
            try:
                temp_output_file.unlink()
            except Exception:
                pass
        print(f"Unexpected error during filtering: {e}")
        raise ValueError(f"Error during filtering: {str(e)}")


# Example usage
if __name__ == "__main__":
    input_paths = ["input1.json", "input2.json"]  # List of input files
    output_dir = "output"  # Directory where separate output files will be saved
    converter = DatasetConverter()
    preview = converter.process_multiple_files(input_paths, output_dir)
    print("Preview of processed conversations:", preview)
