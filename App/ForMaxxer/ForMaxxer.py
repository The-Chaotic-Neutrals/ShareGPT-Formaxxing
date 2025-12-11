import json
import os
import re
from fuzzywuzzy import fuzz
from typing import Optional, List, Dict, Any

# Format identifiers
FORMAT_SHAREGPT = "sharegpt"
FORMAT_HUGGINGFACE = "huggingface"
FORMAT_VICUNA = "vicuna"
FORMAT_ALPACA = "alpaca"
FORMAT_CHATML = "chatml"
FORMAT_UNKNOWN = "unknown"


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
            value = msg.get('value', '')
            
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
            
            if value and isinstance(value, str):
                conversations.append({
                    "from": role,
                    "value": value.strip()
                })
        
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
                        value = message.get('value', '')
                        if isinstance(value, str):
                            conversations.append({"from": role, "value": value.strip()})
        
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
                    conversations.append({"from": role if role != 'assistant' else 'gpt', "value": message.get('value', '').strip()})
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
            data = DatasetConverter.load_data(input_path)
            preview, detected_format = DatasetConverter.process_data(data, output_path)
            preview_entries[filename] = (preview, detected_format)
        return preview_entries

# Example usage
if __name__ == "__main__":
    input_paths = ["input1.json", "input2.json"]  # List of input files
    output_dir = "output"  # Directory where separate output files will be saved
    converter = DatasetConverter()
    preview = converter.process_multiple_files(input_paths, output_dir)
    print("Preview of processed conversations:", preview)
