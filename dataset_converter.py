import json
import os

class DatasetConverter:
    @staticmethod
    def sanitize_file(input_path, output_path):
        """
        Remove special or hidden characters from a file and save it as UTF-8.
        """
        try:
            with open(input_path, 'rb') as f:
                content = f.read()

            # Filter out invalid bytes and decode to UTF-8
            sanitized_content = bytearray()
            for byte in content:
                try:
                    # Try to decode byte to UTF-8; if it fails, skip it
                    sanitized_content.append(byte)
                except UnicodeDecodeError:
                    continue  # Ignore invalid bytes

            with open(output_path, 'wb') as f:
                f.write(sanitized_content.decode('utf-8', errors='ignore').encode('utf-8'))

        except IOError as e:
            print(f"Error reading or writing file: {e}")

    @staticmethod
    def load_data(input_path):
        """
        Load data from a file based on its extension.
        """
        # Sanitize file before processing
        sanitized_path = input_path + '.sanitized'
        DatasetConverter.sanitize_file(input_path, sanitized_path)

        ext = os.path.splitext(sanitized_path)[1].lower()
        if ext == '.json':
            return DatasetConverter.load_json_data(sanitized_path)
        elif ext == '.jsonl':
            return DatasetConverter.load_jsonl_data(sanitized_path)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def load_json_data(input_path):
        """
        Load data from a JSON file, handling arrays of JSON objects and ignoring extra data.
        """
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
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
    def load_jsonl_data(input_path):
        """
        Load data from a JSONL file, handling each line as a separate JSON object.
        """
        data = []
        try:
            with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
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
    def detect_format(sample_text):
        """
        Detect the format of the dataset based on a sample text.
        """
        if 'system:' in sample_text or 'user:' in sample_text or 'assistant:' in sample_text:
            return 'plaintext'
        else:
            return 'unknown'

    @staticmethod
    def process_data(data, output_path):
        """
        Process data and write conversations to an output file.
        """
        preview_entries = []
        conversations_found = False
        detected_format = 'unknown'

        # Detect format based on a sample entry
        if data:
            sample_entry = json.dumps(data[0])
            detected_format = DatasetConverter.detect_format(sample_entry)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry)
                if conversations:
                    f.write(json.dumps({"conversations": conversations}) + '\n')
                    conversations_found = True
                    if len(preview_entries) < 3:
                        preview_entries.append({"conversations": conversations})

        status_message = f"Conversations completed successfully. Format detected: {detected_format}" if conversations_found else f"No conversations found for this dataset. Format detected: {detected_format}"
        print(status_message)

        DatasetConverter.validate_jsonl(output_path)

        return preview_entries

    @staticmethod
    def process_plaintext_line(line):
        """
        Convert a formatted plain text line into a list of conversations.
        """
        conversations = []
        for role in ['system', 'user', 'assistant']:
            if f'{role}:' in line:
                value = line.split(f'{role}:', 1)[1].strip()
                conversations.append({"from": role if role != 'assistant' else 'gpt', "value": value})
        return conversations

    @staticmethod
    def extract_conversations(entry):
        """
        Extract conversations from an entry based on different keys and roles.
        """
        conversations = []
        if 'conversations' in entry:
            for message in entry['conversations']:
                role = message.get('from')
                if role == 'user':
                    role = 'human'
                conversations.append({"from": role, "value": message.get('value', '')})
        else:
            if 'system' in entry:
                conversations.append({"from": "system", "value": entry['system']})
            if 'completion' in entry:
                DatasetConverter.process_completion(entry['completion'], conversations)
            elif 'messages' in entry:
                # Handle cases with "messages" instead of "completion"
                for message in entry.get('messages', []):
                    if isinstance(message, dict):
                        role = message.get('role')
                        if role == 'user':
                            role = 'human'
                        elif role == 'assistant':
                            role = 'gpt'
                        conversations.append({"from": role, "value": message.get('content', '')})
        return conversations

    @staticmethod
    def process_completion(completion, conversations):
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
    def add_conversation(message, conversations):
        """
        Add a conversation message to the list of conversations.
        """
        role = message.get('role')
        if role == 'user':
            role = 'human'
        elif role == 'assistant':
            role = 'gpt'
        conversations.append({"from": role, "value": message.get('content', '')})

    @staticmethod
    def fallback_parse_line(line):
        """
        Fallback method to handle lines that cannot be parsed as JSON.
        This method tries to infer a structured format from raw lines using fuzzy matching and string searches.
        """
        # Example of simple keyword-based parsing
        keywords = {
            'system': 'system:',
            'user': 'user:',
            'assistant': 'assistant:',
        }
        conversations = []
        
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
            conversations.append({"raw_text": line})
        
        return conversations

    @staticmethod
    def validate_jsonl(output_path):
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
