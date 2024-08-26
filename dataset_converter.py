import json
import pandas as pd
import sqlite3
import csv
import os
import re

class DatasetConverter:
    @staticmethod
    def load_data(input_path):
        """
        Load data from a file based on its extension.
        """
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.json':
            return DatasetConverter.load_json_data(input_path)
        elif ext == '.jsonl':
            return DatasetConverter.load_jsonl_data(input_path)
        elif ext == '.parquet':
            return DatasetConverter.load_parquet_data(input_path)
        elif ext == '.txt':
            return DatasetConverter.load_txt_data(input_path)
        elif ext == '.csv':
            return DatasetConverter.load_csv_data(input_path)
        elif ext == '.sql':
            return DatasetConverter.load_sql_data(input_path)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def load_json_data(input_path):
        """
        Load data from a JSON file, handling arrays of JSON objects and ignoring extra data.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            try:
                # Attempt to parse the file content as a single JSON object or array
                data = json.loads(file_content)
                if not isinstance(data, list):
                    data = [data]  # Wrap the single object in a list if it's not already a list
            except json.JSONDecodeError:
                print("JSON Decode Error. Attempting to process line by line.")
                # Process each line individually if JSON array parsing fails
                lines = file_content.splitlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            # Parse each line as a separate JSON object
                            json_object = json.loads(line)
                            if isinstance(json_object, dict):
                                data.append(json_object)
                        except json.JSONDecodeError:
                            # Ignore invalid JSON lines and log the issue
                            print(f"Skipping invalid JSON line: {line}")
                            # Attempt fallback parsing
                            data.append(DatasetConverter.fallback_parse_line(line))
        return data

    @staticmethod
    def fallback_parse_line(line):
        """
        Attempt to extract structured data from non-JSON lines.
        """
        conversations = DatasetConverter.process_plaintext_line(line)
        if conversations:
            return {"conversations": conversations}
        # Fallback to handling structured text patterns if applicable
        return {"raw_text": line}

    @staticmethod
    def load_jsonl_data(input_path):
        """
        Load data from a JSONL file, handling each line as a separate JSON object.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Ignore invalid JSON lines
                        print(f"Skipping invalid JSON line: {line}")
                        # Attempt fallback parsing
                        data.append(DatasetConverter.fallback_parse_line(line))
        return data

    @staticmethod
    def load_parquet_data(input_path):
        """
        Load data from a Parquet file.
        """
        return pd.read_parquet(input_path).to_dict(orient='records')

    @staticmethod
    def load_txt_data(input_path):
        """
        Load data from a plain text file, where each line may be JSON or a formatted conversation line.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        data.append(json_data)
                    except json.JSONDecodeError:
                        conversations = DatasetConverter.process_plaintext_line(line)
                        if conversations:
                            data.append({"conversations": conversations})
                        else:
                            data.append({"raw_text": line})
        return data

    @staticmethod
    def load_csv_data(input_path):
        """
        Load data from a CSV file with columns 'system' and 'completion'.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('system') or row.get('completion'):
                    data.append({'system': row.get('system', ''), 'completion': row.get('completion', '')})
        return data

    @staticmethod
    def load_sql_data(input_path):
        """
        Load data from a SQL file by executing its script in memory.
        """
        data = []
        try:
            with sqlite3.connect(":memory:") as conn:
                cursor = conn.cursor()
                with open(input_path, 'r', encoding='utf-8') as f:
                    cursor.executescript(f.read())
                cursor.execute("CREATE TABLE IF NOT EXISTS data (system TEXT, completion TEXT)")
                data = cursor.execute("SELECT * FROM data").fetchall()
                column_names = [description[0] for description in cursor.description]
                data = [dict(zip(column_names, row)) for row in data]
        except sqlite3.Error as e:
            print(f"SQL Error: {e}")
        return data

    @staticmethod
    def process_data(data, output_path):
        """
        Process data and write conversations to an output file.
        """
        preview_entries = []
        conversations_found = False  # Flag to check if any conversations are found

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry)
                if conversations:
                    f.write(json.dumps({"conversations": conversations}) + '\n')
                    conversations_found = True  # Set the flag to True if conversations are found
                    if len(preview_entries) < 3:
                        preview_entries.append({"conversations": conversations})

        # Update status based on whether conversations were found
        status_message = "Conversations completed successfully." if conversations_found else "No conversations found for this dataset."
        print(status_message)  # Replace with your actual status bar update code

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
        conversations.append({"from": role, "value": message.get('content', '')})
