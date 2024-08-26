import json
import pandas as pd
import sqlite3
import csv
import os
import re
from fuzzywuzzy import fuzz

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
        elif ext == '.alpaca':
            return DatasetConverter.load_alpaca_data(input_path)
        elif ext == '.vicuna':
            return DatasetConverter.load_vicuna_data(input_path)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def load_json_data(input_path):
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
                                data.append(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Attempting with different encoding.")
            with open(input_path, 'r', encoding='utf-16') as f:
                file_content = f.read()
                # Retry parsing
                # Same parsing logic as above
        return data

    @staticmethod
    def load_jsonl_data(input_path):
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
                            data.append(DatasetConverter.fallback_parse_line(line))
        except UnicodeDecodeError:
            print("Unicode Decode Error. Attempting with different encoding.")
            with open(input_path, 'r', encoding='utf-16') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line: {line}")
                            data.append(DatasetConverter.fallback_parse_line(line))
        return data

    @staticmethod
    def load_parquet_data(input_path):
        """
        Load data from a Parquet file.
        """
        try:
            return pd.read_parquet(input_path).to_dict(orient='records')
        except Exception as e:
            print(f"Failed to load Parquet data: {e}")
            # Try pyarrow if pandas fails
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(input_path)
                return table.to_pandas().to_dict(orient='records')
            except Exception as e:
                print(f"Failed to load Parquet data with pyarrow: {e}")
                return []

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
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('system') or row.get('completion'):
                        data.append({'system': row.get('system', ''), 'completion': row.get('completion', '')})
        except csv.Error as e:
            print(f"CSV Error: {e}")
            # Try with different delimiters
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        if row.get('system') or row.get('completion'):
                            data.append({'system': row.get('system', ''), 'completion': row.get('completion', '')})
            except csv.Error as e:
                print(f"CSV Error with alternative delimiter: {e}")
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
            # Fallback to exporting SQL data manually or checking SQL dialect
        return data

    @staticmethod
    def load_alpaca_data(input_path):
        """
        Load data from an Alpaca dataset file.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        if 'instruction' in json_data and 'response' in json_data:
                            data.append({
                                'system': 'system', 
                                'completion': [
                                    {'role': 'user', 'content': json_data['instruction']},
                                    {'role': 'assistant', 'content': json_data['response']}
                                ]
                            })
                    except json.JSONDecodeError:
                        print(f"Skipping invalid Alpaca line: {line}")
        return data

    @staticmethod
    def load_vicuna_data(input_path):
        """
        Load data from a Vicuna dataset file.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        json_data = json.loads(line)
                        if 'prompt' in json_data and 'completion' in json_data:
                            data.append({
                                'system': 'system',
                                'completion': [
                                    {'role': 'user', 'content': json_data['prompt']},
                                    {'role': 'assistant', 'content': json_data['completion']}
                                ]
                            })
                    except json.JSONDecodeError:
                        print(f"Skipping invalid Vicuna line: {line}")
        return data

    @staticmethod
    def process_data(data, output_path):
        """
        Process data and write conversations to an output file.
        """
        preview_entries = []
        conversations_found = False

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry)
                if conversations:
                    f.write(json.dumps({"conversations": conversations}) + '\n')
                    conversations_found = True
                    if len(preview_entries) < 3:
                        preview_entries.append({"conversations": conversations})

        status_message = "Conversations completed successfully." if conversations_found else "No conversations found for this dataset."
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
