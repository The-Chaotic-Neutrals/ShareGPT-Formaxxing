import json
import pandas as pd
import sqlite3
import csv
import os

class DatasetConverter:
    @staticmethod
    def load_data(input_path):
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
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]

    @staticmethod
    def load_jsonl_data(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def load_parquet_data(input_path):
        return pd.read_parquet(input_path).to_dict(orient='records')

    @staticmethod
    def load_txt_data(input_path):
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        conversations = DatasetConverter.process_plaintext_line(line)
                        if conversations:
                            data.append({"conversations": conversations})
        return data

    @staticmethod
    def load_csv_data(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [{'system': row.get('system', ''), 'completion': row.get('completion', '')} for row in reader]

    @staticmethod
    def load_sql_data(input_path):
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
        except Exception:
            pass
        return data

    @staticmethod
    def process_data(data, output_path):
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
        if conversations_found:
            status_message = "Conversations completed successfully."
        else:
            status_message = "No conversations found for this dataset."

        print(status_message)  # Replace with your actual status bar update code

        return preview_entries

    @staticmethod
    def process_plaintext_line(line):
        conversations = []
        for role in ['system', 'user', 'assistant']:
            if f'{role}:' in line:
                value = line.split(f'{role}:', 1)[1].strip()
                conversations.append({"from": role if role != 'assistant' else 'gpt', "value": value})
        return conversations

    @staticmethod
    def extract_conversations(entry):
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
        role = message.get('role')
        if role == 'user':
            role = 'human'
        conversations.append({"from": role, "value": message.get('content', '')})
