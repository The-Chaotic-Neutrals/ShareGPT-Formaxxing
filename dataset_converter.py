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
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.jsonl':
            return [json.loads(line) for line in open(input_path, 'r', encoding='utf-8') if line.strip()]
        elif ext == '.parquet':
            return pd.read_parquet(input_path).to_dict(orient='records')
        elif ext == '.txt':
            return DatasetConverter.load_txt_data(input_path)
        elif ext == '.csv':
            return DatasetConverter.load_csv_data(input_path)
        elif ext == '.sql':
            return DatasetConverter.load_sql_data(input_path)
        else:
            raise ValueError("Unsupported file format")

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
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            with open(input_path, 'r', encoding='utf-8') as f:
                cursor.executescript(f.read())
            data = cursor.execute("SELECT * FROM data").fetchall()
            column_names = [description[0] for description in cursor.description]
            return [dict(zip(column_names, row)) for row in data]

    @staticmethod
    def process_data(data, output_path):
        preview_entries = []
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                conversations = DatasetConverter.extract_conversations(entry)
                if conversations:
                    f.write(json.dumps({"conversations": conversations}) + '\n')
                    if len(preview_entries) < 3:
                        preview_entries.append({"conversations": conversations})
        return preview_entries

    @staticmethod
    def process_plaintext_line(line):
        conversations = []
        for role in ['system', 'user', 'assistant']:
            if f'{role}:' in line:
                conversations.append({"from": role if role != 'assistant' else 'gpt', 
                                      "value": line.split(f'{role}:')[1].strip()})
        return conversations

    @staticmethod
    def extract_conversations(entry):
        conversations = []
        if 'conversations' in entry:
            for message in entry['conversations']:
                role = message.get('from')
                if role in ['system', 'user', 'gpt']:
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
            conversations.append({"from": "human", "value": message.get('content', '')})
        elif role == 'assistant':
            conversations.append({"from": "gpt", "value": message.get('content', '')})