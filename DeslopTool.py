import json
import os
from pathlib import Path
import yaml
import polars as pl

def filter_dataset(input_path, output_dir, filter_files):
    try:
        # Read the JSON Lines file
        with open(input_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # Combine all filter criteria from selected YAML/JSON/TXT files
        combined_filter_criteria = []
        for filter_file in filter_files:
            with open(filter_file, 'r') as f:
                if filter_file.endswith(('.yaml', '.yml')):
                    criteria = yaml.safe_load(f)
                elif filter_file.endswith('.json'):
                    criteria = json.load(f)
                elif filter_file.endswith('.txt'):
                    criteria = [json.loads(line.strip()) for line in f if line.strip()]
                else:
                    continue
                combined_filter_criteria.extend(criteria)

        # Define a function to check if a conversation matches any filter criteria
        def matches_criteria(conversation):
            if isinstance(conversation, str):
                try:
                    conversation = json.loads(conversation)
                except json.JSONDecodeError:
                    return False
            if isinstance(conversation, list):
                return any(
                    all(
                        key in msg and msg[key] == value
                        for key, value in criteria.items()
                    )
                    for msg in conversation
                    for criteria in combined_filter_criteria
                )
            return False

        # Convert the data to a Polars DataFrame
        df = pl.DataFrame(data)

        # Create a boolean column based on the filter criteria
        df = df.with_columns(
            pl.col('conversations').map_elements(matches_criteria, return_dtype=pl.Boolean).alias('matches_criteria')
        )

        # Filter the data based on the new boolean column (keep non-matching entries)
        filtered_data = df.filter(~pl.col('matches_criteria'))  # Invert the condition to keep non-matching entries

        # Create the deslop directory if it doesn't exist
        deslop_dir = Path(output_dir) / "deslop"
        deslop_dir.mkdir(exist_ok=True)

        # Save the filtered data as JSONL
        output_file = deslop_dir / f"{Path(input_path).stem}_deslop.jsonl"

        with output_file.open('w') as f:
            for row in filtered_data.drop('matches_criteria').to_dicts():
                json.dump(row, f)
                f.write('\n')

        return f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(filtered_data)}"

    except Exception as e:
        raise ValueError(f"Error during filtering: {str(e)}")
