import json
import yaml
import polars as pl
from pathlib import Path

def filter_dataset_by_yaml(input_path, yaml_path, output_dir):
    try:
        # Read the YAML file for strings to exclude
        with open(yaml_path, 'r') as yaml_file:
            exclude_strings = yaml.safe_load(yaml_file)

        # Read the JSON Lines file
        with open(input_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # Define a function to check if a conversation contains any excluded strings
        def contains_excluded_strings(conversation):
            text = ' '.join(msg.get('text', '') for msg in conversation)
            return any(exclude_string in text for exclude_string in exclude_strings)

        # Convert the data to a Polars DataFrame
        df = pl.DataFrame(data)

        # Create a boolean column based on the presence of excluded strings
        df = df.with_columns(
            pl.col('conversations').map_elements(contains_excluded_strings, return_dtype=pl.Boolean).alias('contains_excluded_strings')
        )

        # Filter the data based on the new boolean column
        filtered_data = df.filter(~pl.col('contains_excluded_strings'))

        # Create the Filtered directory if it doesn't exist
        filtered_dir = Path(output_dir) / "Filtered"
        filtered_dir.mkdir(exist_ok=True)

        # Save the filtered data as JSONL
        output_file = filtered_dir / f"{Path(input_path).stem}_filtered.jsonl"

        with output_file.open('w') as f:
            for row in filtered_data.drop('contains_excluded_strings').to_dicts():
                json.dump(row, f)
                f.write('\n')

        return f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(filtered_data)}"

    except Exception as e:
        raise ValueError(f"Error during filtering: {str(e)}")
