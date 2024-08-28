import json
import polars as pl
from pathlib import Path

def filter_dataset(input_path, output_dir):
    try:
        # Read the JSON Lines file
        with open(input_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # Define a function to check if a conversation contains all required roles
        def has_required_roles(conversation):
            roles = set(msg['from'] for msg in conversation)
            return 'system' in roles and 'human' in roles and 'gpt' in roles

        # Convert the data to a Polars DataFrame
        df = pl.DataFrame(data)

        # Create a boolean column based on the presence of required roles
        df = df.with_columns(
            pl.col('conversations').map_elements(has_required_roles, return_dtype=pl.Boolean).alias('has_required_roles')
        )

        # Filter the data based on the new boolean column
        filtered_data = df.filter(pl.col('has_required_roles'))

        # Create the Filtered directory if it doesn't exist
        filtered_dir = Path(output_dir) / "Filtered"
        filtered_dir.mkdir(exist_ok=True)

        # Save the filtered data as JSONL
        output_file = filtered_dir / f"{Path(input_path).stem}_filtered.jsonl"

        with output_file.open('w') as f:
            for row in filtered_data.drop('has_required_roles').to_dicts():
                json.dump(row, f)
                f.write('\n')

        return f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(filtered_data)}"

    except Exception as e:
        raise ValueError(f"Error during filtering: {str(e)}")
