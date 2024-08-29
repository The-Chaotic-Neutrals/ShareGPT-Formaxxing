import json
import polars as pl
from pathlib import Path

def filter_dataset(input_path, output_dir):
    try:
        # Read the JSON Lines file
        with open(input_path, 'r') as file:
            data = [json.loads(line) for line in file]

        # Define required roles
        required_roles = {'human', 'gpt'}

        # Define a function to check if a conversation contains all required roles
        def has_required_roles(conversation):
            roles = {msg['from'] for msg in conversation}
            return required_roles.issubset(roles)

        # Convert the data to a Polars DataFrame
        df = pl.DataFrame(data)

        # Filter the data based on required roles
        df_filtered = df.filter(
            pl.col('conversations').apply(lambda conv: has_required_roles(conv))
        )

        # Create the Filtered directory if it doesn't exist
        filtered_dir = Path(output_dir) / "Filtered"
        filtered_dir.mkdir(exist_ok=True)

        # Save the filtered data as JSONL
        output_file = filtered_dir / f"{Path(input_path).stem}_filtered.jsonl"
        with output_file.open('w') as f:
            for row in df_filtered.to_dicts():
                json.dump(row, f)
                f.write('\n')

        return f"Filtered data saved to {output_file}\nOriginal size: {len(df)}\nFiltered size: {len(df_filtered)}"

    except Exception as e:
        raise ValueError(f"Error during filtering: {str(e)}")