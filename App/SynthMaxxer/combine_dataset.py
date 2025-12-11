import os
import json
import pathlib

from datetime import datetime
from App.SynthMaxxer.config import ACTIVE_CONFIG as config

# Specify the raw directory and output file

# Get the directory of the current script
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.absolute()
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
raw_directory = f'{SCRIPT_DIR}/Datasets/Raw/{config["DIRECTORY_NAME"]}'
output_file = str(OUTPUTS_DIR / f'{config["DIRECTORY_NAME"]}_{timestamp}.jsonl')


def combine_json_files(raw_directory, output_file):
    combined_data = []

    # Get a list of JSONL files in the raw directory
    jsonl_files = [file for file in os.listdir(raw_directory) if file.endswith('.jsonl')]

    # Iterate over each JSONL file
    for jsonl_file in jsonl_files:
        file_path = os.path.join(raw_directory, jsonl_file)

        # Read the JSONL file and append its contents to the combined data
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                data = json.loads(content)
                combined_data.append(data)
            except json.JSONDecodeError:
                print(f"Skipping file {jsonl_file} due to JSON decoding error")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the combined data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"Combined JSON files saved to {output_file}")


# Call the function to combine JSON files
combine_json_files(raw_directory, output_file)
