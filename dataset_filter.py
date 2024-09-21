import json
from pathlib import Path

def filter_dataset(input_path, output_dir):
    try:
        # Read the JSON Lines file
        with open(input_path, 'r') as file:
            data = [json.loads(line) for line in file]

        print(f"Sample data: {data[:5]}")  # Inspect the input data

        # Function to filter conversations
        filtered_data = []
        
        for item in data:
            conversations = item.get("conversations", [])
            
            # Check if both roles are present
            roles = set(msg['from'] for msg in conversations)
            if 'human' in roles and 'gpt' in roles:
                # Check and remove the last human message if it's the last one
                if conversations and conversations[-1]['from'] == 'human':
                    conversations = conversations[:-1]  # Remove the last message
                
                print(f"Filtered conversations: {conversations}")  # Debugging output
                filtered_data.append({"conversations": conversations})

        # Create the Filtered directory if it doesn't exist
        filtered_dir = Path(output_dir) / "filtered"
        filtered_dir.mkdir(exist_ok=True)

        # Save the filtered data as JSONL
        output_file = filtered_dir / f"{Path(input_path).stem}_filtered.jsonl"

        with output_file.open('w') as f:
            for row in filtered_data:
                json.dump(row, f)
                f.write('\n')

        return f"Filtered data saved to {output_file}\nOriginal size: {len(data)}\nFiltered size: {len(filtered_data)}"

    except Exception as e:
        raise ValueError(f"Error during filtering: {str(e)}")
