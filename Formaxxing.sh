#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Find and activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
else
    echo "Virtual environment not found."
    exit 1
fi

# Start the Python script
python main.py
