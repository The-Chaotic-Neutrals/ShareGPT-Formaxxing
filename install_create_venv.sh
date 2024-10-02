#!/bin/bash

# Exit immediately if any command fails
set -e

# Define the name of the virtual environment
VENV_NAME="venv"

# Create the virtual environment
python3 -m venv $VENV_NAME

# Check if the virtual environment was created successfully
if [ ! -f "$VENV_NAME/bin/activate" ]; then
  echo "Failed to create virtual environment. Exiting."
  exit 1
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Confirm that the venv is activated by checking the VIRTUAL_ENV variable
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Virtual environment activation failed. Exiting."
  exit 1
fi

# Install the PyTorch package with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other requirements from requirements.txt if the file exists
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Please add your dependencies to the file."
  exit 1
fi

# Download the SpaCy model
python -m spacy download en_core_web_sm

# Check if the SpaCy model was installed correctly
python -c "import spacy; spacy.load('en_core_web_sm')" || {
  echo "Failed to verify the SpaCy model. Exiting."
  exit 1
}

# Notify the user
echo "Virtual environment is set up, and requirements have been installed."

# Deactivate the virtual environment
deactivate

# Pause for the user to see the message
read -p "Press any key to exit..."
