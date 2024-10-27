:: Create the virtual environment 
python -m venv venv

:: Activate the virtual environment
CALL venv\Scripts\activate

:: Install the PyTorch package with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: Install other requirements from requirements.txt
pip install -r requirements.txt

:: Download the SpaCy model
python -m spacy download en_core_web_sm

:: Check if the model was installed correctly
python -c "import spacy; spacy.load('en_core_web_sm')"

:: Notify the user
echo Virtual environment is set up, and all requirements have been installed.
pause
