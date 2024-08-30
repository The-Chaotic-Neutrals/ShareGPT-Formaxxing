@echo off
SETLOCAL

:: Define the name of the virtual environment
SET VENV_NAME=venv

:: Create the virtual environment
python -m venv %VENV_NAME%

:: Activate the virtual environment
CALL %VENV_NAME%\Scripts\activate

:: Install the requirements from requirements.txt
pip install -r requirements.txt

:: Download the SpaCy model
python -m spacy download en_core_web_sm

:: Check if the model was installed correctly
python -c "import spacy; spacy.load('en_core_web_sm')"

:: Notify the user
echo Virtual environment is set up and requirements have been installed.

:: Deactivate the virtual environment
CALL deactivate

ENDLOCAL
pause