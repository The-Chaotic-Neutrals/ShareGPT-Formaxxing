@echo off
SETLOCAL

:: Define the name of the virtual environment
SET VENV_NAME=venv

:: Create the virtual environment
python -m venv %VENV_NAME%

:: Check if the virtual environment was created successfully
IF NOT EXIST "%VENV_NAME%\Scripts\activate.bat" (
    echo Failed to create virtual environment. Exiting.
    exit /b 1
)

:: Activate the virtual environment
CALL %VENV_NAME%\Scripts\activate

:: Confirm that the venv is activated by checking the prompt
IF NOT DEFINED VIRTUAL_ENV (
    echo Virtual environment activation failed. Exiting.
    exit /b 1
)

:: Install the PyTorch package with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 || exit /b 1

:: Install other requirements from requirements.txt
IF EXIST requirements.txt (
    pip install -r requirements.txt || exit /b 1
) ELSE (
    echo requirements.txt not found. Please add your dependencies to the file.
    exit /b 1
)

:: Download the SpaCy model
python -m spacy download en_core_web_sm || exit /b 1

:: Check if the model was installed correctly
python -c "import spacy; spacy.load('en_core_web_sm')" || (
    echo Failed to verify the SpaCy model. Exiting.
    exit /b 1
)

:: Download the flash-attn wheel from GitHub
echo Downloading flash-attn wheel...
curl -L -o flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl ^
https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl || (
    echo Failed to download flash-attn wheel. Exiting.
    exit /b 1
)

:: Install the flash-attn wheel
echo Installing flash-attn wheel...
pip install flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl || (
    echo Failed to install flash-attn wheel. Exiting.
    exit /b 1
)

:: Clean up downloaded wheel file
echo Cleaning up...
del flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp311-cp311-win_amd64.whl

:: Notify the user
echo Virtual environment is set up, and all requirements have been installed.

:: Deactivate the virtual environment
deactivate

ENDLOCAL
pause
