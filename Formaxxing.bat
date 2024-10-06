@echo off

REM Change to the directory where the script is located
cd /d "%~dp0"

REM Find and activate the virtual environment
IF EXIST "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) ELSE (
    echo Virtual environment not found.
    exit /b 1
)

REM Start the Python script
start python main.py