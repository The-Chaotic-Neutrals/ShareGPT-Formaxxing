@echo off
cd /d "%~dp0"
chcp 65001 >nul

:: Enable Virtual Terminal Processing for ANSI colors (Windows 10+)
for /f "tokens=2 delims=: " %%A in ('reg query "HKCU\Console" /v VirtualTerminalLevel 2^>nul') do (
  if %%A neq 1 (
    reg add "HKCU\Console" /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul
  )
)

:: Escape sequences and colors
set "ESC="
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "BLUE_BG=%ESC%[44m"
set "WHITE_FG=%ESC%[97m"
set "YELLOW_FG=%ESC%[93m"
set "GREEN_FG=%ESC%[92m"
set "CYAN_FG=%ESC%[96m"
set "RED_FG=%ESC%[91m"
set "MAGENTA_FG=%ESC%[95m"
set "BLACK_BG=%ESC%[40m"

set "TOP_BORDER=╔════════════════════════════════════════════════════════╗"
set "BOTTOM_BORDER=╚════════════════════════════════════════════════════════╝"
set "EMPTY_LINE=║                                                        ║"

:MENU
cls
echo %BLUE_BG%%WHITE_FG%%BOLD%%TOP_BORDER%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%EMPTY_LINE%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%║           ShareGPT Formaxxing Tool — Main Menu         ║%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%EMPTY_LINE%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%BOTTOM_BORDER%%RESET%
echo.

:: *** ESCAPE PARENTHESES AND AMPERSAND HERE ***
echo %YELLOW_FG%%BOLD%    1) Setup Environment ^(Delete ^& Reinstall^) %RESET%
echo %GREEN_FG%%BOLD%     2) Start Program with Updates (Upgrade)%RESET%
echo %CYAN_FG%%BOLD%      3) Start Program without Updates (Run)%RESET%
echo %RED_FG%%BOLD%       4) Exit%RESET%
echo.

<nul set /p= %WHITE_FG%%BOLD%Choose an option [1-4]: %RESET%
set /p choice=

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto RUN_UPDATE
if "%choice%"=="3" goto RUN_NO_UPDATE
if "%choice%"=="4" goto END

echo.
echo %RED_FG%%BOLD%⚠ Invalid choice! Try again, darling. ⚠%RESET%
echo %ESC%[5m%MAGENTA_FG%* That’s a no-go *%RESET%
echo.
echo %ESC%[7mPress any key to sashay back to menu...%RESET%
pause >nul
goto MENU

:SETUP
cls
echo %GREEN_FG%Setting up the virtual environment...%RESET%

if exist venv (
    echo Deleting existing virtual environment...
    rmdir /s /q venv
)

python -m venv venv

call venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

python -m spacy download en_core_web_sm

python -c "import spacy; spacy.load('en_core_web_sm')"

echo.
echo %GREEN_FG%Virtual environment is set up, and all requirements installed!%RESET%
pause
goto MENU

:RUN_UPDATE
cls
echo %CYAN_FG%Starting program with updates...%RESET%

call venv\Scripts\activate.bat

echo Upgrading Python packages from requirements.txt...
pip install --upgrade -r requirements.txt

echo Updating PyTorch packages (explicit versions)...
pip install --upgrade torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

echo Updating SpaCy model...
python -m spacy download en_core_web_sm

echo Starting the program...
start python main.py
pause
goto MENU

:RUN_NO_UPDATE
cls
echo %YELLOW_FG%Starting program without updates...%RESET%

call venv\Scripts\activate.bat

start python main.py
pause
goto MENU

:END
cls
echo %MAGENTA_FG%Thanks for stopping by! Bye bye!%RESET%
exit /b
