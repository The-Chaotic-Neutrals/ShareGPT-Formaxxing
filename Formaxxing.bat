@echo off
setlocal EnableExtensions
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

set "TOP_BORDER=╔════════════════════════════════════════════════════════╗"
set "BOTTOM_BORDER=╚════════════════════════════════════════════════════════╝"
set "EMPTY_LINE=║                                                        ║"

:: Always use the repo's requirements files (do NOT generate in main dir)
set "REQ_FILE=%~dp0App\Assets\requirements.txt"
set "EXTRA_REQ_FILE=%~dp0App\Assets\extra_requirements.txt"

:MENU
cls
echo %BLUE_BG%%WHITE_FG%%BOLD%%TOP_BORDER%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%EMPTY_LINE%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%║           ShareGPT Formaxxing Tool — Main Menu         ║%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%EMPTY_LINE%%RESET%
echo %BLUE_BG%%WHITE_FG%%BOLD%%BOTTOM_BORDER%%RESET%
echo.

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
pause >nul
goto MENU

:CHECK_REQ
if not exist "%REQ_FILE%" (
    echo.
    echo %RED_FG%%BOLD%ERROR: Requirements file not found:%RESET%
    echo %RED_FG%  "%REQ_FILE%"%RESET%
    echo.
    pause
    goto MENU
)
exit /b 0

:SETUP
cls
echo %GREEN_FG%Setting up the virtual environment...%RESET%

call :CHECK_REQ

if exist venv (
    echo Deleting existing virtual environment...
    rmdir /s /q venv
)

python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip --no-cache-dir

:: Force NumPy 2.0.x
python -m pip install "numpy>=2.0.0,<2.1.0"

:: Install base dependencies from the correct requirements file
python -m pip install -r "%REQ_FILE%"

:: Install extra dependencies if the repo has them
if exist "%EXTRA_REQ_FILE%" (
    python -m pip install -r "%EXTRA_REQ_FILE%"
)

:: Install your pinned GPU-only PyTorch stack
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not detected!'"

:: Install SpaCy model
python -m spacy download en_core_web_sm

:: Install fastText wheel
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/mdrehan4all/fasttext_wheels_for_windows/raw/main/fasttext-0.9.2-cp311-cp311-win_amd64.whl','fasttext-0.9.2-cp311-cp311-win_amd64.whl')"
python -m pip install fasttext-0.9.2-cp311-cp311-win_amd64.whl

echo %GREEN_FG%Environment ready with NumPy 2.0.x and GPU-only PyTorch.%RESET%
pause
goto MENU

:RUN_UPDATE
cls
echo %CYAN_FG%Starting program with updates...%RESET%
call venv\Scripts\activate.bat

call :CHECK_REQ

python -m pip install --upgrade pip --no-cache-dir

:: Force NumPy 2.0.x
python -m pip install "numpy>=2.0.0,<2.1.0"

:: Upgrade base dependencies from the correct requirements file
python -m pip install --upgrade -r "%REQ_FILE%"

:: Upgrade extra dependencies if the repo has them
if exist "%EXTRA_REQ_FILE%" (
    python -m pip install --upgrade -r "%EXTRA_REQ_FILE%"
)

:: Upgrade your pinned GPU-only PyTorch stack
python -m pip install --upgrade torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not detected!'"

:: Update SpaCy model
python -m spacy download en_core_web_sm

:: Install/upgrade fastText wheel
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/mdrehan4all/fasttext_wheels_for_windows/raw/main/fasttext-0.9.2-cp311-cp311-win_amd64.whl','fasttext-0.9.2-cp311-cp311-win_amd64.whl')"
python -m pip install --upgrade fasttext-0.9.2-cp311-cp311-win_amd64.whl

set TRANSFORMERS_USE_FLASH_ATTENTION=1
python -m App.Other.UI_Manager
pause >nul
goto MENU

:RUN_NO_UPDATE
cls
echo %YELLOW_FG%Starting program without updates...%RESET%
call venv\Scripts\activate.bat
set TRANSFORMERS_USE_FLASH_ATTENTION=1
python -m App.Other.UI_Manager
pause >nul
goto MENU

:END
cls
echo %MAGENTA_FG%Thanks for stopping by! Bye bye!%RESET%
exit /b
