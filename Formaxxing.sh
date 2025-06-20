#!/bin/bash

cd "$(dirname "$0")"

# ANSI Colors and Styles
ESC=$(printf '\033')
RESET="${ESC}[0m"
BOLD="${ESC}[1m"
BLUE_BG="${ESC}[44m"
WHITE_FG="${ESC}[97m"
YELLOW_FG="${ESC}[93m"
GREEN_FG="${ESC}[92m"
CYAN_FG="${ESC}[96m"
RED_FG="${ESC}[91m"
MAGENTA_FG="${ESC}[95m}"
BLACK_BG="${ESC}[40m"

TOP_BORDER="╔════════════════════════════════════════════════════════╗"
BOTTOM_BORDER="╚════════════════════════════════════════════════════════╝"
EMPTY_LINE="║                                                        ║"

function pause() {
    echo
    read -rp "$(echo -e "${BOLD}${WHITE_FG}Press Enter to return to the menu...${RESET}")"
}

function show_menu() {
    clear
    echo -e "${BLUE_BG}${WHITE_FG}${BOLD}${TOP_BORDER}${RESET}"
    echo -e "${BLUE_BG}${WHITE_FG}${BOLD}${EMPTY_LINE}${RESET}"
    echo -e "${BLUE_BG}${WHITE_FG}${BOLD}║           ShareGPT Formaxxing Tool — Main Menu         ║${RESET}"
    echo -e "${BLUE_BG}${WHITE_FG}${BOLD}${EMPTY_LINE}${RESET}"
    echo -e "${BLUE_BG}${WHITE_FG}${BOLD}${BOTTOM_BORDER}${RESET}"
    echo
    echo -e "${YELLOW_FG}${BOLD}    1) Setup Environment (Delete & Reinstall)${RESET}"
    echo -e "${GREEN_FG}${BOLD}     2) Start Program with Updates (Upgrade)${RESET}"
    echo -e "${CYAN_FG}${BOLD}      3) Start Program without Updates (Run)${RESET}"
    echo -e "${RED_FG}${BOLD}       4) Exit${RESET}"
    echo
    read -rp "$(echo -e "${WHITE_FG}${BOLD}Choose an option [1-4]: ${RESET}")" choice

    case "$choice" in
        1) setup_environment ;;
        2) run_with_updates ;;
        3) run_without_updates ;;
        4) end_script ;;
        *) 
            echo -e "${RED_FG}${BOLD}⚠ Invalid choice! Try again, darling. ⚠${RESET}"
            echo -e "${ESC}[5m${MAGENTA_FG}* That’s a no-go *${RESET}"
            pause
            show_menu
            ;;
    esac
}

function setup_environment() {
    clear
    echo -e "${GREEN_FG}Setting up the virtual environment...${RESET}"
    
    if [ -d "venv" ]; then
        echo "Deleting existing virtual environment..."
        rm -rf venv
    fi

    python3 -m venv venv
    source venv/bin/activate

    python -m pip install --upgrade pip
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

    python -m spacy download en_core_web_sm
    python -c "import spacy; spacy.load('en_core_web_sm')"

    echo
    echo -e "${GREEN_FG}Virtual environment is set up, and all requirements installed!${RESET}"
    pause
    show_menu
}

function run_with_updates() {
    clear
    echo -e "${CYAN_FG}Starting program with updates...${RESET}"

    source venv/bin/activate

    echo "Upgrading Python packages from requirements.txt..."
    pip install --upgrade -r requirements.txt

    echo "Updating PyTorch packages (explicit versions)..."
    pip install --upgrade torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

    echo "Updating SpaCy model..."
    python -m spacy download en_core_web_sm

    echo "Starting the program..."
    python main.py
    pause
    show_menu
}

function run_without_updates() {
    clear
    echo -e "${YELLOW_FG}Starting program without updates...${RESET}"

    source venv/bin/activate
    python main.py
    pause
    show_menu
}

function end_script() {
    clear
    echo -e "${MAGENTA_FG}Thanks for stopping by! Bye bye!${RESET}"
    exit 0
}

show_menu
