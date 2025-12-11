# ShareGPT Formaxxing Tool

A comprehensive PyQt5-based GUI toolkit for processing, filtering, converting, and analyzing ShareGPT conversation datasets. This tool provides a unified interface for various data manipulation tasks commonly needed when working with training datasets.

## 🎯 Overview

ShareGPT Formaxxing Tool is a modular application designed to help researchers and developers process conversation datasets efficiently. It features a modern dark-themed GUI with animated backgrounds and provides access to multiple specialized tools organized into two categories: **Maxxer Tools** and **Mancer Tools**.

## ✨ Features

### 🛠 Maxxer Tools

- **DataMaxxer** - Filter and clean conversation datasets with configurable criteria
- **WordCloudMaxxer** - Generate word clouds from conversation data
- **SafetensorMaxxer** - Convert, Verify and fix safetensor model index files
- **ParquetMaxxer** - Convert between JSONL and Parquet formats
- **EnglishMaxxer** - Filter conversations by language (English detection)
- **TokenMaxxer** - Analyze token counts, clean datasets by token limits, and tokenize files
- **ForMaxxer** - Convert various JSON formats to standardized JSONL format
- **GrammarMaxxer** - Correct grammar and spelling in conversations using LanguageTool
- **SynthMaxxer** - Generate synthetic conversation data using LLM APIs

### ⚒️ Mancer Tools

- **DeslopMancer** - Filter conversations based on custom criteria files
- **RefusalMancer** - Classification models to detect and filter refusal responses
- **DedupMancer** - Deduplicate conversations using SHA-256 or MinHash algorithms
- **LineMancer** - Split, merge, and shuffle JSONL files
- **N-GraMancer** - Analyze n-gram patterns in conversation datasets

## 📋 Requirements

- Python 3.11+
- CUDA-capable GPU (for PyTorch operations)
- Windows 10+ or Linux
- ~10GB disk space for dependencies

## 🚀 Installation

### Windows

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ShareGPT-Formaxxing
   ```

2. Run the setup script:
   ```bash
   Formaxxing.bat
   ```

3. Select option `1` to set up the environment (creates virtual environment and installs dependencies)

4. Select option `3` to start the program

### Linux/Mac

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ShareGPT-Formaxxing
   ```

2. Make the script executable and run:
   ```bash
   chmod +x Formaxxing.sh
   ./Formaxxing.sh
   ```

3. Follow the menu prompts to set up and run

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r App/Assets/requirements.txt

# Install PyTorch with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install SpaCy model
python -m spacy download en_core_web_sm

# Install fastText (Windows - use wheel from Assets folder)
# Linux/Mac:
pip install fasttext
```

## 🎮 Usage

### Starting the Application

**Windows:**
```bash
Formaxxing.bat
# Select option 3: Start Program without Updates
```

**Linux/Mac:**
```bash
./Formaxxing.sh
# Select option 3: Start Program without Updates
```

**Direct Python:**
```bash
python -m App.Other.UI_Manager
```

### Main Interface

The application opens with a tabbed interface:

- **🛠 Maxxer Tools Tab**: Contains data processing and conversion tools
- **⚒️ Mancer Tools Tab**: Contains analysis and filtering tools

Click any tool button to open its dedicated window.

## 📖 Tool Documentation

### DataMaxxer
Filters conversation datasets based on multiple criteria:
- Remove blank turns
- Remove invalid endings
- Remove null GPT responses
- Remove duplicate system messages
- Remove duplicate human→GPT turns
- Allow/deny empty system roles

**Output**: Filtered dataset saved to `Outputs/datamaxxer/`

### DedupMancer
Removes duplicate conversations using two methods:
- **SHA-256**: Exact duplicate detection via content hashing
- **MinHash**: Similarity-based deduplication with configurable thresholds

**Features**:
- Cross-file deduplication support
- Semantic similarity detection
- Progress tracking

**Output**: Deduplicated dataset saved to `Outputs/`

### TokenMaxxer
Analyzes and processes datasets by token count:
- **Analyze**: Get token count statistics (percentiles, min/max, longest entry)
- **Clean**: Remove entries exceeding token limit
- **Tokenize**: Generate token count reports

**Features**:
- Supports any HuggingFace tokenizer model
- Remembers recently used models
- Detailed statistics and previews

**Output**: Processed files saved to `Outputs/`

### ForMaxxer
Converts various JSON formats to standardized JSONL:
- Handles JSON arrays
- Handles JSONL files
- Extracts conversations from nested structures
- Validates output format

**Output**: Converted files saved to `Outputs/formaxxer/`

### ParquetMaxxer
Converts between JSONL and Parquet formats:
- **JSONL → Parquet**: Efficient binary format conversion
- **Parquet → JSONL**: Convert back to text format
- Chunked processing for large files
- Preview of first rows

**Output**: Converted files saved to `Outputs/`

### EnglishMaxxer
Filters conversations by language:
- Uses FastText language detection model
- Configurable confidence threshold
- Multi-processing support for large files

**Output**: English-only dataset saved to `Outputs/`

### GrammarMaxxer
Corrects grammar and spelling in conversations:
- Uses LanguageTool for grammar checking
- Processes GPT responses only
- Live correction tracking

**Output**: Corrected dataset saved to `corrected/`

### SynthMaxxer
Generates synthetic conversation data:
- Supports multiple LLM APIs (Anthropic, OpenAI, etc.)
- Configurable system prompts and message templates
- Automatic retry on refusals
- Batch processing capabilities

**Output**: Generated conversations saved to `Datasets/Raw/`

### RefusalMancer
Classification tool for detecting refusals:
- Two modes: RP (roleplay) and Normal
- Uses transformer-based classification models
- Configurable confidence thresholds
- Batch processing support

**Output**: Filtered dataset saved to `Outputs/`

### DeslopMancer
Filters conversations based on custom criteria:
- Loads filter phrases from text files
- Configurable threshold multipliers
- Detailed matching statistics
- YAML-based filtering support

**Output**: Filtered dataset saved to `Outputs/`

### LineMancer
Manipulates JSONL files:
- **Split**: Divide large files into smaller chunks
- **Merge**: Combine multiple JSONL files
- **Shuffle**: Randomize line order

**Output**: Processed files saved to `Outputs/linemancer/`

### N-GraMancer
Analyzes n-gram patterns:
- Configurable n-gram size
- Frequency analysis
- Pattern visualization

**Output**: Analysis reports saved to `Outputs/`

### WordCloudMaxxer
Generates word clouds from conversation data:
- Customizable appearance
- Frequency-based word sizing
- Export to image formats

**Output**: Word cloud images saved to `Outputs/`

### SafetensorMaxxer
Verifies and repairs safetensor model files:
- Index file validation
- Automatic repair capabilities
- Batch processing

**Output**: Verified/repaired files in place

## 📁 Project Structure

```
ShareGPT-Formaxxing/
├── App/
│   ├── Assets/              # Icons, models, requirements
│   ├── DataMaxxer/          # Dataset filtering tool
│   ├── DedupeMancer/        # Deduplication tool
│   ├── DeslopMancer/        # Criteria-based filtering
│   ├── EnglishMaxxer/       # Language filtering
│   ├── ForMaxxer/           # Format conversion
│   ├── GrammarMaxxer/       # Grammar correction
│   ├── LineMancer/          # File manipulation
│   ├── N-GraMancer/         # N-gram analysis
│   ├── Other/               # UI components, theme, manager
│   ├── ParquetMaxxer/       # Parquet conversion
│   ├── RefusalMancer/       # Refusal detection
│   ├── SafetensorMaxxer/    # Safetensor tools
│   ├── SynthMaxxer/         # Synthetic data generation
│   ├── TokenMaxxer/         # Token analysis
│   └── WordCloudMaxxer/     # Word cloud generation
├── Datasets/                # Input datasets directory
├── Outputs/                 # Output directory for all tools
├── Formaxxing.bat           # Windows launcher
├── Formaxxing.sh            # Linux/Mac launcher
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables

- `TRANSFORMERS_USE_FLASH_ATTENTION=1` - Enables flash attention for transformers (set automatically by launcher scripts)

### Tool-Specific Configs

- **TokenMaxxer**: `App/TokenMaxxer/tokenmaxxer_config.json` - Stores recent models
- **SynthMaxxer**: `App/SynthMaxxer/synthmaxxer_config.json` - API and generation settings
- **DeslopMancer**: `App/DeslopMancer/filter_criteria.txt` - Custom filter phrases

## 📝 Output Locations

All tools save their output to the `Outputs/` directory in the repository root, organized by tool name:

- `Outputs/datamaxxer/` - DataMaxxer filtered datasets
- `Outputs/formaxxer/` - ForMaxxer converted files
- `Outputs/linemancer/` - LineMancer processed files
- `Outputs/` - General output for other tools

## 🐛 Troubleshooting

### CUDA Not Detected
- Ensure you have a CUDA-capable GPU
- Install CUDA 12.4 compatible drivers
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- Ensure virtual environment is activated
- Run: `pip install -r App/Assets/requirements.txt`
- Check that all dependencies are installed

### Memory Issues
- Use chunked processing options where available
- Reduce batch sizes in tool settings
- Close other applications to free memory

### FastText Model Issues (Windows)
- The wheel file is included in `App/Assets/`
- If download fails, manually install: `pip install App/Assets/fasttext-0.9.2-cp311-cp311-win_amd64.whl`


## 🙏 Acknowledgments

- Built with PyQt5
- Uses transformers, sentence-transformers, and other excellent libraries
- FastText for language detection
- LanguageTool for grammar checking

---

**Note**: This tool is designed for processing ShareGPT-format conversation datasets. Ensure your input files follow the expected format with `conversations` arrays containing `from` and `value` fields.
