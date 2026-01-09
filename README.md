# ShareGPT Formaxxing Tool

A comprehensive PyQt5-based GUI toolkit for processing, filtering, converting, and analyzing ShareGPT conversation datasets. This tool provides a unified interface for various data manipulation tasks commonly needed when working with training datasets.

## 🎯 Overview

ShareGPT Formaxxing Tool is a modular application designed to help researchers and developers process conversation datasets efficiently. It features a modern dark-themed GUI with animated backgrounds and provides access to multiple specialized tools organized into two categories: **Maxxer Tools** and **Mancer Tools**.

## ✨ Features

### 🛠 Maxxer Tools

- **ForMaxxer** - Convert various JSON formats to standardized JSONL format, with optional filtering/cleaning (formerly DataMaxxer)
- **LanguageMaxxer** - Filter for English conversations and/or correct grammar (combines EnglishMaxxer + GrammarMaxxer)
- **SafetensorMaxxer** - Convert, Verify and fix safetensor model index files
- **ParquetMaxxer** - Convert between JSONL and Parquet formats
- **TokenMaxxer** - Analyze token counts, clean datasets by token limits, and tokenize files
- **SynthMaxxer** - Generate synthetic conversation data using LLM APIs

### ⚒️ Mancer Tools

- **DeslopMancer** - Filter conversations based on custom criteria files
- **RefusalMancer** - Classification models to detect and filter refusal responses
- **DedupeMancer** - Unified deduplication for datasets (SHA-256/MinHash) and images (dHash/CLIP embeddings)
- **LineMancer** - Split, merge, and shuffle JSONL files
- **N-GraMancer** - Analyze n-gram patterns in conversation datasets

## 📋 Requirements

- Python 3.11+
- CUDA-capable GPU (for PyTorch operations)
- Windows 10+ or Linux/Mac
- ~10GB disk space for dependencies
- NumPy 2.0.x (automatically installed by setup scripts)

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
   - The script will automatically generate `requirements.txt` in the root directory using `pipreqs` if it doesn't exist
   - It will also install NumPy 2.0.x and PyTorch with CUDA support
   - fastText wheel will be automatically downloaded from [fasttext_wheels_for_windows](https://github.com/mdrehan4all/fasttext_wheels_for_windows) repository
   - If `extra_requirements.txt` exists, it will be installed as well

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

3. Select option `1` to set up the environment (creates virtual environment and installs dependencies)

4. Select option `3` to start the program

**Note**: The Linux/Mac script (`Formaxxing.sh`) has a simpler setup process compared to the Windows version. For full feature parity, you may need to manually install NumPy 2.0.x and ensure all dependencies are up to date.

**fastText Installation on Linux/Mac:**
- The script attempts `pip install fasttext`, which may work on some systems
- If installation fails, you'll need to compile fastText from source or find a precompiled wheel for your system
- See the [Troubleshooting](#-troubleshooting) section for detailed fastText installation instructions

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

# Upgrade pip
python -m pip install --upgrade pip

# Install NumPy 2.0.x (required)
pip install "numpy>=2.0.0,<2.1.0"

# Install dependencies from Assets folder
pip install -r App/Assets/requirements.txt

# Install extra dependencies if extra_requirements.txt exists
# (Optional, only if present in repository root)
if exist extra_requirements.txt pip install -r extra_requirements.txt

# Install PyTorch with CUDA support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU not detected!'"

# Install SpaCy model
python -m spacy download en_core_web_sm

# Install fastText
# Windows - download appropriate wheel from GitHub repository:
# Visit https://github.com/mdrehan4all/fasttext_wheels_for_windows
# Download the wheel matching your Python version (e.g., fasttext-0.9.2-cp311-cp311-win_amd64.whl for Python 3.11)
# Then install: pip install fasttext-0.9.2-cp311-cp311-win_amd64.whl
# Or use direct download:
# pip install https://github.com/mdrehan4all/fasttext_wheels_for_windows/raw/main/fasttext-0.9.2-cp311-cp311-win_amd64.whl
# Linux/Mac - compile from source or find precompiled wheels:
# Option 1: Try pip install fasttext (may work on some systems)
# Option 2: Compile from source: https://github.com/facebookresearch/fastText/tree/main/python
# Option 3: Search for precompiled wheels online for your specific Python version and architecture
```

**Note**: The launcher scripts (`Formaxxing.bat`/`Formaxxing.sh`) automatically generate a `requirements.txt` file in the repository root using `pipreqs` if it doesn't exist. This is used for dependency management during setup.

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

### DedupeMancer
Unified deduplication tool with tabs for datasets and images (formerly DedupMancer + ImageDedupMancer):

**Dataset Deduplication Tab:**
- **SHA-256/String-Match**: Exact duplicate detection via content hashing
- **MinHash + Semantic**: Similarity-based deduplication with configurable thresholds
- Cross-file deduplication support
- Progress tracking with speed display

**Image Deduplication Tab:**
- **Perceptual Hash (Fast)**: SHA-256 exact + dHash near-duplicate detection
- **Image Embeddings (Deep)**: CLIP-based semantic similarity detection
- **Text/Caption Dedup**: Deduplicate by caption similarity using sentence-transformers
  - Works with HuggingFace-style datasets (metadata.jsonl + images/ folder)
  - Configurable text field, filename field, and similarity threshold
- Drag-and-drop folder/image support
- Copy unique images to output, optionally move duplicates

**Output**: 
- Datasets: `Outputs/dedupemancer/datasets/`
- Images: `Outputs/dedupemancer/images/unique/` and `Outputs/dedupemancer/images/duplicates/`
- Text dedup: `Outputs/dedupemancer/text_dedup/unique/`

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
Converts various JSON formats to standardized JSONL with optional filtering:

**Format Conversion:**
- Auto-detects input format (ShareGPT, HuggingFace, Vicuna, Alpaca, ChatML)
- Handles JSON arrays and JSONL files
- Extracts conversations from nested structures
- Validates and normalizes output format

**Optional Filtering (formerly DataMaxxer):**
Enable post-conversion filtering to clean your dataset:
- Remove blank turns
- Remove invalid endings
- Remove null GPT responses
- Remove duplicate system messages
- Remove duplicate human→GPT turns
- Allow/deny empty system roles

**Output**: Converted files saved to `Outputs/formaxxer/`, filtered files saved to `Outputs/formaxxer/filtered/`

### ParquetMaxxer
Converts between JSONL and Parquet formats:
- **JSONL → Parquet**: Efficient binary format conversion
- **Parquet → JSONL**: Convert back to text format
- Chunked processing for large files
- Preview of first rows

**Output**: Converted files saved to `Outputs/`

### LanguageMaxxer
Combined English filtering and grammar correction tool (formerly EnglishMaxxer + GrammarMaxxer):

**English Filtering (FastText):**
- Uses FastText language detection model
- Configurable confidence threshold (default 0.69)
- Multi-processing support for large files
- Filters out non-English conversations

**Grammar Correction (LanguageTool):**
- Uses LanguageTool for grammar checking
- Processes GPT responses only
- Live correction tracking
- Can be applied after English filtering or standalone

**Workflow:**
1. Enable English filtering to keep only English conversations
2. Optionally enable grammar correction to fix grammar in GPT responses
3. Both operations can be run together or separately

**Output**: Files saved to `Outputs/languagemaxxer/english_filtered/` and `Outputs/languagemaxxer/grammar_corrected/`

### SynthMaxxer
Comprehensive synthetic data generation and processing tool with multiple modes and mechanisms:

**Generation Tab - Conversation Generation:**
- **API Providers Supported:**
  - Anthropic Claude (Messages API)
  - OpenAI Official (Chat Completions)
  - OpenAI Chat Completions (Custom endpoints)
  - OpenAI Text Completions (Legacy format)
  - Grok (xAI) - Uses xAI SDK
  - Gemini (Google) - Stream Generate Content API
  - OpenRouter - Multi-provider routing
  - DeepSeek - DeepSeek API

- **Generation Mechanisms:**
  - **Streaming Support**: Real-time token streaming for all supported APIs
  - **Refusal Detection**: Automatic detection and retry on refusal phrases
  - **Force Retry Phrases**: Configurable phrases that trigger mandatory retries
  - **Conversation Tags**: Custom start/end tags for user and assistant messages
  - **Instruct vs Chat Mode**: Toggle between instruction-following and conversational formats
  - **Temperature Control**: Configurable temperature (0.69-1.42 range) for generation diversity
  - **Top-p/Top-k Sampling**: Advanced sampling controls for response quality
  - **Delay Mechanisms**: Configurable min/max delays between API calls to respect rate limits
  - **Stop Percentage**: Probabilistic stopping after minimum turns (25% for instruct, 5% for chat)
  - **Minimum Turns**: Enforce minimum conversation length before stopping

**Processing Tab - Entry Enhancement:**
- **Improvement Mechanisms:**
  - **Improve**: Rewrite existing conversations for clarity, coherence, and style diversity
  - **Extend**: Add additional conversation turns to existing entries
  - **Improve & Extend**: Combined operation for simultaneous enhancement and extension
  - **Generate New Entries**: Create entirely new entries from scratch with optional example-based generation

- **Advanced Features:**
  - **Dynamic Names Mode**: Automatic character name generation and injection into conversations
  - **Names Cache System**: Persistent cache of generated names with categories for consistency
  - **Human Cache System**: Pre-generated human conversation turns for consistent character interactions
  - **Human Cache Tools**: 
    - Generate new human turns for the cache
    - Improve/rewrite existing human turns for better quality
  - **Reply in Character**: Ensures system messages are preserved and responses stay in character
  - **Schema Validation**: Automatic validation of ShareGPT format compliance
  - **Auto-Fixing**: Intelligent repair of schema violations (missing fields, incorrect structure)
  - **Batch Processing**: Configurable concurrency for parallel processing
  - **Line Range Processing**: Process specific line ranges from input files
  - **Resume Capability**: Automatic resume from last processed entry

- **Processing Worker Mechanisms:**
  - Uses xAI SDK (Grok) for processing operations
  - JSON repair for handling malformed model responses
  - Temperature randomization (0.69-1.42) for diversity
  - Entry filtering: Invalid entries are skipped automatically
  - Progress tracking with byte and entry counters

**Multimodal Tab - Image Captioning:**
- **Vision API Support:**
  - OpenAI Vision API
  - Anthropic Claude (Vision capabilities)
  - Grok (xAI) Vision API
  - OpenRouter (Vision models)

- **Captioning Mechanisms:**
  - **Batch Processing**: Process multiple images with configurable batch sizes
  - **Resume Support**: Automatically resume from last processed image
  - **Existing Caption Detection**: Skip images that already have valid captions
  - **Custom Prompts**: Configurable caption generation prompts
  - **Max Captions Limit**: Optional limit on number of images to process
  - **Parquet Output**: HuggingFace Dataset format with text and images columns
  - **Image Format Support**: JPEG, PNG, GIF, WebP, BMP

**Core Mechanisms:**
- **Worker Functions:**
  - `worker.py`: Handles generation loop with streaming and refusal detection
  - `processing_worker.py`: Manages JSONL file processing, improvement, and extension
  - `multimodal_worker.py`: Handles image captioning with multiple API support
  - `llm_helpers.py`: Core LLM interaction functions (improve, extend, generate)

- **Schema Management:**
  - Automatic ShareGPT format validation
  - Schema repair for common violations
  - Conversation structure enforcement (system → human → gpt alternation)
  - Final message validation (must end with GPT response)

- **Configuration System:**
  - Per-API-type configuration storage
  - Separate API keys for different providers
  - Model name caching and auto-refresh
  - Template-based configuration system

**Output Locations:**
- Generated conversations: `Datasets/Raw/{directory_name}/`
- Processed files: `Outputs/` (or custom location)
- Multimodal outputs: Parquet files in specified location
- Cache files: `App/SynthMaxxer/global_human_cache.json`, `global_names_cache.json`

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
│   ├── DedupeMancer/        # Dataset & image deduplication
│   ├── DeslopMancer/        # Criteria-based filtering
│   ├── ForMaxxer/           # Format conversion & filtering
│   ├── LanguageMaxxer/      # English filtering & grammar correction
│   ├── LineMancer/          # File manipulation
│   ├── N-GraMancer/         # N-gram analysis
│   ├── Other/               # UI components, theme, manager
│   ├── ParquetMaxxer/       # Parquet conversion
│   ├── RefusalMancer/       # Refusal detection
│   ├── SafetensorMaxxer/    # Safetensor tools
│   ├── SynthMaxxer/         # Synthetic data generation
│   └── TokenMaxxer/         # Token analysis
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

- **TokenMaxxer**: `App/TokenMaxxer/tokenmaxxer_config.json` - Stores recent tokenizer models
- **SynthMaxxer**: 
  - `App/SynthMaxxer/synthmaxxer_config.json` - API keys, endpoints, and generation settings
  - `App/SynthMaxxer/grok_tool_config.json` - xAI Grok-specific configuration
  - `App/SynthMaxxer/global_human_cache.json` - Cached human conversation turns
  - `App/SynthMaxxer/global_names_cache.json` - Cached character names for dynamic naming
  - `App/SynthMaxxer/config.py` - Template configuration for generation patterns
- **DeslopMancer**: `App/DeslopMancer/filter_criteria.txt` - Custom filter phrases

## 📝 Output Locations

All tools save their output to the `Outputs/` directory in the repository root, organized by tool name:

- `Outputs/formaxxer/` - ForMaxxer converted files
- `Outputs/formaxxer/filtered/` - ForMaxxer filtered datasets
- `Outputs/languagemaxxer/english_filtered/` - LanguageMaxxer English-filtered files
- `Outputs/languagemaxxer/grammar_corrected/` - LanguageMaxxer grammar-corrected files
- `Outputs/dedupemancer/datasets/` - DedupeMancer deduplicated JSONL files
- `Outputs/dedupemancer/images/unique/` - DedupeMancer unique images
- `Outputs/dedupemancer/images/duplicates/` - DedupeMancer duplicate images (if moved)
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
- If `requirements.txt` exists in root, the launcher uses that instead
- Check that all dependencies are installed
- Ensure NumPy 2.0.x is installed: `pip install "numpy>=2.0.0,<2.1.0"`

### Memory Issues
- Use chunked processing options where available
- Reduce batch sizes in tool settings
- Close other applications to free memory

### FastText Installation Issues

**Windows:**
- Download the appropriate wheel file from [fasttext_wheels_for_windows](https://github.com/mdrehan4all/fasttext_wheels_for_windows)
- Choose the wheel matching your Python version (e.g., `fasttext-0.9.2-cp311-cp311-win_amd64.whl` for Python 3.11 on 64-bit Windows)
- Install directly: `pip install https://github.com/mdrehan4all/fasttext_wheels_for_windows/raw/main/fasttext-0.9.2-cp311-cp311-win_amd64.whl`
- Or download the wheel file first, then: `pip install fasttext-0.9.2-cp311-cp311-win_amd64.whl`

**Linux/Mac:**
- Try `pip install fasttext` first (may work on some systems)
- If that fails, compile from source: Follow instructions at [fastText Python bindings](https://github.com/facebookresearch/fastText/tree/main/python)
- Alternatively, search for precompiled wheels online for your specific Python version and system architecture


## 🙏 Acknowledgments

- Built with PyQt5
- Uses transformers, sentence-transformers, and other excellent libraries
- FastText for language detection
- LanguageTool for grammar checking
- xAI SDK for Grok API integration
- Various LLM API providers (Anthropic, OpenAI, Google, xAI, etc.)

---

**Note**: This tool is designed for processing ShareGPT-format conversation datasets. Ensure your input files follow the expected format with `conversations` arrays containing `from` and `value` fields.
