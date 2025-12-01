# Agentic AI Sandbox

This repo is a collection of AI tools that I've been working on for speech-to-text processing and RAG (Retrieval-Augmented Generation) systems.
From high-quality to low-quality, you'll find a mix of things that I've been tinkering with.
Everything is functional, but still work in progress.

Ultimately I'm trying to build a Front-end CLI Interface to run local LLM models for general conversations and rapid documentation. 
I've also made versions that embed your documents into a vector database, for better LLM contextualization.
I also have a model that will allow you to connect to Ollama, Open Router, or OpenAI.

## Projects

### Speech-to-Text Tools
- **Speech_to_English_Txt_Spacebar.py** - Spacebar-controlled speech-to-text using Vosk ASR
- **Speech_to_English_Txt_to_Spanish_Txt.py** - Speech-to-text with translation to Spanish
- **Speech_to_English_Txt_to_Spanish_Txt_Spacebar.py** - Spacebar-controlled version with translation

### RAG System
- **Multi_Model_RAG.py** - Integrated RAG CLI with support for multiple file types and programming languages
  - Supports Python, TypeScript, JavaScript, C/C++, SystemVerilog, Verilog, VHDL, and many more
  - Inline code editing capabilities
  - Multiple LLM providers (Ollama, OpenAI, OpenRouter)
  - Voice input mode (with `--voice` flag)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt  # If available
   # Or install manually:
   pip install vosk sounddevice pynput langchain-chroma langchain-huggingface langchain-openai langchain-ollama chromadb python-dotenv
   ```

2. Download Vosk models (for speech recognition):
   - Visit: https://alphacephei.com/vosk/models
   - Download the `en-us` model for English recognition

3. Set up environment variables (create `.env` file):
   ```
   OPENAI_API_KEY=your_key_here  # Optional, for OpenAI
   OPENROUTER_API_KEY=your_key_here  # Optional, for OpenRouter
   ```

## Usage

### RAG System
```bash
# Basic usage with Ollama (default)
python Speech_to_LLM/Multi_Model_RAG.py --documents ./documents

# With voice input
python Speech_to_LLM/Multi_Model_RAG.py --voice --documents ./documents

# With OpenAI
python Speech_to_LLM/Multi_Model_RAG.py --openai --openai-model gpt-4o-mini --documents ./documents
```

### Speech-to-Text
```bash
python Speech_to_LLM/Speech_to_English_Txt_Spacebar.py
# Press SPACEBAR to start/stop recording
```

## Features

- **Multi-language support**: Index and query code in many programming languages
- **Voice input**: Ask questions using speech recognition
- **Inline editing**: Edit files using natural language instructions
- **Incremental indexing**: Only reindexes changed files
- **Conversation logging**: Full conversation history logging
- **Metadata filtering**: Filter queries by file, language, or function

## License

[Add your license here]

