# Sandbox Editor - GUI Chat Interface

A modern GUI chat interface for your RAG-powered code editing system. Chat with your codebase and see edits appear in VS Code.

## Features

- 💬 **Chat Interface**: Clean, modern chat UI built with PyQt6
- 🔍 **RAG Integration**: Full integration with your existing RAG system
- 📝 **VS Code Integration**: Code edits appear as diffs in VS Code
- 🎨 **Syntax Highlighting**: Code blocks in chat are syntax-highlighted
- 🔄 **Real-time Processing**: Non-blocking UI with background query processing

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate sandbox
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run the GUI
python -m sandbox_editor.main

# Or if installed as package:
sandbox-editor
```

### With Options

```bash
# Specify documents directory
python -m sandbox_editor.main --documents ./my_code

# Use OpenAI
python -m sandbox_editor.main --openai --openai-model gpt-4

# Use OpenRouter
python -m sandbox_editor.main --openrouter --openrouter-model openai/gpt-4

# Specify workspace for VS Code integration
python -m sandbox_editor.main --workspace /path/to/workspace
```

## Chat Commands

- **Ask questions**: Just type your question about the codebase
- **Filter by file**: `filter: filename.ext`
- **Filter by language**: `filter: language python`
- **Edit files**: `edit: filename.ext instruction`
- **Clear history**: `clear`

## VS Code Integration

When you use the `edit:` command, the system creates diff files in `.sandbox-editor-diffs/` that VS Code can detect. 

### Viewing Diffs in VS Code

1. Open VS Code in your workspace
2. Use the `edit:` command in RAG Editor
3. VS Code will show the diff in the editor (you may need to refresh or open the diff file)

### Future: VS Code Extension

A VS Code extension is planned to automatically show diffs inline, similar to Cursor's accept/reject blocks.

## Project Structure

```
src/sandbox_editor/
├── __init__.py
├── main.py              # Entry point
├── core/
│   ├── __init__.py
│   └── rag_engine.py    # RAG engine wrapper
├── gui/
│   ├── __init__.py
│   └── chat_window.py   # Main chat window
└── vscode/
    ├── __init__.py
    └── diff_handler.py  # VS Code diff communication
```

## Development

### Installing in Development Mode

```bash
pip install -e .
```

### Running Tests

```bash
pytest
```

## Requirements

- Python 3.10+
- PyQt6
- All dependencies from `requirements.txt`

## License

MIT

