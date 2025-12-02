# Quick Start Guide - Sandbox Editor GUI

## Prerequisites

1. **Conda environment** (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate sandbox
   ```

2. **Or install dependencies manually**:
   ```bash
   pip install -r requirements.txt
   ```

## First Launch

1. **Test imports** (optional but recommended):
   ```bash
   python test_gui.py
   ```
   Should show: ✅ All imports successful!

2. **Launch the GUI**:
   ```bash
   # Basic usage (uses default documents directory)
   python -m sandbox_editor.main
   
   # Or specify your documents directory
   python -m sandbox_editor.main --documents ./Speech_to_LLM/documents
   ```

## Using the GUI

### Basic Chat
- Type your question in the input field and press Enter
- The AI will search your codebase and respond

### Commands
- `filter: filename.ext` - Filter queries to a specific file
- `filter: language python` - Filter queries to a specific language
- `filter: clear` - Clear the filter
- `edit: filename.ext instruction` - Edit a file with AI assistance
- `clear` - Clear conversation history

### Example Questions
- "What does the main function do?"
- "Show me all the Python classes"
- "How does the RAG system work?"
- `edit: test.py Add error handling to the main function`

## VS Code Integration

When you use the `edit:` command:
1. The system creates diff files in `.sandbox-editor-diffs/`
2. VS Code can detect these files
3. Open the diff files in VS Code to see changes

**Note**: A VS Code extension for automatic inline diffs is planned for the future.

## Troubleshooting

### "No module named 'PyQt6'"
```bash
pip install PyQt6 PyQt6-QScintilla
```

### "RAG engine initialization failed"
- Check that your documents directory exists
- Verify Ollama is running (if using local models)
- Check API keys in `.env` file (if using OpenAI/OpenRouter)

### GUI doesn't appear
- Check that you're in the conda environment
- Try running `python test_gui.py` first
- Check for error messages in the terminal

## Next Steps

- Read [RAG_EDITOR_README.md](RAG_EDITOR_README.md) for detailed documentation
- Check the main [README.md](README.md) for project overview

