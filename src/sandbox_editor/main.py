"""
Main entry point for RAG Editor GUI.
"""

import sys
import argparse
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from .gui.chat_window import ChatWindow
from .core.rag_engine import RAGEngine
from .vscode.diff_handler import VSCodeDiffHandler


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sandbox Editor - GUI chat interface for code editing")
    parser.add_argument(
        "--documents",
        type=str,
        default="documents",
        help="Path to documents directory (default: documents)"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API"
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:1b",
        help="Ollama model name (default: gemma3:1b)"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="openai/gpt-3.5-turbo",
        help="OpenRouter model name (default: openai/gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="ChromaDB collection name (default: default)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root directory for VS Code integration (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG engine
    print("🔄 Initializing RAG engine...")
    rag_engine = None
    try:
        rag_engine = RAGEngine(
            documents_path=args.documents,
            use_openai=args.openai,
            use_openrouter=args.openrouter,
            ollama_model=args.model,
            openai_model=args.openai_model,
            openrouter_model=args.openrouter_model,
            collection_name=args.collection
        )
        print("✅ RAG engine initialized")
    except Exception as e:
        print(f"❌ Error: RAG engine initialization failed: {e}")
        print("   Please check your configuration and try again.")
        print("   The GUI will not launch without a working RAG engine.")
        sys.exit(1)
    
    # Initialize VS Code handler
    workspace_root = args.workspace or Path.cwd()
    try:
        vscode_handler = VSCodeDiffHandler(workspace_root=str(workspace_root))
    except Exception as e:
        print(f"⚠️  Warning: VS Code handler initialization failed: {e}")
        print("   Continuing without VS Code integration...")
        vscode_handler = None
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Sandbox Editor")
    
    # Enable high DPI scaling (if available)
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    except AttributeError:
        # Attribute not available in this PyQt6 version, skip it
        pass
    
    # Create and show main window
    window = ChatWindow(rag_engine, vscode_handler)
    
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

