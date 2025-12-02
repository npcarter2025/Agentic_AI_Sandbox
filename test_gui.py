#!/usr/bin/env python3
"""
Quick test script to verify the GUI can be imported and initialized.
Run this before launching the full GUI to check for import errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 imported successfully")
    except ImportError as e:
        print(f"❌ PyQt6 import failed: {e}")
        print("   Install with: pip install PyQt6")
        return False
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from sandbox_editor.core.rag_engine import RAGEngine
        print("✅ RAG engine imported successfully")
    except ImportError as e:
        print(f"❌ RAG engine import failed: {e}")
        return False
    
    try:
        from sandbox_editor.gui.chat_window import ChatWindow
        print("✅ Chat window imported successfully")
    except ImportError as e:
        print(f"❌ Chat window import failed: {e}")
        return False
    
    try:
        from sandbox_editor.vscode.diff_handler import VSCodeDiffHandler
        print("✅ VS Code handler imported successfully")
    except ImportError as e:
        print(f"❌ VS Code handler import failed: {e}")
        return False
    
    print("\n✅ All imports successful! GUI should work.")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

