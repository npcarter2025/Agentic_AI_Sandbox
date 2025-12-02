"""
Chat Window - Main GUI interface for RAG Editor.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame,
    QSplitter, QMessageBox, QFileDialog, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QTextDocument
from datetime import datetime

from ..core.rag_engine import RAGEngine
from ..vscode.diff_handler import VSCodeDiffHandler


class CodeHighlighter(QSyntaxHighlighter):
    """Simple syntax highlighter for code blocks in chat."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))
        keyword_format.setFontWeight(700)
        
        keywords = [
            'def', 'class', 'import', 'from', 'if', 'else', 'elif',
            'for', 'while', 'return', 'try', 'except', 'finally',
            'async', 'await', 'with', 'as', 'pass', 'break', 'continue'
        ]
        
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            self.highlighting_rules.append((pattern, keyword_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class QueryWorker(QThread):
    """Worker thread for handling RAG queries without blocking UI."""
    
    response_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, rag_engine: RAGEngine, question: str):
        super().__init__()
        self.rag_engine = rag_engine
        self.question = question
    
    def run(self):
        try:
            response = self.rag_engine.ask(self.question)
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatWindow(QMainWindow):
    """Main chat window for RAG Editor."""
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        vscode_handler: Optional[VSCodeDiffHandler] = None,
        parent=None
    ):
        super().__init__(parent)
        self.rag_engine = rag_engine
        self.vscode_handler = vscode_handler or VSCodeDiffHandler()
        self.last_code_blocks: List[Dict[str, str]] = []  # Store code blocks from last response
        
        self.setWindowTitle("Sandbox Editor - AI Code Assistant")
        self.setGeometry(100, 100, 900, 700)
        
        self.init_ui()
        self.setup_styles()
        
        # Check if RAG engine initialized successfully
        if not hasattr(self.rag_engine, 'initialized') or not self.rag_engine.initialized:
            self.add_message(
                "assistant",
                "⚠️ Warning: RAG engine initialization had issues. Some features may not work."
            )
    
    def init_ui(self):
        """Initialize the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header
        header = QLabel("💬 Sandbox Editor - Chat with your codebase")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        # Chat area (scrollable)
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("Monaco", 11) if sys.platform == "darwin" else QFont("Consolas", 10))
        main_layout.addWidget(self.chat_area, stretch=1)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question or type 'edit: filename.ext instruction' to edit code...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field, stretch=1)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setDefault(True)
        input_layout.addWidget(self.send_button)
        
        main_layout.addLayout(input_layout)
        
        # Status bar with save button
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        status_layout.addWidget(self.status_label, stretch=1)
        
        self.save_code_button = QPushButton("💾 Save Code")
        self.save_code_button.setEnabled(False)
        self.save_code_button.clicked.connect(self.save_code_blocks)
        self.save_code_button.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
        """)
        status_layout.addWidget(self.save_code_button)
        main_layout.addLayout(status_layout)
        
        # Add welcome message
        self.add_message("assistant", "👋 Welcome to Sandbox Editor! I can help you understand and edit your codebase.\n\n"
                                      "Try asking questions about your code, or use commands like:\n"
                                      "• `filter: filename.ext` - Filter by file\n"
                                      "• `filter: language python` - Filter by language\n"
                                      "• `edit: filename.ext instruction` - Edit a file\n"
                                      "• `clear` - Clear conversation history")
    
    def setup_styles(self):
        """Setup application styles."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTextEdit {
                background-color: #252526;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 10px;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0a4d73;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666;
            }
            QLabel {
                color: #d4d4d4;
            }
        """)
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat area."""
        timestamp = datetime.now().strftime("%H:%M")
        
        if role == "user":
            prefix = f"<b style='color: #4ec9b0;'>You [{timestamp}]</b>"
        else:
            prefix = f"<b style='color: #569cd6;'>Assistant [{timestamp}]</b>"
        
        # Format code blocks
        formatted_content = self.format_message(content)
        
        message_html = f"""
        <div style='margin: 10px 0; padding: 10px; background-color: #2d2d30; border-radius: 4px;'>
            {prefix}
            <div style='margin-top: 5px;'>{formatted_content}</div>
        </div>
        """
        
        self.chat_area.append(message_html)
        
        # Scroll to bottom
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def format_message(self, content: str) -> str:
        """Format message content with markdown and code highlighting."""
        # Escape HTML first
        import html
        content = html.escape(content)
        
        # Convert markdown code blocks to HTML
        import re
        
        # Code blocks
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return f"<pre style='background-color: #1e1e1e; padding: 10px; border-radius: 4px; overflow-x: auto;'><code class='language-{lang}'>{code}</code></pre>"
        
        content = re.sub(r'```(\w+)?\n(.*?)```', replace_code_block, content, flags=re.DOTALL)
        
        # Inline code
        content = re.sub(r'`([^`]+)`', r"<code style='background-color: #1e1e1e; padding: 2px 4px; border-radius: 2px;'>\1</code>", content)
        
        # Line breaks
        content = content.replace('\n', '<br>')
        
        return content
    
    def send_message(self):
        """Send the current message."""
        question = self.input_field.text().strip()
        if not question:
            return
        
        # Clear input
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Add user message to chat
        self.add_message("user", question)
        
        # Handle edit commands
        if question.lower().startswith("edit:"):
            self.handle_edit_command(question)
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
            self.status_label.setText("Ready")
            self.save_code_button.setEnabled(False)
            return
        
        # Process query in background thread
        self.worker = QueryWorker(self.rag_engine, question)
        self.worker.response_ready.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
    
    def handle_response(self, response: Dict):
        """Handle response from RAG engine."""
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        code_blocks = response.get("code_blocks", [])
        
        # Store code blocks for saving
        self.last_code_blocks = code_blocks
        
        # Add sources to answer if available
        if sources:
            source_text = "\n\n📄 Sources: " + ", ".join(sources[:3])  # Show first 3
            answer += source_text
        
        # Add assistant message
        self.add_message("assistant", answer)
        
        # Enable save button if there are code blocks
        if code_blocks:
            self.save_code_button.setEnabled(True)
            self.status_label.setText(f"Ready - {len(code_blocks)} code block(s) available to save")
        else:
            self.save_code_button.setEnabled(False)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        if not code_blocks:
            self.status_label.setText("Ready")
    
    def handle_error(self, error_msg: str):
        """Handle errors from RAG engine."""
        self.add_message("assistant", f"❌ Error: {error_msg}")
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_label.setText("Ready")
        self.save_code_button.setEnabled(False)
    
    def handle_edit_command(self, command: str):
        """Handle edit commands."""
        # Parse edit command: "edit: filename.ext instruction"
        parts = command[5:].strip().split(None, 1)
        
        if len(parts) < 2:
            self.add_message("assistant", "❌ Usage: edit: filename.ext instruction")
            return
        
        file_path = parts[0]
        instruction = parts[1]
        
        self.status_label.setText(f"Editing {file_path}...")
        
        # Perform edit
        result = self.rag_engine.edit_file(file_path, instruction)
        
        if result["success"]:
            message = result["message"]
            self.add_message("assistant", message)
            
            # Create diff file for VS Code and open diff view
            if (
                self.vscode_handler
                and result.get("original_content")
                and result.get("modified_content")
            ):
                try:
                    diff_info = self.vscode_handler.create_diff(
                        result["file_path"],
                        result["original_content"],
                        result["modified_content"],
                        result.get("language")
                    )
                    self.open_diff_in_vscode(diff_info)
                    self.status_label.setText("✅ Edit applied. VS Code diff opened.")
                except Exception as diff_error:
                    self.status_label.setText("✅ Edit applied (diff unavailable)")
                    self.add_message(
                        "assistant",
                        f"⚠️ Could not open diff in VS Code: {diff_error}"
                    )
            else:
                self.status_label.setText("✅ Edit applied.")
        else:
            self.add_message("assistant", result["message"])
        
        self.status_label.setText("Ready")
        self.save_code_button.setEnabled(False)

    def open_diff_in_vscode(self, diff_info: Dict[str, str]):
        """Open a diff in VS Code using the `code --diff` CLI."""
        if not diff_info:
            return
        
        original = diff_info.get("original_file")
        modified = diff_info.get("modified_file")
        if not original or not modified:
            return
        
        try:
            subprocess.run(
                ["code", "--diff", original, modified, "--reuse-window"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "VS Code CLI not found",
                "Could not find the `code` command. In VS Code, open the Command Palette "
                "(⇧⌘P / Ctrl+Shift+P) and run 'Shell Command: Install 'code' command in PATH', "
                "then try again."
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "VS Code Diff Error",
                f"Could not open diff in VS Code:\n{exc}"
            )
    
    def save_code_blocks(self):
        """Save code blocks from the last response to files."""
        if not self.last_code_blocks:
            QMessageBox.warning(self, "No Code", "No code blocks available to save.")
            return
        
        # Get default save directory (documents path or current directory)
        default_dir = str(self.rag_engine.documents_path) if hasattr(self.rag_engine, 'documents_path') else str(Path.cwd())
        
        if len(self.last_code_blocks) == 1:
            # Single code block - ask for filename
            code_block = self.last_code_blocks[0]
            language = code_block.get("language", "txt")
            
            # Suggest extension based on language
            ext_map = {
                "python": "py",
                "javascript": "js",
                "typescript": "ts",
                "java": "java",
                "cpp": "cpp",
                "c": "c",
                "systemverilog": "sv",
                "verilog": "v",
                "vhdl": "vhd",
                "html": "html",
                "css": "css",
                "json": "json",
                "yaml": "yaml",
                "markdown": "md",
                "bash": "sh",
                "shell": "sh",
            }
            extension = ext_map.get(language.lower(), "txt")
            default_filename = f"code.{extension}"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Code Block",
                str(Path(default_dir) / default_filename),
                f"{language.upper()} files (*.{extension});;All files (*.*)"
            )
            
            if file_path:
                try:
                    Path(file_path).write_text(code_block["code"], encoding='utf-8')
                    self.status_label.setText(f"✅ Saved to {Path(file_path).name}")
                    QMessageBox.information(self, "Success", f"Code saved to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
        else:
            # Multiple code blocks - ask for directory
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Code Blocks",
                default_dir
            )
            
            if save_dir:
                saved_count = 0
                errors = []
                
                for i, code_block in enumerate(self.last_code_blocks):
                    language = code_block.get("language", "txt")
                    
                    # Suggest extension based on language
                    ext_map = {
                        "python": "py",
                        "javascript": "js",
                        "typescript": "ts",
                        "java": "java",
                        "cpp": "cpp",
                        "c": "c",
                        "systemverilog": "sv",
                        "verilog": "v",
                        "vhdl": "vhd",
                        "html": "html",
                        "css": "css",
                        "json": "json",
                        "yaml": "yaml",
                        "markdown": "md",
                        "bash": "sh",
                        "shell": "sh",
                    }
                    extension = ext_map.get(language.lower(), "txt")
                    filename = f"code_block_{i+1}.{extension}"
                    file_path = Path(save_dir) / filename
                    
                    try:
                        file_path.write_text(code_block["code"], encoding='utf-8')
                        saved_count += 1
                    except Exception as e:
                        errors.append(f"{filename}: {str(e)}")
                
                if saved_count > 0:
                    msg = f"✅ Saved {saved_count} code block(s) to:\n{save_dir}"
                    if errors:
                        msg += f"\n\nErrors:\n" + "\n".join(errors)
                    QMessageBox.information(self, "Success", msg)
                    self.status_label.setText(f"✅ Saved {saved_count} file(s)")
                else:
                    QMessageBox.critical(self, "Error", "Failed to save files:\n" + "\n".join(errors))

