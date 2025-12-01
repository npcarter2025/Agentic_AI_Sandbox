#!/usr/bin/env python3
"""
Integrated RAG CLI - RAG system with support for all file types
Features:
- All features from Inline_Edit_Rag.py
- Support for multiple programming languages:
  - TypeScript (.ts, .tsx)
  - JavaScript (.js, .jsx)
  - C/C++ (.c, .cpp, .cc, .cxx, .h, .hpp)
  - SystemVerilog (.sv, .svh)
  - Verilog (.v, .vh)
  - VHDL (.vhd, .vhdl)
  - Python (.py)
  - Perl (.pl, .pm, .pod)
  - Tcl/Tk (.tcl, .tk)
  - Build systems:
    - Makefile (Makefile, .mk, .make)
    - CMake (CMakeLists.txt, .cmake)
  - ASIC Physical Design formats:
    - LEF (.lef) - Library Exchange Format
    - DEF (.def) - Design Exchange Format
    - SPEF (.spef) - Standard Parasitic Exchange Format
    - SDC (.sdc) - Synopsys Design Constraints
    - LIB (.lib) - Liberty timing library
    - SDF (.sdf) - Standard Delay Format
    - SPICE (.sp, .spice, .cir) - Circuit simulation
    - CDL (.cdl) - Circuit Description Language
    - UPF (.upf) - Unified Power Format
    - CPF (.cpf) - Common Power Format
    - GDSII (.gds, .gds2) - Layout database
  - Text files (.txt, .md)
  - And more!
- Language-aware chunking and metadata
- Inline editing of any supported file type
"""

import os
import sys
import re
import json
import hashlib
import ast
import difflib
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from dotenv import load_dotenv

# Optional imports for voice input
try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
    from pynput import keyboard
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False

# Load environment variables
load_dotenv()


class IntegratedRAG:
    # Language mapping: extension -> language name
    LANGUAGE_MAP = {
        # Python
        '.py': 'python',
        # TypeScript
        '.ts': 'typescript',
        '.tsx': 'typescript',
        # JavaScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        # C/C++
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        # SystemVerilog/Verilog
        '.sv': 'systemverilog',
        '.svh': 'systemverilog',
        '.v': 'verilog',
        '.vh': 'verilog',
        # VHDL
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        # Text/Markdown
        '.txt': 'text',
        '.md': 'markdown',
        '.rst': 'text',
        # Other common languages
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'bash',
        '.fish': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.sql': 'sql',
        '.r': 'r',
        '.m': 'matlab',
        # Perl
        '.pl': 'perl',
        '.pm': 'perl',
        '.pod': 'perl',
        # Tcl/Tk
        '.tcl': 'tcl',
        '.tk': 'tcl',
        # Build systems
        '.mk': 'makefile',  # Makefile
        '.make': 'makefile',  # Makefile
        '.cmake': 'cmake',  # CMake script
        # ASIC Physical Design formats
        '.lef': 'lef',  # Library Exchange Format
        '.def': 'def',  # Design Exchange Format
        '.spef': 'spef',  # Standard Parasitic Exchange Format
        '.sdc': 'sdc',  # Synopsys Design Constraints
        '.tlf': 'tlf',  # Timing Library Format
        '.lib': 'lib',  # Liberty timing library
        '.sdf': 'sdf',  # Standard Delay Format
        '.sp': 'spice',  # SPICE netlist
        '.spice': 'spice',  # SPICE netlist
        '.cir': 'spice',  # SPICE circuit
        '.cdl': 'cdl',  # Circuit Description Language
        '.upf': 'upf',  # Unified Power Format
        '.cpf': 'cpf',  # Common Power Format
        '.gds': 'gds',  # GDSII layout
        '.gds2': 'gds',  # GDSII layout
        '.mw': 'milkyway',  # Cadence Milkyway database
        # Other formats
        '.lua': 'lua',
        '.clj': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.fs': 'fsharp',
        '.ex': 'elixir',
        '.erl': 'erlang',
    }
    
    def __init__(
        self,
        documents_path: str = "documents",
        use_openai: bool = False,
        use_openrouter: bool = False,
        ollama_model: str = "gemma3:1b",
        openai_model: str = "gpt-3.5-turbo",
        openrouter_model: str = "openai/gpt-3.5-turbo",
        collection_name: str = "default",
        memory_file: str = ".rag_memory.json",
        log_file: Optional[str] = ".rag_conversation.log",
        use_tokenizer: bool = None,  # None = auto-detect based on LLM provider
        chunk_size: int = None,  # None = use defaults (1000 chars or 1000 tokens)
        chunk_overlap: int = None,  # None = use defaults (200 chars or 200 tokens)
        use_voice_input: bool = False  # Enable voice input mode
    ):
        """
        Initialize the Integrated RAG system with multi-language support.
        
        Args:
            documents_path: Path to a file or directory containing files of any supported type
            use_openai: If True, use OpenAI API. If False, use local LLM (Ollama)
            use_openrouter: If True, use OpenRouter API (overrides use_openai)
            ollama_model: Ollama model name (default: "gemma3:1b")
            openai_model: OpenAI model name (default: "gpt-3.5-turbo"). Cheaper options: "gpt-4o-mini", "gpt-3.5-turbo"
            openrouter_model: OpenRouter model name (default: "openai/gpt-3.5-turbo"). Examples: "openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-pro"
            collection_name: Name of the ChromaDB collection (allows multiple indexes)
            memory_file: Path to save conversation history
            log_file: Path to log file for full conversation history (default: ".rag_conversation.log", None to disable)
            use_tokenizer: If True, use token-based chunking. If False, use character-based. If None, auto-detect (token-based for OpenAI/OpenRouter, character-based for Ollama)
            chunk_size: Chunk size in tokens (if use_tokenizer=True) or characters (if False). Default: 1000
            chunk_overlap: Chunk overlap in tokens (if use_tokenizer=True) or characters (if False). Default: 200
            use_voice_input: If True, enable voice input mode (press spacebar to record questions)
        """
        self.documents_path = Path(documents_path)
        self.use_openai = use_openai
        self.use_openrouter = use_openrouter
        self.ollama_model = ollama_model
        self.openai_model = openai_model
        self.openrouter_model = openrouter_model
        self.collection_name = collection_name
        self.memory_file = Path(memory_file)
        self.log_file = Path(log_file) if log_file else None
        
        # Auto-detect tokenizer usage if not specified
        if use_tokenizer is None:
            # Use tokenizer for cloud APIs (OpenAI/OpenRouter), character-based for local (Ollama)
            self.use_tokenizer = use_openai or use_openrouter
        else:
            self.use_tokenizer = use_tokenizer
        
        # Set chunk sizes (defaults: 1000 tokens/chars, 200 overlap)
        self.chunk_size = chunk_size if chunk_size is not None else 1000
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else 200
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.file_hashes = {}  # Track file hashes for incremental indexing
        self.hash_file = Path(".file_hashes.json")
        self.backup_dir = Path(".backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Voice input setup
        self.use_voice_input = use_voice_input
        self.voice_model = None
        self.voice_samplerate = 16000
        self.voice_recorded_audio = deque()
        self.voice_is_recording = False
        self.voice_recording_lock = threading.Lock()
        self.voice_listener = None
        self.voice_audio_stream = None
        self.voice_transcribed_text = None
        self.voice_recording_complete = threading.Event()
        
        if self.use_voice_input:
            if not VOICE_INPUT_AVAILABLE:
                print("⚠️  Voice input requested but required packages not installed.")
                print("   Install with: pip install vosk sounddevice pynput")
                print("   Also download a Vosk model: https://alphacephei.com/vosk/models")
                self.use_voice_input = False
            else:
                try:
                    print("Loading speech recognition model...")
                    self.voice_model = Model(lang="en-us")
                    print("✅ Speech model loaded!")
                except Exception as e:
                    print(f"⚠️  Could not load speech model: {e}")
                    print("   Voice input disabled. Install Vosk and download a model.")
                    self.use_voice_input = False
        
        # Create documents directory if it doesn't exist (only if it's a directory)
        if self.documents_path.is_dir() or not self.documents_path.exists():
            self.documents_path.mkdir(exist_ok=True)
        
        # Load conversation history
        self.load_memory()
        
        # Load file hashes for incremental indexing
        self.load_file_hashes()
        
        # Initialize embeddings (free, local)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def get_text_splitter(self):
        """
        Get the appropriate text splitter based on configuration.
        Returns TokenTextSplitter for token-based chunking, RecursiveCharacterTextSplitter for character-based.
        """
        if self.use_tokenizer:
            # Use token-based splitting (better for cloud APIs)
            try:
                # Try to use tiktoken (OpenAI's tokenizer) - works for most OpenAI-compatible models
                return TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    encoding_name="cl100k_base"  # Used by GPT-3.5, GPT-4, and most OpenAI models
                )
            except Exception as e:
                print(f"⚠️  Warning: TokenTextSplitter failed ({e}), falling back to character-based splitting")
                # Fallback to character-based if tokenizer fails
                return RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len
                )
        else:
            # Use character-based splitting (for local models)
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
    
    def get_language_from_extension(self, file_path: Path) -> str:
        """Get language name from file extension or special filename."""
        # Handle special filenames (no extension)
        filename_lower = file_path.name.lower()
        if filename_lower == 'makefile' or filename_lower.startswith('makefile.'):
            return 'makefile'
        if filename_lower == 'cmakelists.txt':
            return 'cmake'
        
        # Handle regular extensions
        ext = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(ext, 'text')
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        return list(self.LANGUAGE_MAP.keys())
    
    def load_memory(self):
        """Load conversation history from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # Restore conversation history
                    for msg in history.get('messages', []):
                        if msg['type'] == 'human':
                            self.memory.chat_memory.add_user_message(msg['content'])
                        elif msg['type'] == 'ai':
                            self.memory.chat_memory.add_ai_message(msg['content'])
                print(f"✅ Loaded conversation history from {self.memory_file}")
            except Exception as e:
                print(f"⚠️  Could not load conversation history: {e}")
    
    def save_memory(self):
        """Save conversation history to disk."""
        try:
            history = {'messages': [], 'last_updated': datetime.now().isoformat()}
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                for msg in self.memory.chat_memory.messages:
                    msg_type = 'human' if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage' else 'ai'
                    history['messages'].append({
                        'type': msg_type,
                        'content': msg.content if hasattr(msg, 'content') else str(msg)
                    })
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save conversation history: {e}")
    
    def log_conversation(self, question: str, answer: str, metadata: Optional[Dict] = None):
        """Log conversation to file with timestamp and metadata."""
        if not self.log_file:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract sources from answer if present
            sources = None
            if "📄 Sources:" in answer:
                sources_line = answer.split("📄 Sources:")[-1].strip()
                sources = sources_line.split(", ") if sources_line else None
                # Remove sources from answer for cleaner log
                answer_clean = answer.split("📄 Sources:")[0].strip()
            else:
                answer_clean = answer
            
            # Check if general knowledge was used
            used_general_knowledge = "💡" in answer
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Timestamp: {timestamp}\n")
                # Determine which model is being used
                if self.use_openrouter:
                    model_name = f"OpenRouter/{self.openrouter_model}"
                elif self.use_openai:
                    model_name = f"OpenAI/{self.openai_model}"
                else:
                    model_name = f"Ollama/{self.ollama_model}"
                f.write(f"Model: {model_name}\n")
                if metadata:
                    f.write(f"Filter: {metadata}\n")
                f.write("-"*80 + "\n")
                f.write(f"QUESTION:\n{question}\n\n")
                f.write(f"ANSWER:\n{answer_clean}\n")
                if sources:
                    f.write(f"\nSources: {', '.join(sources)}\n")
                if used_general_knowledge:
                    f.write("\n[Used general knowledge fallback]\n")
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"⚠️  Could not write to log file: {e}")
    
    def load_file_hashes(self):
        """Load file hashes for incremental indexing."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    self.file_hashes = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load file hashes: {e}")
                self.file_hashes = {}
    
    def save_file_hashes(self):
        """Save file hashes for incremental indexing."""
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save file hashes: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            return ""
    
    def backup_file(self, file_path: Path) -> Optional[Path]:
        """Create a backup of a file before editing."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
            return None
    
    def create_diff(self, old_content: str, new_content: str, filename: str = "file") -> str:
        """Create a unified diff between old and new content."""
        # Normalize line endings and ensure both end with newline
        old_content = old_content.replace('\r\n', '\n').replace('\r', '\n')
        new_content = new_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into lines, preserving newlines
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        # If content doesn't end with newline, add it for proper diff
        if old_lines and not old_lines[-1].endswith('\n'):
            old_lines[-1] += '\n'
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=filename,
            tofile=filename,
            lineterm='',
            n=3
        )
        
        return ''.join(diff)
    
    def find_function_or_class(self, file_path: Path, name: str) -> Optional[Dict]:
        """
        Find a function or class by name in a Python file.
        Returns dict with 'type', 'name', 'start_line', 'end_line', 'code', 'full_content'.
        Note: Currently only supports Python. Can be extended for other languages.
        """
        language = self.get_language_from_extension(file_path)
        if language != 'python':
            return None  # Only Python AST parsing supported for now
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == name:
                        code = ast.get_source_segment(content, node)
                        return {
                            'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                            'name': node.name,
                            'start_line': node.lineno,
                            'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                            'code': code or '',
                            'full_content': content
                        }
            
            return None
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
            return None
    
    def parse_python_file(self, file_path: Path, content: str) -> List[Document]:
        """
        Parse Python file using AST to extract functions and classes.
        Returns list of Document objects with metadata.
        """
        chunks = []
        try:
            tree = ast.parse(content)
            
            # Extract file-level docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                chunks.append(Document(
                    page_content=f"# {file_path.name}\n\n{docstring}",
                    metadata={
                        "source": str(file_path),
                        "type": "module_docstring",
                        "language": "python",
                        "file_name": file_path.name
                    }
                ))
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_code = ast.get_source_segment(content, node) or ""
                    func_doc = ast.get_docstring(node) or ""
                    
                    chunks.append(Document(
                        page_content=f"def {node.name}:\n{func_doc}\n\n{func_code}",
                        metadata={
                            "source": str(file_path),
                            "type": "function",
                            "name": node.name,
                            "language": "python",
                            "file_name": file_path.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                        }
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    class_code = ast.get_source_segment(content, node) or ""
                    class_doc = ast.get_docstring(node) or ""
                    
                    chunks.append(Document(
                        page_content=f"class {node.name}:\n{class_doc}\n\n{class_code}",
                        metadata={
                            "source": str(file_path),
                            "type": "class",
                            "name": node.name,
                            "language": "python",
                            "file_name": file_path.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                        }
                    ))
            
            # If no functions/classes found, chunk by size
            if not chunks:
                text_splitter = self.get_text_splitter()
                text_chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(text_chunks):
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": str(file_path),
                            "type": "text",
                            "language": "python",
                            "file_name": file_path.name,
                            "chunk_index": i
                        }
                    ))
            
        except SyntaxError:
            # If AST parsing fails, fall back to regular chunking
            text_splitter = self.get_text_splitter()
            text_chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "type": "text",
                        "language": "python",
                        "file_name": file_path.name,
                        "chunk_index": i
                    }
                ))
        
        return chunks
    
    def parse_code_file(self, file_path: Path, content: str) -> List[Document]:
        """
        Parse a code file (non-Python) with language-aware chunking.
        Returns list of Document objects with metadata.
        """
        language = self.get_language_from_extension(file_path)
        
        # Use text-based chunking for non-Python files
        # Could be enhanced with language-specific parsers in the future
        text_splitter = self.get_text_splitter()
        text_chunks = text_splitter.split_text(content)
        
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "type": "code",
                    "language": language,
                    "file_name": file_path.name,
                    "chunk_index": i
                }
            ))
        
        return chunks
    
    def load_documents(self, incremental: bool = True) -> Tuple[List[Document], List[str]]:
        """
        Load files of all supported types with metadata and incremental indexing support.
        Returns tuple of (documents, file_paths).
        """
        documents = []
        file_paths = []
        supported_extensions = self.get_supported_extensions()
        new_files = []
        changed_files = []
        
        # Check if it's a single file
        if self.documents_path.is_file():
            file_path = self.documents_path
            ext = file_path.suffix.lower()
            filename_lower = file_path.name.lower()
            
            # Check if it's a supported extension or special filename
            is_supported = (ext in supported_extensions or 
                          filename_lower == 'makefile' or 
                          filename_lower.startswith('makefile.') or
                          filename_lower == 'cmakelists.txt')
            
            if is_supported:
                file_hash = self.get_file_hash(file_path)
                stored_hash = self.file_hashes.get(str(file_path))
                
                if not incremental or file_hash != stored_hash:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():
                                language = self.get_language_from_extension(file_path)
                                
                                if language == 'python':
                                    # Use AST parsing for Python files
                                    chunks = self.parse_python_file(file_path, content)
                                    documents.extend(chunks)
                                else:
                                    # Use language-aware chunking for other files
                                    chunks = self.parse_code_file(file_path, content)
                                    documents.extend(chunks)
                                
                                file_paths.append(str(file_path))
                                self.file_hashes[str(file_path)] = file_hash
                                
                                if stored_hash is None:
                                    new_files.append(file_path)
                                    print(f"📄 New: {file_path} ({language})")
                                else:
                                    changed_files.append(file_path)
                                    print(f"🔄 Updated: {file_path} ({language})")
                                print(f"   Loaded: {file_path}")
                            else:
                                print(f"⚠️  File '{file_path}' is empty")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"⏭️  Skipped (unchanged): {file_path}")
            else:
                print(f"⚠️  File '{file_path}' is not a supported file type")
                print(f"   Supported extensions: {', '.join(sorted(set([ext for ext in supported_extensions[:20]])))}...")
        
        # Otherwise, treat it as a directory
        elif self.documents_path.is_dir():
            # Find all supported files
            for ext in supported_extensions:
                pattern = f"*{ext}"
                for file_path in self.documents_path.rglob(pattern):
                    # Skip if already processed
                    if str(file_path) in file_paths:
                        continue
                    file_hash = self.get_file_hash(file_path)
                    stored_hash = self.file_hashes.get(str(file_path))
                    
                    if not incremental or file_hash != stored_hash:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():
                                    language = self.get_language_from_extension(file_path)
                                    
                                    if language == 'python':
                                        # Use AST parsing for Python files
                                        chunks = self.parse_python_file(file_path, content)
                                        documents.extend(chunks)
                                    else:
                                        # Use language-aware chunking for other files
                                        chunks = self.parse_code_file(file_path, content)
                                        documents.extend(chunks)
                                    
                                    file_paths.append(str(file_path))
                                    self.file_hashes[str(file_path)] = file_hash
                                    
                                    if stored_hash is None:
                                        new_files.append(file_path)
                                        print(f"📄 New: {file_path} ({language})")
                                    else:
                                        changed_files.append(file_path)
                                        print(f"🔄 Updated: {file_path} ({language})")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        print(f"⏭️  Skipped (unchanged): {file_path}")
            
            # Also search for special filenames (no extension)
            special_filenames = ['Makefile', 'CMakeLists.txt']
            for special_name in special_filenames:
                for file_path in self.documents_path.rglob(special_name):
                    # Skip if already processed
                    if str(file_path) in file_paths:
                        continue
                    
                    file_hash = self.get_file_hash(file_path)
                    stored_hash = self.file_hashes.get(str(file_path))
                    
                    if not incremental or file_hash != stored_hash:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                if content.strip():
                                    language = self.get_language_from_extension(file_path)
                                    
                                    # Use language-aware chunking
                                    chunks = self.parse_code_file(file_path, content)
                                    documents.extend(chunks)
                                    
                                    file_paths.append(str(file_path))
                                    self.file_hashes[str(file_path)] = file_hash
                                    
                                    if stored_hash is None:
                                        new_files.append(file_path)
                                        print(f"📄 New: {file_path} ({language})")
                                    else:
                                        changed_files.append(file_path)
                                        print(f"🔄 Updated: {file_path} ({language})")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        print(f"⏭️  Skipped (unchanged): {file_path}")
        else:
            print(f"⚠️  Path '{self.documents_path}' does not exist")
            return [], []
        
        if not documents:
            print(f"\n⚠️  No new or changed files found at '{self.documents_path}'")
            if incremental:
                print(f"   (Use --reindex to force full reindex)")
            print(f"   Continuing anyway - you can still ask general questions")
            return [], []
        
        print(f"\n✅ Loaded {len(documents)} chunk(s) from {len(file_paths)} file(s)")
        if new_files:
            print(f"   📄 {len(new_files)} new file(s)")
        if changed_files:
            print(f"   🔄 {len(changed_files)} updated file(s)")
        
        # Save updated hashes
        self.save_file_hashes()
        
        return documents, file_paths
    
    def index_documents(self, force_reindex: bool = False, incremental: bool = True):
        """Index documents into the vector store with incremental support."""
        persist_directory = "./chroma_db"
        
        # Check if vectorstore already exists
        if not force_reindex and Path(persist_directory).exists():
            print(f"Loading existing vector store from {persist_directory}...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                print("✅ Loaded existing index")
                
                # Do incremental update if enabled
                if incremental:
                    print("\n🔄 Checking for new or changed files...")
                    new_docs, _ = self.load_documents(incremental=True)
                    if new_docs:
                        print(f"Adding {len(new_docs)} new/updated chunks...")
                        self.vectorstore.add_documents(new_docs)
                        print("✅ Incremental update complete")
                    else:
                        print("✅ All files up to date")
                
                return
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
        
        # Load and process documents
        documents, _ = self.load_documents(incremental=False if force_reindex else incremental)
        if not documents:
            # No documents found, but continue anyway - create empty vectorstore for general knowledge mode
            print("⚠️  No documents found, but continuing in general knowledge mode...")
            print("   (You can still ask questions, and add documents later)")
            # Create an empty vectorstore so the system can still work
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                print("✅ Initialized empty vector store (ready for documents)")
            except Exception as e:
                print(f"⚠️  Could not initialize vector store: {e}")
            return
        
        # Create vector store
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=self.collection_name
        )
        print(f"✅ Indexed documents in {persist_directory} (collection: {self.collection_name})")
    
    def setup_qa_chain(self, retry_count: int = 3):
        """Set up the question-answering chain with retry logic."""
        if not self.vectorstore:
            print("⚠️  No vector store available. Setting up in general knowledge mode...")
            print("   (You can ask general questions, but document-based queries won't work)")
            # We'll still set up the LLM for direct questions
            # But we'll handle the QA chain differently
        
        # Initialize LLM
        if self.use_openrouter:
            # Use OpenRouter API
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("❌ OPENROUTER_API_KEY not found in environment variables.")
                print("   Please create a .env file with: OPENROUTER_API_KEY=your_key")
                print("   Get your key from: https://openrouter.ai/keys")
                print("   Or use Ollama (default) or OpenAI by removing --openrouter flag")
                return
            print(f"Using OpenRouter: {self.openrouter_model}")
            print("   (Access multiple models through one API)")
            # OpenRouter uses OpenAI-compatible API
            llm = ChatOpenAI(
                model=self.openrouter_model,
                temperature=0,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/yourusername/RootSearch",  # Optional: for attribution
                    "X-Title": "RootSearch RAG System"  # Optional: for attribution
                }
            )
        elif self.use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("❌ OPENAI_API_KEY not found in environment variables.")
                print("   Please create a .env file with: OPENAI_API_KEY=your_key")
                print("   Or use Ollama (default) by removing --openai flag")
                return
            print(f"Using OpenAI: {self.openai_model}")
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=0,
                api_key=api_key
            )
        else:
            # Use Ollama for local LLM
            try:
                print(f"Using Ollama model: {self.ollama_model}")
                print("   (Make sure Ollama is running: 'ollama serve')")
                llm = ChatOllama(
                    model=self.ollama_model,
                    temperature=0,
                )
            except Exception as e:
                print(f"❌ Error connecting to Ollama: {e}")
                print("   Make sure Ollama is running: 'ollama serve'")
                print(f"   And that model '{self.ollama_model}' is available: 'ollama list'")
                return
        
        # Store LLM for use
        self.retry_count = retry_count
        self.llm = llm
        
        # Create QA chain with metadata filtering support (if vectorstore exists)
        if self.vectorstore:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Get more results for better context
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            print("✅ QA chain ready (with document retrieval)")
        else:
            # No vectorstore - we'll use direct LLM calls in ask() method
            self.qa_chain = None
            print("✅ LLM ready (general knowledge mode - no documents indexed)")
    
    def _extract_file_references(self, question: str) -> List[Path]:
        """
        Extract file references from a question.
        Looks for file paths, filenames, or phrases like "this file", "the file", etc.
        Returns list of file paths that exist.
        """
        file_refs = []
        
        # Look for file paths (absolute or relative)
        # Pattern: words that look like file paths (contain / or .ext)
        path_pattern = r'[\w/\.-]+\.(?:sv|svh|v|vh|vhd|vhdl|py|ts|tsx|js|jsx|cpp|c|h|hpp|java|go|rs|rb|php|swift|kt|scala|sh|yaml|yml|json|xml|html|css|sql|pl|pm|tcl|mk|cmake|lef|def|spef|sdc|lib|sdf|sp|spice|cir|cdl|upf|cpf|gds|gds2|txt|md|rst)'
        matches = re.findall(path_pattern, question, re.IGNORECASE)
        
        for match in matches:
            # Try as relative path from current directory
            file_path = Path(match)
            if file_path.exists():
                file_refs.append(file_path)
            else:
                # Try in documents directory
                doc_path = self.documents_path / match if self.documents_path.is_dir() else None
                if doc_path and doc_path.exists():
                    file_refs.append(doc_path)
                else:
                    # Try just the filename in current directory
                    filename_only = Path(match).name
                    if Path(filename_only).exists():
                        file_refs.append(Path(filename_only))
                    # Try in documents directory
                    elif self.documents_path.is_dir():
                        doc_file = self.documents_path / filename_only
                        if doc_file.exists():
                            file_refs.append(doc_file)
        
        # Look for phrases like "this file", "the file", "that file" and check for recently created files
        # This is a simple heuristic - could be enhanced
        if any(phrase in question.lower() for phrase in ["this file", "that file", "the file", "for this", "for that"]):
            # Check current directory for recently created .sv files (or other common extensions)
            current_dir = Path(".")
            for ext in ['.sv', '.v', '.py', '.ts', '.js', '.cpp', '.c']:
                for file_path in current_dir.glob(f"*{ext}"):
                    if file_path.is_file() and file_path not in file_refs:
                        file_refs.append(file_path)
                        break  # Just get one for "this file"
                if file_refs:
                    break
        
        return list(set(file_refs))  # Remove duplicates
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content if it exists."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def ask(self, question: str, filter_metadata: Optional[Dict] = None) -> str:
        """
        Ask a question and get an answer with retry logic and metadata filtering.
        
        Args:
            question: The question to ask
            filter_metadata: Optional metadata filter (e.g., {"file_name": "utils.py"}, {"language": "typescript"})
        """
        # If no QA chain (no documents), use direct LLM
        if not self.qa_chain:
            if not self.llm:
                return "❌ LLM not set up. Please check your configuration."
            
            # Direct LLM mode (no RAG)
            print("   💡 General knowledge mode (no documents indexed)")
            try:
                if self.use_openai or self.use_openrouter:
                    provider = "OpenRouter" if self.use_openrouter else "OpenAI"
                    print(f"   🔄 Querying {provider} API...")
                
                # Check for file references even without vectorstore
                file_refs = self._extract_file_references(question)
                enhanced_question = question
                
                if file_refs:
                    file_contents = []
                    for file_path in file_refs:
                        content = self._read_file_content(file_path)
                        if content:
                            language = self.get_language_from_extension(file_path)
                            file_contents.append(f"\n\n--- Content of {file_path.name} ({language}) ---\n{content}\n--- End of {file_path.name} ---")
                    
                    if file_contents:
                        enhanced_question = question + "\n\n" + "\n".join(file_contents)
                        print(f"   📄 Including content from: {', '.join([f.name for f in file_refs])}")
                
                # Use conversation history if available
                if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages') and self.memory.chat_memory.messages:
                    # Build context from conversation history
                    context = ""
                    for msg in self.memory.chat_memory.messages[-4:]:  # Last 4 messages for context
                        if hasattr(msg, 'content'):
                            msg_type = "Human" if msg.__class__.__name__ == 'HumanMessage' else "Assistant"
                            context += f"{msg_type}: {msg.content}\n"
                    
                    if context:
                        enhanced_question = f"{context}\n\nHuman: {enhanced_question}\n\nAssistant:"
                
                response = self.llm.invoke(enhanced_question)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # Update memory
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(answer)
                
                return answer
            except Exception as e:
                return f"❌ Error: {e}"
        
        # Check for file references in the question
        file_refs = self._extract_file_references(question)
        enhanced_question = question
        
        # If files are referenced, include their content in the question
        if file_refs:
            file_contents = []
            for file_path in file_refs:
                content = self._read_file_content(file_path)
                if content:
                    language = self.get_language_from_extension(file_path)
                    file_contents.append(f"\n\n--- Content of {file_path.name} ({language}) ---\n{content}\n--- End of {file_path.name} ---")
            
            if file_contents:
                enhanced_question = question + "\n\n" + "\n".join(file_contents)
                print(f"   📄 Including content from: {', '.join([f.name for f in file_refs])}")
        
        # Apply metadata filter if provided
        if filter_metadata and self.vectorstore:
            # Create filtered retriever
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": filter_metadata}
            )
            # Temporarily replace retriever
            original_retriever = self.qa_chain.retriever
            self.qa_chain.retriever = retriever
        
        # Retry logic
        last_error = None
        for attempt in range(self.retry_count):
            try:
                # Notify user about API call (RAG always uses API when OpenAI/OpenRouter is enabled)
                if (self.use_openai or self.use_openrouter) and attempt == 0:
                    provider = "OpenRouter" if self.use_openrouter else "OpenAI"
                    print(f"   🔄 Querying {provider} API (searching documents + generating answer)...")
                
                result = self.qa_chain.invoke({"question": enhanced_question})
                answer = result["answer"]
                
                # Restore original retriever if we changed it
                if filter_metadata and self.vectorstore:
                    self.qa_chain.retriever = original_retriever
                
                # Check if answer indicates no knowledge
                answer_lower = answer.lower().strip()
                dont_know_phrases = [
                    "i don't know", "i do not know", "i cannot", "i can't",
                    "i don't have", "i do not have", "unable to", "no information",
                    "not available", "not found in", "not in the"
                ]
                
                is_dont_know = any(phrase in answer_lower for phrase in dont_know_phrases)
                
                # Show sources with metadata
                sources = result.get("source_documents", [])
                has_sources = len(sources) > 0
                
                if sources:
                    source_info = []
                    for doc in sources:
                        meta = doc.metadata if hasattr(doc, "metadata") else {}
                        source_name = meta.get("file_name", Path(meta.get("source", "")).name if meta.get("source") else "unknown")
                        source_type = meta.get("type", "text")
                        language = meta.get("language", "unknown")
                        if meta.get("name"):
                            source_info.append(f"{source_name} ({language}, {source_type}: {meta['name']})")
                        else:
                            source_info.append(f"{source_name} ({language})")
                    if source_info:
                        answer += f"\n\n📄 Sources: {', '.join(set(source_info))}"
                
                # Fallback to general knowledge if RAG didn't help
                # This handles cases where:
                # 1. No sources found (general knowledge questions)
                # 2. Sources found but answer is still "I don't know" (irrelevant documents)
                if is_dont_know and hasattr(self, 'llm') and self.llm:
                    # Try direct LLM call for general knowledge questions
                    try:
                        # Notify user that we're making an API call
                        if self.use_openai or self.use_openrouter:
                            provider = "OpenRouter" if self.use_openrouter else "OpenAI"
                            print(f"   🔄 Making {provider} API call (RAG didn't find relevant documents)...")
                        
                        # Check if question is about current date/time
                        question_lower = question.lower()
                        is_date_question = any(word in question_lower for word in [
                            "today", "date", "what day", "current date", "what's the date",
                            "what date", "now", "current time", "what time"
                        ])
                        
                        # Add current date/time context if needed
                        enhanced_question = question
                        if is_date_question:
                            current_datetime = datetime.now()
                            current_date_str = current_datetime.strftime("%A, %B %d, %Y")
                            # Try to get timezone, fallback to local time if not available
                            try:
                                import time
                                timezone_name = time.tzname[0] if time.tzname else "local time"
                            except:
                                timezone_name = "local time"
                            current_time_str = current_datetime.strftime(f"%I:%M %p ({timezone_name})")
                            enhanced_question = f"""Current date and time information:
- Date: {current_date_str}
- Time: {current_time_str}
- Day of week: {current_datetime.strftime('%A')}

User question: {question}

Please answer the user's question using the current date/time information provided above."""
                        
                        direct_response = self.llm.invoke(enhanced_question)
                        if direct_response and hasattr(direct_response, 'content'):
                            fallback_answer = direct_response.content
                            # Only use fallback if it's different and more helpful
                            fallback_lower = fallback_answer.lower()
                            if not any(phrase in fallback_lower for phrase in dont_know_phrases) and len(fallback_answer) > 20:
                                source_note = "no relevant documents found" if not has_sources else "retrieved documents weren't relevant"
                                return f"{fallback_answer}\n\n💡 (Answered using general knowledge - {source_note})"
                    except Exception as fallback_error:
                        # Log the error but don't crash - return original answer
                        print(f"   ⚠️  Fallback failed: {str(fallback_error)[:100]}")
                        pass  # If fallback fails, return original answer
                
                return answer
            except Exception as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    error_msg = str(e)
                    if "500" in error_msg or "model runner" in error_msg.lower():
                        return f"❌ Ollama error: The model may not be loaded or there's a resource issue.\n   Try: ollama run {self.ollama_model}\n   Or check Ollama server logs.\n   Error: {error_msg}"
                    return f"❌ Error after {self.retry_count} attempts: {error_msg}"
        
        # Restore original retriever if we changed it
        if filter_metadata and self.vectorstore:
            self.qa_chain.retriever = original_retriever
        
        return f"❌ Error: {last_error}"
    
    def extract_code_blocks(self, text: str) -> List[tuple]:
        """Extract code blocks from markdown-formatted text."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.strip() if lang else '', code.strip()) for lang, code in matches]
    
    def _get_filename_from_user(self, default_filename: str, code_lang: str) -> Optional[str]:
        """
        Ask user for a filename, with a default suggestion.
        Returns the filename (with extension) or None if cancelled.
        """
        lang_to_ext = {
            'python': '.py',
            'typescript': '.ts',
            'javascript': '.js',
            'cpp': '.cpp',
            'c': '.c',
            'java': '.java',
            'go': '.go',
            'rust': '.rs',
            'ruby': '.rb',
            'php': '.php',
            'swift': '.swift',
            'kotlin': '.kt',
            'scala': '.scala',
            'bash': '.sh',
            'yaml': '.yaml',
            'json': '.json',
            'html': '.html',
            'css': '.css',
            'sql': '.sql',
            'systemverilog': '.sv',
            'verilog': '.v',
            'vhdl': '.vhd',
            'perl': '.pl',
            'tcl': '.tcl',
            'makefile': 'Makefile',
            'cmake': 'CMakeLists.txt',
            'lef': '.lef',
            'def': '.def',
            'spef': '.spef',
            'sdc': '.sdc',
            'lib': '.lib',
            'sdf': '.sdf',
            'spice': '.sp',
            'cdl': '.cdl',
            'upf': '.upf',
            'cpf': '.cpf',
        }
        default_ext = lang_to_ext.get(code_lang.lower(), '.txt')
        
        # Ensure default filename has the correct extension
        if not default_filename.endswith(default_ext) and default_ext not in ['Makefile', 'CMakeLists.txt']:
            # Remove any existing extension and add the correct one
            default_filename = Path(default_filename).stem + default_ext
        
        print(f"\n📝 Enter filename (default: {default_filename}, or 'cancel' to cancel): ", end="")
        user_input = input().strip()
        
        if not user_input:
            # User pressed Enter, use default
            return default_filename
        
        # Check for cancel
        if user_input.lower() in ['cancel', 'c', 'q', 'quit']:
            return None
        
        # User provided a custom name
        filename = user_input
        
        # If user didn't provide extension, add the default one
        if not any(filename.endswith(ext) for ext in lang_to_ext.values()):
            # Check if it looks like they want no extension (special cases)
            if code_lang.lower() in ['makefile', 'cmake']:
                # For Makefile/CMakeLists, don't add extension if user didn't provide one
                pass
            else:
                filename += default_ext
        
        # Validate filename (remove invalid characters)
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        
        return filename
    
    def edit_file_inline(self, file_path: Path, instruction: str) -> Tuple[bool, str]:
        """
        Edit a file based on a natural language instruction.
        Uses the LLM to generate the edit, then applies it with diff preview.
        Supports all file types.
        """
        if not self.llm:
            return False, "❌ LLM not set up. Please check your configuration."
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return False, f"❌ Error reading file: {e}"
        
        # Detect language
        language = self.get_language_from_extension(file_path)
        
        # Ask LLM to generate the edited version
        prompt = f"""You are a code editor. Edit the following {language} file according to the instruction.

CURRENT FILE ({file_path.name}):
```{language}
{file_content}
```

INSTRUCTION: {instruction}

REQUIREMENTS:
1. You MUST modify the code according to the instruction
2. Return the COMPLETE file with your changes
3. Preserve all existing code that doesn't need to change
4. Only modify what the instruction asks for
5. Keep the same file structure, imports, and formatting
6. Maintain the correct syntax for {language}

Return ONLY the complete edited code in a markdown code block:
```{language}
[complete file with edits applied]
```

Do not include explanations. Only return the code block."""
        
        try:
            # Use LLM directly (works with or without QA chain)
            if self.qa_chain:
                result = self.qa_chain.invoke({"question": prompt})
                answer = result["answer"]
            else:
                # Direct LLM call for file editing
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            
            # Extract code block
            code_blocks = self.extract_code_blocks(answer)
            if not code_blocks:
                return False, "❌ LLM did not return code in a code block. Try rephrasing your request."
            
            # Get the code (should be the full file)
            new_content = code_blocks[0][1]  # (lang, code)
            
            # Normalize whitespace for comparison
            old_normalized = file_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            new_normalized = new_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            # Check if content actually changed
            if old_normalized == new_normalized:
                return False, "❌ LLM returned unchanged code. The edit instruction may not have been clear enough, or the model didn't make the requested changes. Try rephrasing your instruction."
            
            # Create backup
            backup_path = self.backup_file(file_path)
            if not backup_path:
                return False, "❌ Could not create backup. Aborting edit."
            
            # Create diff
            diff = self.create_diff(file_content, new_content, file_path.name)
            
            # Show preview
            print(f"\n📝 Proposed changes to {file_path.name}:")
            print("=" * 60)
            print(diff)
            print("=" * 60)
            
            # Ask for approval
            while True:
                choice = input("\nApply changes? [y/n]: ").strip().lower()
                if choice == 'y' or choice == 'yes':
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        # Update file hash
                        self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                        self.save_file_hashes()
                        
                        return True, f"✅ Updated {file_path.name} (backup: {backup_path.name})"
                    except Exception as e:
                        return False, f"❌ Error writing file: {e}"
                elif choice == 'n' or choice == 'no':
                    return False, "❌ Edit cancelled by user."
                else:
                    print("Please enter 'y' or 'n'")
        
        except Exception as e:
            return False, f"❌ Error generating edit: {e}"
    
    def _voice_audio_callback(self, indata, frames, time, status):
        """Audio callback for microphone input in voice mode."""
        if status:
            print(status, file=sys.stderr)
        
        with self.voice_recording_lock:
            if self.voice_is_recording:
                self.voice_recorded_audio.append(bytes(indata))
    
    def _voice_keyboard_on_press(self, key):
        """Handle key press events for voice input - toggle recording on spacebar."""
        try:
            if key == keyboard.Key.space:
                with self.voice_recording_lock:
                    if not self.voice_is_recording:
                        # Start recording
                        self.voice_is_recording = True
                        self.voice_recorded_audio.clear()
                        print("\n🎤 RECORDING... (Press spacebar again to stop)")
                    else:
                        # Stop recording
                        self.voice_is_recording = False
                        print("⏹️  Recording stopped")
                        self.voice_recording_complete.set()
        except AttributeError:
            pass
    
    def _voice_process_recorded_audio(self) -> Optional[str]:
        """Process the recorded audio chunks and return transcribed text."""
        if not self.voice_recorded_audio:
            return None
        
        print("\n🔄 Processing audio...")
        
        # Create a new recognizer for this recording
        rec = KaldiRecognizer(self.voice_model, self.voice_samplerate)
        
        # Process all recorded audio chunks
        with self.voice_recording_lock:
            audio_chunks = list(self.voice_recorded_audio)
            self.voice_recorded_audio.clear()
        
        for chunk in audio_chunks:
            rec.AcceptWaveform(chunk)
        
        # Get final result
        result = rec.FinalResult()
        if result:
            try:
                result_dict = json.loads(result)
                if 'text' in result_dict and result_dict['text'].strip():
                    transcribed_text = result_dict['text'].strip()
                    print(f"\n🗣️  You said: {transcribed_text}")
                    return transcribed_text
                else:
                    print("⚠️  No speech detected in recording")
                    return None
            except Exception as e:
                print(f"❌ Error processing result: {e}")
                return None
        return None
    
    def _voice_get_input(self) -> Optional[str]:
        """Get voice input by waiting for spacebar recording. Returns transcribed text or None."""
        if not self.use_voice_input or not self.voice_model:
            return None
        
        # Ensure audio stream and listener are running (they should be initialized in chat())
        if self.voice_audio_stream is None:
            print("⚠️  Audio stream not initialized. Falling back to text input.")
            return None
        
        # Reset state for new recording
        self.voice_transcribed_text = None
        self.voice_recording_complete.clear()
        with self.voice_recording_lock:
            self.voice_is_recording = False
            self.voice_recorded_audio.clear()
        
        # Wait for user to press spacebar to start recording
        print("\n🎤 Press SPACEBAR to start recording your question...")
        
        # Wait for recording to start (spacebar pressed)
        timeout = 30  # 30 second timeout
        elapsed = 0
        while not self.voice_is_recording:
            time.sleep(0.1)
            elapsed += 0.1
            if elapsed > timeout:
                print("⏱️  Timeout waiting for recording to start")
                return None
        
        # Wait for recording to stop (spacebar pressed again)
        elapsed = 0
        while self.voice_is_recording:
            time.sleep(0.1)
            elapsed += 0.1
            if elapsed > 300:  # 5 minute max recording time
                print("⏱️  Maximum recording time reached")
                with self.voice_recording_lock:
                    self.voice_is_recording = False
                break
        
        # Small delay to ensure all audio chunks are captured
        time.sleep(0.2)
        
        # Process the audio
        transcribed_text = self._voice_process_recorded_audio()
        return transcribed_text
    
    def chat(self):
        """Start an interactive chat session with inline editing capabilities."""
        if not self.qa_chain and not self.llm:
            print("❌ LLM not set up. Please check your configuration.")
            return
        
        if not self.qa_chain:
            print("⚠️  Running in general knowledge mode (no documents indexed)")
            print("   You can still ask questions, and add documents later using --reindex")
            print()
        
        # Initialize log file with session header
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"NEW SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {self.openai_model if self.use_openai else self.ollama_model}\n")
                    f.write(f"Collection: {self.collection_name}\n")
                    f.write("="*80 + "\n")
                print(f"📝 Conversation logging to: {self.log_file}")
            except Exception as e:
                print(f"⚠️  Could not initialize log file: {e}")
        
        print("\n" + "="*60)
        print("💬 Integrated RAG - Multi-Language Chat with Code Editing")
        if self.use_voice_input:
            print("   🎤 VOICE MODE: Press SPACEBAR to record your question")
            print("   (You can still type commands like 'quit', 'clear', etc.)")
        else:
            print("   Type 'quit' or 'exit' to end the conversation")
        print("   Type 'clear' to clear conversation history")
        print("   Type 'filter: filename.ext' to filter by file")
        print("   Type 'filter: language typescript' to filter by language")
        print("   Type 'edit: filename.ext instruction' to edit a file")
        print("   When code is generated, you'll be asked to save or display it")
        if self.log_file:
            print(f"   📝 Full conversation logged to: {self.log_file}")
        print("="*60 + "\n")
        
        current_filter = None
        
        # Initialize voice input audio stream if using voice mode
        if self.use_voice_input and self.voice_model:
            try:
                self.voice_audio_stream = sd.RawInputStream(
                    samplerate=self.voice_samplerate,
                    blocksize=8000,
                    dtype="int16",
                    channels=1,
                    callback=self._voice_audio_callback
                )
                self.voice_audio_stream.start()
                self.voice_listener = keyboard.Listener(
                    on_press=self._voice_keyboard_on_press
                )
                self.voice_listener.start()
            except Exception as e:
                print(f"⚠️  Could not initialize voice input: {e}")
                print("   Falling back to text input mode")
                self.use_voice_input = False
        
        while True:
            try:
                # Get input from voice or text
                if self.use_voice_input and self.voice_model:
                    # Use voice input
                    question = self._voice_get_input()
                    if question is None:
                        # If voice input failed or was cancelled, allow typing as fallback
                        print("   (Type your question or press SPACEBAR to try again)")
                        question = input("You: ").strip()
                else:
                    # Use text input
                    question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["quit", "exit", "q"]:
                    # Cleanup voice input resources
                    if self.use_voice_input:
                        if self.voice_audio_stream:
                            try:
                                self.voice_audio_stream.stop()
                                self.voice_audio_stream.close()
                            except:
                                pass
                        if self.voice_listener:
                            try:
                                self.voice_listener.stop()
                            except:
                                pass
                    
                    self.save_memory()
                    if self.log_file:
                        try:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(f"\nSESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write("="*80 + "\n\n")
                        except:
                            pass
                    print("👋 Goodbye! Conversation history saved.")
                    if self.log_file:
                        print(f"📝 Full conversation logged to: {self.log_file}")
                    break
                
                if question.lower() == "clear":
                    self.memory.clear()
                    current_filter = None
                    print("✅ Conversation history cleared\n")
                    continue
                
                # Handle inline editing commands
                if question.lower().startswith("edit:"):
                    edit_cmd = question[5:].strip()
                    parts = edit_cmd.split(None, 1)
                    
                    if len(parts) < 2:
                        print("❌ Usage: edit: filename.ext instruction")
                        print("   Example: edit: app.ts Add error handling")
                        continue
                    
                    target = parts[0]
                    instruction = parts[1]
                    
                    # Find the file
                    file_path = None
                    if Path(target).exists():
                        file_path = Path(target)
                    elif (self.documents_path / target).exists():
                        file_path = self.documents_path / target
                    else:
                        # Search in documents directory for all supported extensions
                        for ext in self.get_supported_extensions():
                            pattern = f"*{ext}"
                            for f in self.documents_path.rglob(pattern):
                                if f.name == target:
                                    file_path = f
                                    break
                            if file_path:
                                break
                    
                    if not file_path or not file_path.exists():
                        print(f"❌ File '{target}' not found")
                        continue
                    
                    # Perform edit
                    success, message = self.edit_file_inline(file_path, instruction)
                    print(f"\n{message}")
                    
                    if success:
                        # Reindex the file
                        print(f"\n🔄 Reindexing {file_path.name}...")
                        self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                        new_docs, _ = self.load_documents(incremental=True)
                        if new_docs and self.vectorstore:
                            self.vectorstore.add_documents(new_docs)
                            
                            # Refresh QA chain to pick up updated documents
                            if self.qa_chain and self.llm:
                                retriever = self.vectorstore.as_retriever(
                                    search_kwargs={"k": 5}
                                )
                                self.qa_chain.retriever = retriever
                            
                            print("✅ File reindexed")
                    
                    print()
                    continue
                
                # Handle metadata filtering
                if question.lower().startswith("filter:"):
                    filter_cmd = question[7:].strip()
                    if filter_cmd.startswith("language "):
                        lang_name = filter_cmd[9:].strip()
                        current_filter = {"language": lang_name}
                        print(f"✅ Filter set to language: {lang_name}\n")
                    elif any(filter_cmd.endswith(ext) for ext in self.get_supported_extensions()):
                        current_filter = {"file_name": filter_cmd}
                        print(f"✅ Filter set to: {filter_cmd}\n")
                    elif filter_cmd.startswith("function "):
                        func_name = filter_cmd[9:].strip()
                        current_filter = {"type": "function", "name": func_name}
                        print(f"✅ Filter set to function: {func_name}\n")
                    elif filter_cmd == "clear" or filter_cmd == "none":
                        current_filter = None
                        print("✅ Filter cleared\n")
                    else:
                        print("❌ Invalid filter. Use 'filter: filename.ext', 'filter: language langname', or 'filter: function name'\n")
                    continue
                
                print("\n🤖 Assistant: ", end="", flush=True)
                answer = self.ask(question, filter_metadata=current_filter)
                print(answer)
                
                # Log conversation to file
                filter_meta = current_filter if current_filter else None
                self.log_conversation(question, answer, metadata=filter_meta)
                
                # Save memory after each exchange
                self.save_memory()
                
                # Check if answer contains code blocks
                code_blocks = self.extract_code_blocks(answer)
                if code_blocks:
                    code = None
                    code_lang = None
                    for lang, code_content in code_blocks:
                        if lang:
                            code = code_content
                            code_lang = lang
                            break
                    
                    if not code and code_blocks:
                        code_lang, code = code_blocks[0]
                        if not code_lang:
                            code_lang = 'text'
                    
                    if code:
                        print(f"\n💡 Found {code_lang} code block. What would you like to do?")
                        print("   [s] Save to file (current directory)")
                        print("   [d] Save to documents directory (will be indexed)")
                        print("   [p] Display code only (don't save)")
                        print("   [n] Nothing (skip)")
                        
                        while True:
                            choice = input("\nYour choice (s/d/p/n): ").strip().lower()
                            
                            if choice == 's' or choice == 'save':
                                # Generate default filename
                                default_filename = re.sub(r'[^\w\s-]', '', question.lower())
                                default_filename = re.sub(r'[-\s]+', '_', default_filename)
                                default_filename = default_filename[:30]
                                if not default_filename:
                                    default_filename = "generated_code"
                                
                                # Ask user for filename
                                filename = self._get_filename_from_user(default_filename, code_lang)
                                
                                if not filename:
                                    print("❌ Filename cancelled.\n")
                                    break
                                
                                try:
                                    output_path = Path(".")
                                    file_path = output_path / filename
                                    
                                    if file_path.exists():
                                        overwrite = input(f"⚠️  File '{file_path}' already exists. Overwrite? [y/n]: ").strip().lower()
                                        if overwrite != 'y' and overwrite != 'yes':
                                            print("❌ File not saved.\n")
                                            break
                                    
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        f.write(code)
                                    print(f"✅ Created file: {file_path}\n")
                                except Exception as e:
                                    print(f"❌ Error creating file: {e}\n")
                                break
                            
                            elif choice == 'd' or choice == 'documents':
                                # Generate default filename
                                default_filename = re.sub(r'[^\w\s-]', '', question.lower())
                                default_filename = re.sub(r'[-\s]+', '_', default_filename)
                                default_filename = default_filename[:30]
                                if not default_filename:
                                    default_filename = "generated_code"
                                
                                # Ask user for filename
                                filename = self._get_filename_from_user(default_filename, code_lang)
                                
                                if not filename:
                                    print("❌ Filename cancelled.\n")
                                    break
                                
                                try:
                                    if not self.documents_path.is_dir():
                                        self.documents_path.mkdir(parents=True, exist_ok=True)
                                    
                                    file_path = self.documents_path / filename
                                    
                                    if file_path.exists():
                                        overwrite = input(f"⚠️  File '{file_path}' already exists. Overwrite? [y/n]: ").strip().lower()
                                        if overwrite != 'y' and overwrite != 'yes':
                                            print("❌ File not saved.\n")
                                            break
                                    
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        f.write(code)
                                    print(f"✅ Created file: {file_path}")
                                    
                                    # Auto-index the new file
                                    if self.vectorstore:
                                        print("   🔄 Auto-indexing file...")
                                        try:
                                            language = self.get_language_from_extension(file_path)
                                            if language == 'python':
                                                chunks = self.parse_python_file(file_path, code)
                                            else:
                                                chunks = self.parse_code_file(file_path, code)
                                            
                                            if chunks:
                                                self.vectorstore.add_documents(chunks)
                                                # Update file hash
                                                self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                                                self.save_file_hashes()
                                                
                                                # Refresh QA chain to pick up new documents
                                                if self.qa_chain and self.llm:
                                                    retriever = self.vectorstore.as_retriever(
                                                        search_kwargs={"k": 5}
                                                    )
                                                    self.qa_chain.retriever = retriever
                                                
                                                print(f"   ✅ File indexed and ready to use!\n")
                                            else:
                                                print(f"   ⚠️  Could not create chunks for indexing\n")
                                        except Exception as e:
                                            print(f"   ⚠️  Could not auto-index file: {e}\n")
                                    elif self.llm:
                                        # No vectorstore yet, but we have LLM - create vectorstore and QA chain
                                        print("   🔄 Creating vectorstore and indexing file...")
                                        try:
                                            language = self.get_language_from_extension(file_path)
                                            if language == 'python':
                                                chunks = self.parse_python_file(file_path, code)
                                            else:
                                                chunks = self.parse_code_file(file_path, code)
                                            
                                            if chunks:
                                                # Create vectorstore if it doesn't exist
                                                persist_directory = "./chroma_db"
                                                self.vectorstore = Chroma.from_documents(
                                                    documents=chunks,
                                                    embedding=self.embeddings,
                                                    persist_directory=persist_directory,
                                                    collection_name=self.collection_name
                                                )
                                                
                                                # Update file hash
                                                self.file_hashes[str(file_path)] = self.get_file_hash(file_path)
                                                self.save_file_hashes()
                                                
                                                # Create QA chain now that we have a vectorstore
                                                retriever = self.vectorstore.as_retriever(
                                                    search_kwargs={"k": 5}
                                                )
                                                self.qa_chain = ConversationalRetrievalChain.from_llm(
                                                    llm=self.llm,
                                                    retriever=retriever,
                                                    memory=self.memory,
                                                    return_source_documents=True,
                                                    verbose=False
                                                )
                                                
                                                print(f"   ✅ File indexed and QA chain created!\n")
                                            else:
                                                print(f"   ⚠️  Could not create chunks for indexing\n")
                                        except Exception as e:
                                            print(f"   ⚠️  Could not auto-index file: {e}\n")
                                except Exception as e:
                                    print(f"❌ Error creating file: {e}\n")
                                break
                            
                            elif choice == 'p' or choice == 'display' or choice == 'print':
                                print(f"\n📝 Code:\n```{code_lang}")
                                print(code)
                                print("```\n")
                                break
                            
                            elif choice == 'n' or choice == 'nothing' or choice == '':
                                print("✅ Skipped saving code.\n")
                                break
                            
                            else:
                                print("❌ Invalid choice. Please enter 's' (save to current dir), 'd' (save to documents), 'p' (print), or 'n' (skip).")
                
                print()
                
            except KeyboardInterrupt:
                # Cleanup voice input resources
                if self.use_voice_input:
                    if self.voice_audio_stream:
                        try:
                            self.voice_audio_stream.stop()
                            self.voice_audio_stream.close()
                        except:
                            pass
                    if self.voice_listener:
                        try:
                            self.voice_listener.stop()
                        except:
                            pass
                
                self.save_memory()
                if self.log_file:
                    try:
                        with open(self.log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nSESSION ENDED (KeyboardInterrupt): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*80 + "\n\n")
                    except:
                        pass
                print("\n\n👋 Goodbye! Conversation history saved.")
                if self.log_file:
                    print(f"📝 Full conversation logged to: {self.log_file}")
                break
            except EOFError:
                # Cleanup voice input resources
                if self.use_voice_input:
                    if self.voice_audio_stream:
                        try:
                            self.voice_audio_stream.stop()
                            self.voice_audio_stream.close()
                        except:
                            pass
                    if self.voice_listener:
                        try:
                            self.voice_listener.stop()
                        except:
                            pass
                
                self.save_memory()
                if self.log_file:
                    try:
                        with open(self.log_file, 'a', encoding='utf-8') as f:
                            f.write(f"\nSESSION ENDED (EOF): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*80 + "\n\n")
                    except:
                        pass
                print("\n\n👋 Goodbye! Conversation history saved.")
                if self.log_file:
                    print(f"📝 Full conversation logged to: {self.log_file}")
                break
            except Exception as e:
                # Catch any other unexpected errors
                print(f"\n❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                print("\n⚠️  Continuing chat session... (Press Ctrl+C to exit)\n")
                # Don't break - continue the loop so user can keep chatting


def interactive_menu():
    """Interactive menu for configuring RAG system options."""
    print("\n" + "="*70)
    print("🔧 Interactive Configuration Menu")
    print("="*70)
    print("Press Enter to use defaults (shown in brackets)")
    print()
    
    config = {}
    
    # Documents path
    print("📁 Documents Path")
    print("   Path to a file or directory containing files to index")
    default_docs = "documents"
    docs_input = input(f"   Documents path [{default_docs}]: ").strip()
    config['documents'] = docs_input if docs_input else default_docs
    
    # LLM Provider
    print("\n🤖 LLM Provider")
    print("   1. Ollama (local, free)")
    print("   2. OpenAI (cloud, requires API key)")
    print("   3. OpenRouter (cloud, access multiple models)")
    provider_choice = input("   Choose provider [1]: ").strip()
    if not provider_choice:
        provider_choice = "1"
    
    config['use_openai'] = False
    config['use_openrouter'] = False
    
    if provider_choice == "2":
        config['use_openai'] = True
        print("\n   OpenAI Model")
        default_openai = "gpt-4o-mini"
        openai_model = input(f"   Model name [{default_openai}]: ").strip()
        config['openai_model'] = openai_model if openai_model else default_openai
        config['ollama_model'] = "gemma3:1b"  # Not used but set for consistency
        config['openrouter_model'] = "openai/gpt-3.5-turbo"  # Not used
    elif provider_choice == "3":
        config['use_openrouter'] = True
        print("\n   OpenRouter Model")
        default_openrouter = "openai/gpt-3.5-turbo"
        openrouter_model = input(f"   Model name [{default_openrouter}]: ").strip()
        config['openrouter_model'] = openrouter_model if openrouter_model else default_openrouter
        config['ollama_model'] = "gemma3:1b"  # Not used
        config['openai_model'] = "gpt-3.5-turbo"  # Not used
    else:
        print("\n   Ollama Model")
        default_ollama = "gemma3:1b"
        ollama_model = input(f"   Model name [{default_ollama}]: ").strip()
        config['ollama_model'] = ollama_model if ollama_model else default_ollama
        config['openai_model'] = "gpt-3.5-turbo"  # Not used
        config['openrouter_model'] = "openai/gpt-3.5-turbo"  # Not used
    
    # Indexing options
    print("\n📚 Indexing Options")
    reindex = input("   Force full reindex? [n]: ").strip().lower()
    config['reindex'] = reindex in ['y', 'yes', '1', 'true']
    
    no_incremental = input("   Disable incremental indexing? [n]: ").strip().lower()
    config['no_incremental'] = no_incremental in ['y', 'yes', '1', 'true']
    
    # Collection name
    print("\n🗂️  Collection Name")
    default_collection = "default"
    collection = input(f"   ChromaDB collection name [{default_collection}]: ").strip()
    config['collection'] = collection if collection else default_collection
    
    # Log file
    print("\n📝 Logging")
    default_log = ".rag_conversation.log"
    log_file = input(f"   Log file path [{default_log}] (or 'none' to disable): ").strip()
    config['log_file'] = log_file if log_file else default_log
    
    # Tokenizer options
    print("\n✂️  Chunking Options")
    print("   Tokenizer: 1=Auto-detect, 2=Force token-based, 3=Force character-based")
    tokenizer_choice = input("   Tokenizer option [1]: ").strip()
    if not tokenizer_choice:
        tokenizer_choice = "1"
    
    if tokenizer_choice == "2":
        config['use_tokenizer'] = True
        config['no_tokenizer'] = False
    elif tokenizer_choice == "3":
        config['use_tokenizer'] = False
        config['no_tokenizer'] = True
    else:
        config['use_tokenizer'] = None
        config['no_tokenizer'] = False
    
    # Chunk size
    chunk_size = input("   Chunk size [1000] (or Enter for default): ").strip()
    config['chunk_size'] = int(chunk_size) if chunk_size else None
    
    # Chunk overlap
    chunk_overlap = input("   Chunk overlap [200] (or Enter for default): ").strip()
    config['chunk_overlap'] = int(chunk_overlap) if chunk_overlap else None
    
    # Voice input
    print("\n🎤 Voice Input")
    voice = input("   Enable voice input mode? [n]: ").strip().lower()
    config['use_voice_input'] = voice in ['y', 'yes', '1', 'true']
    
    print("\n" + "="*70)
    print("✅ Configuration complete!")
    print("="*70 + "\n")
    
    return config


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated RAG CLI - RAG with support for all file types"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Show interactive configuration menu"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default=None,
        help="Path to a file or directory containing files of any supported type (default: 'documents')"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force full reindexing of all documents"
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental indexing (reindex everything)"
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI API instead of Ollama (Ollama is the default)"
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API instead of Ollama or OpenAI (Ollama is the default). Access multiple models through one API."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: 'gemma3:1b'). Examples: gemma3:1b, llama3.1:latest, mistral, qwen2.5"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        dest="openai_model",
        default=None,
        help="OpenAI model name (default: 'gpt-3.5-turbo'). Cheaper options: 'gpt-4o-mini' (cheapest), 'gpt-3.5-turbo'"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        dest="openrouter_model",
        default=None,
        help="OpenRouter model name (default: 'openai/gpt-3.5-turbo'). Examples: 'openai/gpt-4o-mini', 'anthropic/claude-3-haiku', 'google/gemini-pro', 'meta-llama/llama-3.1-8b-instruct'. See https://openrouter.ai/models"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (allows multiple indexes, default: 'default')"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        dest="log_file",
        default=None,
        help="Path to conversation log file (default: '.rag_conversation.log', use 'none' to disable)"
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        dest="use_tokenizer",
        help="Force token-based chunking (default: auto-detect based on LLM provider)"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        dest="no_tokenizer",
        help="Force character-based chunking (default: auto-detect based on LLM provider)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        dest="chunk_size",
        default=None,
        help="Chunk size in tokens (if using tokenizer) or characters (if not). Default: 1000"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        dest="chunk_overlap",
        default=None,
        help="Chunk overlap in tokens (if using tokenizer) or characters (if not). Default: 200"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        dest="use_voice_input",
        help="Enable voice input mode (press spacebar to record questions). Requires: pip install vosk sounddevice pynput"
    )
    
    args = parser.parse_args()
    
    # Check if interactive mode or no arguments provided
    use_interactive = args.interactive
    if not use_interactive:
        # Check if minimal arguments provided (just check a few key ones)
        has_args = any([
            args.documents is not None,
            args.openai,
            args.openrouter,
            args.model is not None,
            args.openai_model is not None,
            args.openrouter_model is not None,
            args.reindex,
            args.no_incremental,
            args.collection is not None,
            args.log_file is not None,
            args.use_tokenizer,
            args.no_tokenizer,
            args.chunk_size is not None,
            args.chunk_overlap is not None,
            args.use_voice_input
        ])
        if not has_args:
            # No arguments provided, show interactive menu
            use_interactive = True
    
    if use_interactive:
        config = interactive_menu()
        # Merge interactive config with any CLI args (CLI args override interactive)
        documents_path = args.documents if args.documents is not None else config['documents']
        use_openai = args.openai if args.openai else config['use_openai']
        use_openrouter = args.openrouter if args.openrouter else config['use_openrouter']
        ollama_model = args.model if args.model is not None else config['ollama_model']
        openai_model = args.openai_model if args.openai_model is not None else config['openai_model']
        openrouter_model = args.openrouter_model if args.openrouter_model is not None else config['openrouter_model']
        collection_name = args.collection if args.collection is not None else config['collection']
        log_file = args.log_file if args.log_file is not None else config['log_file']
        reindex = args.reindex if args.reindex else config['reindex']
        no_incremental = args.no_incremental if args.no_incremental else config['no_incremental']
        use_voice_input = args.use_voice_input if args.use_voice_input else config['use_voice_input']
        
        # Handle tokenizer
        if args.use_tokenizer:
            use_tokenizer = True
        elif args.no_tokenizer:
            use_tokenizer = False
        else:
            use_tokenizer = config['use_tokenizer']
        
        chunk_size = args.chunk_size if args.chunk_size is not None else config['chunk_size']
        chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else config['chunk_overlap']
    else:
        # Use CLI arguments with defaults
        documents_path = args.documents if args.documents is not None else "documents"
        use_openai = args.openai
        use_openrouter = args.openrouter
        ollama_model = args.model if args.model is not None else "gemma3:1b"
        openai_model = args.openai_model if args.openai_model is not None else "gpt-3.5-turbo"
        openrouter_model = args.openrouter_model if args.openrouter_model is not None else "openai/gpt-3.5-turbo"
        collection_name = args.collection if args.collection is not None else "default"
        log_file = args.log_file if args.log_file is not None else ".rag_conversation.log"
        reindex = args.reindex
        no_incremental = args.no_incremental
        use_voice_input = args.use_voice_input
        
        # Handle tokenizer
        if args.use_tokenizer:
            use_tokenizer = True
        elif args.no_tokenizer:
            use_tokenizer = False
        else:
            use_tokenizer = None
        
        chunk_size = args.chunk_size
        chunk_overlap = args.chunk_overlap
    
    # Handle log file option
    log_file_final = None if (log_file and log_file.lower() == 'none') else log_file
    
    # Initialize RAG system
    rag = IntegratedRAG(
        documents_path=documents_path,
        use_openai=use_openai,
        use_openrouter=use_openrouter,
        ollama_model=ollama_model,
        openai_model=openai_model,
        openrouter_model=openrouter_model,
        collection_name=collection_name,
        log_file=log_file_final,
        use_tokenizer=use_tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_voice_input=use_voice_input
    )
    
    # Index documents
    rag.index_documents(
        force_reindex=reindex,
        incremental=not no_incremental
    )
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Start chat
    rag.chat()


if __name__ == "__main__":
    main()

