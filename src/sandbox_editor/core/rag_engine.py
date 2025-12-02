"""
RAG Engine Wrapper - Integrates with existing IntegratedRAG class.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

# Add parent directory to path to import existing RAG code
# This allows importing from the Speech_to_LLM directory
project_root = Path(__file__).parent.parent.parent.parent
speech_llm_path = project_root / "Speech_to_LLM"
if speech_llm_path.exists():
    sys.path.insert(0, str(speech_llm_path))

try:
    from Multi_Model_RAG_Voice import IntegratedRAG
except ImportError:
    try:
        # Fallback to non-voice version
        from Multi_Model_RAG import IntegratedRAG
    except ImportError:
        raise ImportError(
            "Could not import IntegratedRAG. Make sure Multi_Model_RAG_Voice.py or "
            "Multi_Model_RAG.py exists in the Speech_to_LLM directory."
        )


class RAGEngine:
    """
    Wrapper around IntegratedRAG for GUI use.
    Handles initialization, queries, and conversation management.
    """
    
    def __init__(
        self,
        documents_path: str = "documents",
        use_openai: bool = False,
        use_openrouter: bool = False,
        ollama_model: str = "gemma3:1b",
        openai_model: str = "gpt-3.5-turbo",
        openrouter_model: str = "openai/gpt-3.5-turbo",
        collection_name: str = "default",
        log_file: Optional[str] = ".rag_conversation.log",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the RAG engine."""
        self.documents_path = Path(documents_path)
        
        try:
            self.rag = IntegratedRAG(
                documents_path=str(self.documents_path),
                use_openai=use_openai,
                use_openrouter=use_openrouter,
                ollama_model=ollama_model,
                openai_model=openai_model,
                openrouter_model=openrouter_model,
                collection_name=collection_name,
                log_file=log_file,
                use_tokenizer=None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_voice_input=False  # GUI handles input
            )
            
            # Index documents (this handles missing directories gracefully)
            self.rag.index_documents(force_reindex=False, incremental=True)
            
            # Setup QA chain
            self.rag.setup_qa_chain()
            
            self.current_filter = None
            self.conversation_history: List[Dict[str, str]] = []
            self.initialized = True
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            raise
    
    def ask(self, question: str, filter_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Ask a question and get a response.
        
        Returns:
            Dict with 'answer', 'sources', 'has_code', 'code_blocks'
        """
        if not self.initialized:
            return {
                "answer": f"❌ RAG engine not initialized: {getattr(self, 'init_error', 'Unknown error')}",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        
        # Handle special commands
        if question.lower().startswith("filter:"):
            return self._handle_filter(question)
        
        if question.lower() == "clear":
            self.rag.memory.clear()
            self.current_filter = None
            self.conversation_history = []
            return {
                "answer": "✅ Conversation history cleared",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        
        # Get answer from RAG
        try:
            answer = self.rag.ask(question, filter_metadata=filter_metadata or self.current_filter)
        except Exception as e:
            return {
                "answer": f"❌ Error processing query: {str(e)}",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        
        # Extract code blocks from answer
        code_blocks = self._extract_code_blocks(answer)
        has_code = len(code_blocks) > 0
        
        # Get sources from last query (if available)
        sources = []
        if hasattr(self.rag, 'last_sources'):
            sources = self.rag.last_sources
        
        # Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "answer": answer,
            "sources": sources,
            "has_code": has_code,
            "code_blocks": code_blocks
        }
    
    def _handle_filter(self, question: str) -> Dict[str, any]:
        """Handle filter commands."""
        filter_cmd = question[7:].strip()
        
        if filter_cmd.startswith("language "):
            lang_name = filter_cmd[9:].strip()
            self.current_filter = {"language": lang_name}
            return {
                "answer": f"✅ Filter set to language: {lang_name}",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        elif any(filter_cmd.endswith(ext) for ext in self.rag.get_supported_extensions()):
            self.current_filter = {"file_name": filter_cmd}
            return {
                "answer": f"✅ Filter set to: {filter_cmd}",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        elif filter_cmd == "clear" or filter_cmd == "none":
            self.current_filter = None
            return {
                "answer": "✅ Filter cleared",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
        else:
            return {
                "answer": "❌ Invalid filter. Use 'filter: filename.ext', 'filter: language langname', or 'filter: clear'",
                "sources": [],
                "has_code": False,
                "code_blocks": []
            }
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown-formatted text."""
        import re
        
        code_blocks = []
        # Match ```language\ncode\n```
        pattern = r'```(\w+)?\n(.*?)```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        return code_blocks
    
    def edit_file(self, file_path: str, instruction: str) -> Dict[str, any]:
        """
        Edit a file using the RAG system.
        
        Returns:
            Dict with 'success', 'message', 'diff', 'file_path'
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            # Try relative to documents path
            file_path_obj = self.documents_path / file_path
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "message": f"File '{file_path}' not found",
                    "diff": None,
                    "file_path": None
                }
        
        try:
            original_content = file_path_obj.read_text(encoding='utf-8')
        except Exception:
            original_content = None
        
        success, message = self.rag.edit_file_inline(file_path_obj, instruction)
        
        modified_content = None
        language = None
        if success:
            try:
                modified_content = file_path_obj.read_text(encoding='utf-8')
            except Exception:
                modified_content = None
            try:
                language = self.rag.get_language_from_extension(file_path_obj)
            except Exception:
                language = None
        
        return {
            "success": success,
            "message": message,
            "file_path": str(file_path_obj),
            "original_content": original_content,
            "modified_content": modified_content,
            "language": language
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(set(self.rag.LANGUAGE_MAP.values()))

