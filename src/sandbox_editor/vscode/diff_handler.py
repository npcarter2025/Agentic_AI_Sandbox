"""
VS Code Diff Handler - Communicates code changes to VS Code.
Uses file-based approach that VS Code can detect.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import difflib


class VSCodeDiffHandler:
    """
    Handles communication with VS Code for showing diffs.
    Creates temporary diff files that VS Code can detect and display.
    """
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize the diff handler.
        
        Args:
            workspace_root: Root directory of the workspace (for relative paths)
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.diff_dir = self.workspace_root / ".sandbox-editor-diffs"
        self.diff_dir.mkdir(exist_ok=True)
        
        # Create .gitignore to exclude diff files
        gitignore = self.diff_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
    
    def create_diff(
        self,
        file_path: str,
        original_content: str,
        modified_content: str,
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create a diff file that VS Code can display.
        
        Args:
            file_path: Path to the file being modified
            original_content: Original file content
            modified_content: Modified file content
            language: Programming language (for syntax highlighting)
        
        Returns:
            Dict with 'diff_file', 'original_file', 'modified_file', 'metadata_file'
        """
        file_path_obj = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = file_path_obj.stem
        
        # Create temporary files
        original_file = self.diff_dir / f"{base_name}.original.{timestamp}"
        modified_file = self.diff_dir / f"{base_name}.modified.{timestamp}"
        diff_file = self.diff_dir / f"{base_name}.diff.{timestamp}"
        metadata_file = self.diff_dir / f"{base_name}.metadata.{timestamp}.json"
        
        # Write original and modified content
        original_file.write_text(original_content, encoding='utf-8')
        modified_file.write_text(modified_content, encoding='utf-8')
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original_content.splitlines(keepends=True),
            modified_content.splitlines(keepends=True),
            fromfile=f"original/{file_path_obj.name}",
            tofile=f"modified/{file_path_obj.name}",
            lineterm=''
        ))
        diff_content = ''.join(diff_lines)
        diff_file.write_text(diff_content, encoding='utf-8')
        
        # Create metadata file
        metadata = {
            "file_path": str(file_path_obj),
            "relative_path": str(file_path_obj.relative_to(self.workspace_root)) if file_path_obj.is_relative_to(self.workspace_root) else str(file_path_obj),
            "language": language,
            "timestamp": timestamp,
            "original_file": str(original_file),
            "modified_file": str(modified_file),
            "diff_file": str(diff_file),
            "created_at": datetime.now().isoformat()
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        
        return {
            "diff_file": str(diff_file),
            "original_file": str(original_file),
            "modified_file": str(modified_file),
            "metadata_file": str(metadata_file),
            "metadata": metadata
        }
    
    def create_edit_suggestion(
        self,
        file_path: str,
        instruction: str,
        suggested_code: str,
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create an edit suggestion that can be shown in VS Code.
        
        Args:
            file_path: Path to the file
            instruction: The instruction that led to this edit
            suggested_code: The suggested code
            language: Programming language
        
        Returns:
            Dict with file paths and metadata
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return {
                "error": f"File {file_path} does not exist"
            }
        
        original_content = file_path_obj.read_text(encoding='utf-8')
        
        # Try to merge the suggested code intelligently
        # For now, we'll create a full replacement
        # In the future, this could use AST parsing for smarter merging
        modified_content = suggested_code
        
        return self.create_diff(file_path, original_content, modified_content, language)
    
    def cleanup_old_diffs(self, max_age_hours: int = 24):
        """Clean up diff files older than max_age_hours."""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file in self.diff_dir.glob("*"):
            if file.is_file() and file.name != ".gitignore":
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file.unlink()
                    except:
                        pass

