import os
import logging
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from callgraph import extract_called_identifiers

logger = logging.getLogger(__name__)

# Map file extensions to tree-sitter language names
EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust"
}

def get_ast_chunks(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Parses the file using LangChain's RecursiveCharacterTextSplitter (Fallback mode).
    (Tree-sitter disabled due to Python 3.14 compatibility)
    """
    return get_fallback_chunks(file_path, content)

def _traverse_and_chunk(node, content: str, file_path: str, chunks: List[Dict[str, Any]], parent_name: Optional[str] = None):
    """
    Heuristically find functions, classes, and methods by inspecting node types.
    """
    node_type = node.type.lower()
    is_target_node = False
    symbol_name = "unknown"
    chunk_type = "unknown"
    
    if "function" in node_type or "method" in node_type or node_type == "arrow_function":
        is_target_node = True
        chunk_type = "function"
    elif "class" in node_type or "struct" in node_type or node_type == "impl_item":
        is_target_node = True
        chunk_type = "class"
        
    if is_target_node:
        # Try to find an identifier child for the name
        for child in node.children:
            if "identifier" in child.type.lower() or child.type == "name":
                try:
                    symbol_name = content[child.start_byte:child.end_byte]
                    break
                except Exception:
                    pass
                    
        if parent_name and symbol_name != "unknown":
            symbol_name = f"{parent_name}.{symbol_name}"
            
        try:
            chunk_text = content[node.start_byte:node.end_byte]
            # tree-sitter lines are 0-indexed, we'll store as 1-indexed for display
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            chunk_metadata = {
                "file_path": file_path,
                "chunk_type": chunk_type,
                "symbol_name": symbol_name,
                "start_line": start_line,
                "end_line": end_line,
                "callees": extract_called_identifiers(node, content)
            }
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        except Exception:
            pass
            
    # For classes, we still want to extract their methods as separate chunks for granularity
    # but we will also have the class chunk. If the class is huge, maybe we just want methods?
    # To keep it simple and high recall, we extract both, but maybe that causes duplication.
    # Let's only recurse if it's a class to get its methods.
    if is_target_node and chunk_type == "class":
        for child in node.children:
            _traverse_and_chunk(child, content, file_path, chunks, parent_name=symbol_name)
    elif not is_target_node:
        # Continue traversing
        for child in node.children:
            _traverse_and_chunk(child, content, file_path, chunks, parent_name)


def get_fallback_chunks(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Fallback chunking using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.create_documents([content])
    
    chunks = []
    # Try to approximate line numbers based on character offsets
    for i, doc in enumerate(docs):
        # We don't have exact line numbers easily without re-calculating, 
        # so we'll just put placeholder or calculate it
        start_idx = content.find(doc.page_content)
        if start_idx != -1:
            start_line = content[:start_idx].count('\n') + 1
            end_line = start_line + doc.page_content.count('\n')
        else:
            start_line = 0
            end_line = 0
            
        chunks.append({
            "text": doc.page_content,
            "metadata": {
                "file_path": file_path,
                "chunk_type": "text",
                "symbol_name": f"chunk_{i}",
                "start_line": start_line,
                "end_line": end_line
            }
        })
    return chunks
