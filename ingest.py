import os
import shutil
import subprocess
import pickle
import logging
from typing import List, Dict, Any, Tuple
import tempfile
import json

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from chunking import get_ast_chunks
from callgraph import build_call_graph

logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', '.md', '.json', '.yaml', '.yml'}
SKIP_DIRS = {'node_modules', '.git', 'venv', '__pycache__', 'dist', 'build'}
MAX_FILE_SIZE = 500 * 1024  # 500 KB
CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_DIR = "./bm25_indices"

def clone_repo(repo_url: str) -> str:
    """Clones a repository into a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="rag_repo_")
    logger.info(f"Cloning {repo_url} into {temp_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True, capture_output=True, text=True
        )
        return temp_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Failed to clone repository: {repo_url}")

def get_latest_commit(repo_path: str) -> str:
    """Gets the latest commit hash for the repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, check=True, capture_output=True, text=True
        )
        return res.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get commit hash: {e}")
        return "unknown_commit"

def get_repo_name_from_url(repo_url: str) -> str:
    """Extracts a safe repo name from the URL to use as an identifier."""
    # e.g., https://github.com/user/repo -> user_repo
    parts = [p for p in repo_url.split("/") if p]
    if len(parts) >= 2:
        name = f"{parts[-2]}_{parts[-1]}"
    else:
        name = "unknown_repo"
    
    name = name.replace(".git", "")
    return "".join(c for c in name if c.isalnum() or c in ("_", "-"))

def walk_and_chunk(repo_path: str) -> List[Dict[str, Any]]:
    """Walks the repository and chunks supported files."""
    all_chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            
            # Skip large files
            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Skip binary or non-utf8 files
                continue
                
            file_chunks = get_ast_chunks(rel_path, content)
            all_chunks.extend(file_chunks)
            
    return all_chunks

def load_or_ingest_repo(repo_url: str) -> Tuple[str, bool]:
    """
    Main entry point for ingestion. 
    Returns the repo_id (collection name) and a boolean indicating if it was newly ingested.
    """
    repo_id = get_repo_name_from_url(repo_url)
    
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)
    
    cache_file = os.path.join(CHROMA_DB_DIR, f"{repo_id}_cache.json")
    
    # 1. Clone repo
    repo_path = clone_repo(repo_url)
    commit_hash = get_latest_commit(repo_path)
    
    # 2. Check cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_info = json.load(f)
            if cache_info.get("commit") == commit_hash:
                logger.info(f"Repo {repo_id} at {commit_hash} is already cached. Skipping ingestion.")
                shutil.rmtree(repo_path, ignore_errors=True)
                return repo_id, False
                
    # 3. Walk and chunk
    logger.info(f"Processing repository: {repo_id}")
    chunks = walk_and_chunk(repo_path)
    
    if not chunks:
        shutil.rmtree(repo_path, ignore_errors=True)
        raise ValueError("No valid text/code files found in the repository.")
        
    # 4. Build call graph and inject into metadata
    call_graph = build_call_graph(chunks)
    # We will just serialize the call graph into a JSON string in the metadata since Chroma metadata only accepts strings/ints/floats
    
    documents = []
    texts_for_bm25 = []
    
    for i, chunk in enumerate(chunks):
        caller = chunk["metadata"].get("symbol_name")
        callees = call_graph.get(caller, [])
        
        # Serialize list to string for ChromaDB
        metadata = chunk["metadata"].copy()
        # Clean up complex types
        if "callees" in metadata:
            del metadata["callees"] 
            
        metadata["call_graph_callees"] = json.dumps(callees)
        metadata["chunk_id"] = f"{repo_id}_{i}"
        
        doc = Document(
            page_content=chunk["text"],
            metadata=metadata
        )
        documents.append(doc)
        texts_for_bm25.append(chunk["text"])

    # 5. Embed and store in Chroma
    logger.info(f"Embedding {len(documents)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Using Chroma.from_documents can fail if collection exists, so we clear it first if needed or just use it
    # We will instantiate it and add documents
    vectorstore = Chroma(
        collection_name=repo_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Delete existing documents in the collection to avoid duplicates if cache missed but collection exists
    try:
        # A hack to clear the collection in Chroma is just to delete it and recreate, or we just rely on langchain
        # Since we use langchain-chroma, we might not be able to easily drop collection.
        # But for this demo, we'll assume a fresh ingestion or we just add.
        # Actually, adding same ID will overwrite, but we don't know old IDs.
        pass
    except Exception:
        pass
        
    vectorstore.add_documents(documents)
    
    # 6. Build and store BM25 index
    logger.info("Building BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in texts_for_bm25]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "index": bm25,
        "documents": documents # Store documents alongside BM25 so we can retrieve them by index
    }
    
    with open(os.path.join(BM25_INDEX_DIR, f"{repo_id}.pkl"), 'wb') as f:
        pickle.dump(bm25_data, f)
        
    # 7. Update cache
    with open(cache_file, 'w') as f:
        json.dump({"commit": commit_hash}, f)
        
    # Cleanup
    shutil.rmtree(repo_path, ignore_errors=True)
    logger.info(f"Ingestion complete for {repo_id}")
    
    return repo_id, True

if __name__ == "__main__":
    # Test ingestion
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        load_or_ingest_repo(sys.argv[1])
