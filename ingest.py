import gc
import json
import logging
import os
import pickle
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from chunking import get_ast_chunks
from callgraph import build_call_graph

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', '.md', '.json', '.yaml', '.yml'}
SKIP_DIRS = {'node_modules', '.git', 'venv', '__pycache__', 'dist', 'build'}
MAX_FILE_SIZE = 500 * 1024
MAX_CHUNKS = 2500
EMBED_BATCH_SIZE = 32
CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_DIR = "./bm25_indices"

_embeddings = None


def get_embeddings():
    """Load the embedding model once per process instead of on every request."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def clone_repo(repo_url: str) -> str:
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
    parts = [p for p in repo_url.split("/") if p]
    if len(parts) >= 2:
        name = f"{parts[-2]}_{parts[-1]}"
    else:
        name = "unknown_repo"
    name = name.replace(".git", "")
    return "".join(c for c in name if c.isalnum() or c in ("_", "-"))


def walk_and_chunk(repo_path: str) -> List[Dict[str, Any]]:
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)

            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue

            all_chunks.extend(get_ast_chunks(rel_path, content))

            if len(all_chunks) >= MAX_CHUNKS:
                logger.warning("Chunk limit reached; limiting repository to %s chunks.", MAX_CHUNKS)
                return all_chunks[:MAX_CHUNKS]

    return all_chunks


def load_or_ingest_repo(repo_url: str) -> Tuple[str, bool]:
    repo_id = get_repo_name_from_url(repo_url)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)

    cache_file = os.path.join(CHROMA_DB_DIR, f"{repo_id}_cache.json")
    repo_path = clone_repo(repo_url)
    commit_hash = get_latest_commit(repo_path)

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_info = json.load(f)
            if cache_info.get("commit") == commit_hash:
                logger.info(f"Repo {repo_id} is already cached. Skipping ingestion.")
                return repo_id, False

        logger.info(f"Processing repository: {repo_id}")
        chunks = walk_and_chunk(repo_path)

        if not chunks:
            raise ValueError("No valid text/code files found in the repository.")

        call_graph = build_call_graph(chunks)
        documents = []

        for i, chunk in enumerate(chunks):
            caller = chunk["metadata"].get("symbol_name")
            callees = call_graph.get(caller, [])
            metadata = chunk["metadata"].copy()
            metadata.pop("callees", None)
            metadata["call_graph_callees"] = json.dumps(callees)
            metadata["chunk_id"] = f"{repo_id}_{i}"
            documents.append(Document(page_content=chunk["text"], metadata=metadata))

        # Build BM25 before loading the embedding model, then free its temporary
        # token corpus and index so peak RAM stays lower during vector indexing.
        logger.info("Building BM25 index...")
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_data = {"index": bm25, "documents": documents}
        bm25_path = os.path.join(BM25_INDEX_DIR, f"{repo_id}.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        del bm25_data, bm25, tokenized_corpus
        gc.collect()

        logger.info("Embedding %s chunks in batches...", len(documents))
        vectorstore = Chroma(
            collection_name=repo_id,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DB_DIR
        )

        for start in range(0, len(documents), EMBED_BATCH_SIZE):
            batch = documents[start:start + EMBED_BATCH_SIZE]
            vectorstore.add_documents(batch)
            del batch
            if start % (EMBED_BATCH_SIZE * 10) == 0:
                gc.collect()

        with open(cache_file, 'w') as f:
            json.dump({"commit": commit_hash}, f)

        logger.info(f"Ingestion complete for {repo_id}")
        return repo_id, True
    finally:
        del repo_path
        shutil.rmtree(repo_path, ignore_errors=True)
        gc.collect()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        load_or_ingest_repo(sys.argv[1])
