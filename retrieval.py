import os
import pickle
import json
import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest import CHROMA_DB_DIR, BM25_INDEX_DIR, get_embeddings

logger = logging.getLogger(__name__)


def retrieve_context(query: str, repo_id: str, top_k_initial: int = 10, top_k_final: int = 6) -> tuple[str, List[Dict[str, Any]]]:
    """Hybrid vector + BM25 retrieval with lightweight reciprocal-rank fusion."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=repo_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    bm25_path = os.path.join(BM25_INDEX_DIR, f"{repo_id}.pkl")
    if not os.path.exists(bm25_path):
        raise ValueError(f"No index found for {repo_id}. Please ingest the repo first.")

    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_index = bm25_data["index"]
    bm25_docs = bm25_data["documents"]

    vector_results = vectorstore.similarity_search(query, k=top_k_initial)

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_n_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k_initial]
    bm25_results = [bm25_docs[i] for i in top_n_indices]

    # Reciprocal Rank Fusion avoids loading a second transformer model just to rerank.
    fused: Dict[str, Dict[str, Any]] = {}
    for rank, doc in enumerate(vector_results):
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            fused.setdefault(chunk_id, {"doc": doc, "score": 0.0, "sources": []})
            fused[chunk_id]["score"] += 1.0 / (60 + rank + 1)
            fused[chunk_id]["sources"].append("Vector")

    for rank, doc in enumerate(bm25_results):
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            fused.setdefault(chunk_id, {"doc": doc, "score": 0.0, "sources": []})
            fused[chunk_id]["score"] += 1.0 / (60 + rank + 1)
            fused[chunk_id]["sources"].append("BM25")

    candidates = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    top_candidates = []
    for item in candidates[:top_k_final]:
        doc = item["doc"]
        doc.metadata["_source"] = " + ".join(item["sources"])
        top_candidates.append(doc)

    if not top_candidates:
        return "No relevant context found.", []

    # Build a lookup map for call-graph expansion.
    symbol_to_doc = {}
    for doc in bm25_docs:
        sym = doc.metadata.get("symbol_name")
        if sym and sym != "unknown":
            symbol_to_doc[sym] = doc

    final_chunks = []
    final_ids = set()
    for doc in top_candidates:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id in final_ids:
            continue
        final_chunks.append(doc)
        final_ids.add(chunk_id)

        try:
            callees = json.loads(doc.metadata.get("call_graph_callees", "[]"))
        except Exception:
            callees = []

        for callee in callees[:2]:
            callee_doc = symbol_to_doc.get(callee)
            if not callee_doc:
                continue
            callee_id = callee_doc.metadata.get("chunk_id")
            if callee_id in final_ids:
                continue
            clone = Document(
                page_content=callee_doc.page_content,
                metadata=callee_doc.metadata.copy()
            )
            clone.metadata["_source"] = f"Call-Graph (called by {doc.metadata.get('symbol_name')})"
            final_chunks.append(clone)
            final_ids.add(callee_id)

    context_parts = []
    sources = []
    for doc in final_chunks:
        meta = doc.metadata
        file_path = meta.get("file_path", "Unknown file")
        sym = meta.get("symbol_name", "Unknown symbol")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        source_type = meta.get("_source", "Unknown")

        context_parts.append(
            f"--- File: {file_path} | Symbol: {sym} | Lines: {start}-{end} ---\n"
            f"{doc.page_content}"
        )
        sources.append({
            "file": file_path,
            "symbol": sym,
            "lines": f"{start}-{end}",
            "source": source_type
        })

    return "\n\n".join(context_parts), sources
