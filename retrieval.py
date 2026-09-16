import os
import pickle
import json
from typing import List, Dict, Any

from langchain_core.documents import Document

from ingest import BM25_INDEX_DIR


def retrieve_context(
    query: str,
    repo_id: str,
    top_k_initial: int = 12,
    top_k_final: int = 6
) -> tuple[str, List[Dict[str, Any]]]:
    """Low-memory BM25 retrieval with call-graph expansion."""
    bm25_path = os.path.join(BM25_INDEX_DIR, f"{repo_id}.pkl")
    if not os.path.exists(bm25_path):
        raise ValueError(f"No index found for {repo_id}. Please ingest the repo first.")

    with open(bm25_path, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_index = bm25_data["index"]
    bm25_docs = bm25_data["documents"]

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # BM25 remains the main ranking signal. A small metadata boost helps code
    # queries that explicitly mention a filename or symbol without needing an
    # embedding model.
    query_terms = {term.strip(".,:;()[]{}\"'`/") for term in tokenized_query if term}
    ranked_indices = list(range(len(scores)))

    def rank_score(i: int) -> float:
        doc = bm25_docs[i]
        metadata = doc.metadata
        path = str(metadata.get("file_path", "")).lower()
        symbol = str(metadata.get("symbol_name", "")).lower()
        boost = 0.0
        for term in query_terms:
            if term and term in path:
                boost += 0.75
            if term and term == symbol:
                boost += 1.5
        return float(scores[i]) + boost

    ranked_indices.sort(key=rank_score, reverse=True)
    top_indices = ranked_indices[:top_k_initial]
    top_candidates = []

    for i in top_indices:
        doc = bm25_docs[i]
        doc.metadata["_source"] = "BM25"
        top_candidates.append(doc)

    if not top_candidates:
        return "No relevant context found.", []

    symbol_to_doc = {}
    for doc in bm25_docs:
        sym = doc.metadata.get("symbol_name")
        if sym and sym != "unknown":
            symbol_to_doc[sym] = doc

    final_chunks = []
    final_ids = set()

    for doc in top_candidates[:top_k_final]:
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
            clone.metadata["_source"] = (
                f"Call-Graph (called by {doc.metadata.get('symbol_name')})"
            )
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
