import os
import pickle
import json
import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ingest import CHROMA_DB_DIR, BM25_INDEX_DIR

logger = logging.getLogger(__name__)

# Initialize cross-encoder lazily to save startup time
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def retrieve_context(query: str, repo_id: str, top_k_initial: int = 15, top_k_final: int = 6) -> tuple[str, List[Dict[str, Any]]]:
    """
    Performs hybrid retrieval, cross-encoder re-ranking, and call graph expansion.
    Returns a formatted context string and the list of source chunks.
    """
    # 1. Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=repo_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # 2. Load BM25 Index
    bm25_path = os.path.join(BM25_INDEX_DIR, f"{repo_id}.pkl")
    if not os.path.exists(bm25_path):
        raise ValueError(f"No index found for {repo_id}. Please ingest the repo first.")
        
    with open(bm25_path, 'rb') as f:
        bm25_data = pickle.load(f)
        
    bm25_index = bm25_data["index"]
    bm25_docs = bm25_data["documents"]
    
    # 3. Retrieve Candidates
    logger.info("Retrieving candidates...")
    
    # Vector Search
    vector_results = vectorstore.similarity_search(query, k=top_k_initial)
    
    # BM25 Search
    tokenized_query = query.split(" ")
    # Get top N indices
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # Sort indices by score descending and get top k
    top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_initial]
    bm25_results = [bm25_docs[i] for i in top_n_indices]
    
    # 4. Merge and Deduplicate
    merged_candidates: Dict[str, Document] = {}
    
    for doc in vector_results:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            # Tag the source for explanation
            doc.metadata["_source"] = "Vector"
            merged_candidates[chunk_id] = doc
            
    for doc in bm25_results:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            if chunk_id in merged_candidates:
                merged_candidates[chunk_id].metadata["_source"] += " + BM25"
            else:
                doc.metadata["_source"] = "BM25"
                merged_candidates[chunk_id] = doc
                
    candidates_list = list(merged_candidates.values())
    
    if not candidates_list:
        return "No relevant context found.", []
        
    # 5. Cross-Encoder Re-ranking
    logger.info(f"Re-ranking {len(candidates_list)} candidates...")
    ce = get_cross_encoder()
    
    pairs = [[query, doc.page_content] for doc in candidates_list]
    scores = ce.predict(pairs)
    
    # Attach scores and sort
    scored_candidates = list(zip(candidates_list, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    top_candidates = [doc for doc, score in scored_candidates[:top_k_final]]
    
    # 6. Call Graph Expansion
    final_chunks = []
    # Build a lookup map for symbol_name to doc for fast expansion
    symbol_to_doc = {}
    for doc in bm25_docs:
        sym = doc.metadata.get("symbol_name")
        if sym and sym != "unknown":
            symbol_to_doc[sym] = doc

    for doc in top_candidates:
        final_chunks.append(doc)
        
        callees_str = doc.metadata.get("call_graph_callees", "[]")
        try:
            callees = json.loads(callees_str)
        except Exception:
            callees = []
            
        added_callees = 0
        for callee in callees:
            if added_callees >= 2: # Limit to 1-2 callees per chunk
                break
                
            if callee in symbol_to_doc:
                callee_doc = symbol_to_doc[callee]
                # Check if we already have this callee in our final list
                if not any(c.metadata.get("chunk_id") == callee_doc.metadata.get("chunk_id") for c in final_chunks):
                    # Clone document to modify source tag without affecting the global one
                    callee_clone = Document(page_content=callee_doc.page_content, metadata=callee_doc.metadata.copy())
                    callee_clone.metadata["_source"] = f"Call-Graph (called by {doc.metadata.get('symbol_name')})"
                    final_chunks.append(callee_clone)
                    added_callees += 1

    # 7. Format Output
    context_str = ""
    sources = []
    
    for doc in final_chunks:
        meta = doc.metadata
        file_path = meta.get("file_path", "Unknown file")
        sym = meta.get("symbol_name", "Unknown symbol")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        source_type = meta.get("_source", "Unknown")
        
        chunk_header = f"File: {file_path} | Symbol: {sym} | Lines: {start}-{end}"
        context_str += f"--- {chunk_header} ---\n{doc.page_content}\n\n"
        
        sources.append({
            "file": file_path,
            "symbol": sym,
            "lines": f"{start}-{end}",
            "source": source_type
        })
        
    return context_str, sources
