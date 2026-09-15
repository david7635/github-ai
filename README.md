# RAG Codebase Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot designed to help you converse with and understand any GitHub repository.

## Features

- **AST-Aware Chunking**: Uses `tree-sitter` to parse code into logical chunks (functions, classes) instead of arbitrary text splits.
- **Hybrid Retrieval**: Combines semantic vector search (ChromaDB + sentence-transformers) with keyword search (BM25) for high-recall candidate selection.
- **Cross-Encoder Re-ranking**: Uses a cross-encoder to accurately score and re-rank the retrieved candidates based on the query.
- **Call Graph Context**: Best-effort expansion of the context to include called functions, providing the LLM with deeper insights into code execution.
- **Streamlit UI**: Simple interface to load a repo and chat with the Groq-powered LLM.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Architecture

- `app.py`: Streamlit frontend.
- `chat.py`: LangChain/Groq integration.
- `ingest.py`: Repository cloning, traversing, and indexing orchestration.
- `chunking.py`: AST parsing logic.
- `callgraph.py`: Call graph extraction logic.
- `retrieval.py`: Hybrid search, re-ranking, and context expansion.
- `eval.py`: CLI tool for evaluating retrieval precision.
