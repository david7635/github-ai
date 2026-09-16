# RAG Codebase Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to help you converse with and understand any GitHub repository.

## Live Demo

[Open the deployed Streamlit app](https://gappai.streamlit.app/)

## Features

- **Code-Aware Chunking**: Extracts logical code structures such as functions, classes, and methods when available, with a text-based fallback.
- **Hybrid Retrieval**: Combines semantic vector search (ChromaDB + sentence-transformers) with keyword search (BM25) for high-recall candidate selection.
- **Cross-Encoder Re-ranking**: Uses a cross-encoder to accurately score and re-rank retrieved candidates based on the query.
- **Call Graph Context**: Best-effort expansion of context to include called functions, providing the LLM with deeper insights into code execution.
- **Multi-Model Groq Support**: Select the generation model directly from the Streamlit sidebar.
- **Streaming Responses**: Streams LLM responses into the chat interface for a responsive experience.
- **Source Citations**: Displays the repository files, symbols, and line ranges used to generate an answer.
- **Repository Caching**: Reuses existing indexes when a repository has already been ingested.
- **Streamlit UI**: Simple interface for loading a GitHub repository and chatting with its codebase.

## Supported LLMs

The Streamlit UI currently provides these Groq models:

- **GPT-OSS 120B** — `openai/gpt-oss-120b`
- **GPT-OSS 20B** — `openai/gpt-oss-20b`
- **Qwen 3 32B** — `qwen/qwen3-32b`

The selected model is passed from `app.py` to `chat.py` at runtime, so the generation model is not hard-coded into the chat integration.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```

   Windows PowerShell:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

5. Enter a GitHub repository URL in the sidebar, select an AI model, and click **Load Repo**.

## Architecture

- `app.py`: Streamlit frontend, repository controls, model selection, chat history, and response display.
- `chat.py`: LangChain/Groq integration and streaming LLM responses.
- `ingest.py`: Repository cloning, traversal, chunking, and indexing orchestration.
- `chunking.py`: Code-aware chunking logic with a text-based fallback.
- `callgraph.py`: Call graph extraction and context expansion logic.
- `retrieval.py`: Hybrid search, cross-encoder re-ranking, and context retrieval.
- `eval.py`: CLI tool for evaluating retrieval precision.

## Workflow

```text
GitHub Repository
       |
       v
Repository Ingestion
       |
       v
Code-Aware Chunking
       |
       +------------------+
       |                  |
       v                  v
 ChromaDB              BM25
 Vector Search       Keyword Search
       |                  |
       +--------+---------+
                |
                v
       Candidate Retrieval
                |
                v
       Cross-Encoder Ranking
                |
                v
         Call Graph Context
                |
                v
        Selected Groq Model
                |
                v
       Streaming Response
                |
                v
        Answer + Sources
```

## References

- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.

## Models

- `sentence-transformers/all-MiniLM-L6-v2` — Apache-2.0
- `cross-encoder/ms-marco-MiniLM-L6-v2` — Apache-2.0
- `openai/gpt-oss-120b` — Apache-2.0
- `openai/gpt-oss-20b` — Apache-2.0
- `Qwen/Qwen3-32B` — Apache-2.0

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
