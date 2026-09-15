import streamlit as st
import os
from dotenv import load_dotenv

# Load env variables before importing other local modules that might need them
load_dotenv()

from ingest import load_or_ingest_repo, get_repo_name_from_url
from retrieval import retrieve_context
from chat import generate_response_stream

st.set_page_config(page_title="RAG Codebase Chat", page_icon="🤖", layout="wide")

st.title("GitHub Repository RAG Chatbot")

# Initialize session state
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Repository Setup")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
    
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("Load Repo")
    with col2:
        clear_btn = st.button("Clear Cache & Reload")
        
    if load_btn and repo_url:
        with st.spinner("Cloning, chunking, and indexing repository..."):
            try:
                repo_id, is_new = load_or_ingest_repo(repo_url)
                st.session_state.repo_id = repo_id
                # Reset chat on new repo
                st.session_state.messages = []
                if is_new:
                    st.success(f"Successfully ingested {repo_id}!")
                else:
                    st.success(f"Loaded cached index for {repo_id}!")
            except Exception as e:
                st.error(f"Error loading repository: {e}")
                
    if clear_btn and repo_url:
        repo_id = get_repo_name_from_url(repo_url)
        # We can implement a proper clean up here, for now just remove cache file to force re-ingest
        cache_file = os.path.join("./chroma_db", f"{repo_id}_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        st.warning(f"Cleared cache for {repo_id}. Click 'Load Repo' to re-ingest.")

# Chat Interface
if st.session_state.repo_id:
    st.write(f"**Chatting with:** `{st.session_state.repo_id}`")
    
    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s['file']}** (`{s['symbol']}`, lines {s['lines']}) [Source: {s['source']}]")

    # Chat Input
    if prompt := st.chat_input("Ask a question about the code..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving context..."):
                    context, sources = retrieve_context(prompt, st.session_state.repo_id)
                
                with st.spinner("Generating response..."):
                    response_stream = generate_response_stream(prompt, context, st.session_state.messages[:-1])
                    
                    full_response = st.write_stream(response_stream)
                    
                    if sources:
                        with st.expander("Sources used"):
                            for s in sources:
                                st.markdown(f"- **{s['file']}** (`{s['symbol']}`, lines {s['lines']}) [Source: {s['source']}]")
                                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please load a repository from the sidebar to begin.")
