import os
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

def get_llm():
    """Initializes the Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set or invalid. Please check your .env file.")
        
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0, # low temp for more factual Q&A
        groq_api_key=api_key,
        streaming=True
    )

def generate_response_stream(query: str, context: str, chat_history: List[dict]):
    """
    Generates a streaming response from the LLM given the context and query.
    """
    llm = get_llm()
    
    system_prompt = f"""You are a code assistant answering questions about the provided repository context. 
Always cite file paths and symbol names for claims you make. 
If the context doesn't contain the answer, say so. Do not hallucinate code that is not provided.

=== CONTEXT ===
{context}
=== END CONTEXT ===
"""

    messages = [SystemMessage(content=system_prompt)]
    
    # Add chat history (simplified for now, alternating human/ai)
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # For this simple implementation, we just append to messages
            # langchain has AIMessage but we'll use HumanMessage as proxy or avoid importing if not needed.
            # Actually, let's just import AIMessage
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=msg["content"]))
            
    # Add current query
    messages.append(HumanMessage(content=query))
    
    # Stream the response
    return llm.stream(messages)
