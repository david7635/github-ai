import os
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def get_llm(model_name: str):
    """Initializes the selected Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or invalid. "
            "Please check your .env file."
        )

    return ChatGroq(
        model_name=model_name,
        temperature=0.0,
        groq_api_key=api_key,
        streaming=True
    )


def generate_response_stream(
    query: str,
    context: str,
    chat_history: List[dict],
    model_name: str
):
    """Generates a streaming response from the selected LLM."""
    llm = get_llm(model_name)

    system_prompt = f"""You are a code assistant answering questions about the provided repository context.
Always cite file paths and symbol names for claims you make.
If the context doesn't contain the answer, say so. Do not hallucinate code that is not provided.

=== CONTEXT ===
{context}
=== END CONTEXT ===
"""

    messages = [SystemMessage(content=system_prompt)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=query))

    return llm.stream(messages)
