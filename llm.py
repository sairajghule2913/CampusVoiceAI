import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved FAQ context.
    """

    if not context:
        return (
            "I couldn't find information about that in the "
            "available college knowledge base."
        )

    prompt = f"""
You are Campus Voice AI, a helpful college FAQ assistant.

Answer the user's question using ONLY the information
provided in the FAQ context below.

IMPORTANT RULES:
- Do not invent or guess information.
- Do not add college policies that are not in the context.
- If the context does not answer the question, clearly say
  that the information was not found in the knowledge base.
- Keep the answer concise and conversational because it will
  eventually be spoken aloud.
- Do not mention "context", "retrieval", or "RAG" to the user.

FAQ CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()