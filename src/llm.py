import ollama 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))        # src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from embeddings import retrieval

def generate_answer(question : str) -> str:
    context = retrieval(question)
    context_str = "\n\n".join(context)

    prompt = f"""Você é um tutor de teoria musical. Use apenas o contexto abaixo para responder.
    Se a resposta não estiver no contexto, diga que não encontrou nos materiais.

    Contexto:
    {context_str}

    Pergunta: {question}
    Resposta:"""

    resposta = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return resposta["message"]["content"]