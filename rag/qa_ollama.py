from typing import Iterable, Tuple
from rag.chroma_store import get_collection
from rag.ollama_client import ollama_embed, ollama_chat_stream

def retrieve(group_id: int, question: str, k: int = 5):
    col = get_collection(group_id)
    q_emb = ollama_embed([question])[0]
    res = col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return docs, metas

def build_messages(question: str, retrieved_docs: list[str], metas: list[dict], history: list[dict]) -> list[dict]:
    blocks = []
    for d, m in zip(retrieved_docs, metas):
        blocks.append(
            f"[Source: {m.get('source')} | Chunk: {m.get('chunk_index')}]\n{d}"
        )
    context = "\n\n---\n\n".join(blocks) if blocks else "No relevant context found in uploaded PDFs."

    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant for question answering over PDFs. "
            "Use the provided context to answer. If the context is insufficient, say so."
        )
    }

    msgs = [system]
    msgs.extend(history[-8:])  # conversation memory
    msgs.append({"role": "user", "content": f"PDF Context:\n{context}\n\nQuestion: {question}"})
    return msgs

def stream_answer_with_citations(
    group_id: int,
    question: str,
    history: list[dict],
    k: int = 5
) -> Tuple[Iterable[str], list[dict]]:
    docs, metas = retrieve(group_id, question, k=k)
    messages = build_messages(question, docs, metas, history)
    token_stream = ollama_chat_stream(messages)
    return token_stream, metas