from typing import Iterable, Tuple

from rag.ollama_client import ollama_embed, ollama_chat_stream
from rag.qdrant_store import get_client, collection_name_for_group

from rag.embedder import embed_query
from rag.reranker import rerank

def retrieve(group_id: int, question: str, k: int = 40):
    client = get_client()
    collection_name = collection_name_for_group(group_id)

    q_emb = embed_query(question)

    results = client.search(
        collection_name=collection_name,
        query_vector=q_emb,
        limit=k,
        with_payload=True
    )

    docs = []
    metas = []

    for r in results:
        payload = r.payload or {}
        docs.append(payload.get("text", ""))
        metas.append({
            "source": payload.get("source", ""),
            "document_id": payload.get("document_id", -1),
            "chunk_index": payload.get("chunk_index", -1),
            "page_start": payload.get("page_start", -1),
            "page_end": payload.get("page_end", -1),
            "section_title": payload.get("section_title", ""),
            "char_length": payload.get("char_length", 0),
            "preview": payload.get("preview", ""),
            "score": r.score,
        })

    docs, metas = rerank(question, docs, metas, top_k=8)

    return docs, metas


def build_messages(question: str, retrieved_docs: list[str], metas: list[dict], history: list[dict]) -> list[dict]:
    blocks = []
    for d, m in zip(retrieved_docs, metas):
        blocks.append(
            f"""
DOCUMENT CHUNK
Source: {m.get('source')}
Chunk: {m.get('chunk_index')}
Pages: {m.get('page_start')} - {m.get('page_end')}
Section: {m.get('section_title')}
Score: {m.get('score')}

{d}
"""
        )

    context = "\n\n---\n\n".join(blocks) if blocks else ""

    system = {
        "role": "system",
        "content": """
You are a document-grounded assistant.

Answer using ONLY the provided document context.

Rules:
- Do NOT use outside knowledge.
- Do NOT guess.
- If the answer is not explicitly supported by the context, reply exactly:

❌ The answer was not found in the uploaded documents.

Return the answer in markdown.
"""
    }

    msgs = [system]
    msgs.extend([m for m in history[-6:] if m.get("role") == "user"])

    msgs.append({
        "role": "user",
        "content": f"""
Use ONLY the document chunks below to answer.

## Document Context
{context}

## Question
{question}

## Important
- Stick strictly to the document.
- Do not use outside knowledge.
- Do not guess.
- Return the answer in markdown.
"""
    })

    return msgs


def stream_answer_with_citations(
    group_id: int,
    question: str,
    history: list[dict],
    k: int = 20
) -> Tuple[Iterable[str], list[dict], list[str]]:
    docs, metas = retrieve(group_id, question, k=k)

    if not docs or not metas:
        def fallback():
            yield "❌ The answer was not found in the uploaded documents."
        return fallback(), [], []

    messages = build_messages(question, docs, metas, history)
    token_stream = ollama_chat_stream(messages)
    return token_stream, metas, docs