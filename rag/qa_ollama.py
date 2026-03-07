from typing import Iterable, Tuple
from rag.chroma_store import get_collection
from rag.ollama_client import ollama_embed, ollama_chat_stream
from rag.keyword_fallback import keyword_fallback

# def retrieve(group_id: int, question: str, k: int = 5):
#     """
#     Always returns (docs, metas) even if collection is empty.
#     """
#     col = get_collection(group_id)

#     # If collection has no items, return empty
#     try:
#         peek = col.get(limit=1, include=["ids"])
#         if not peek.get("ids"):
#             return [], []
#     except Exception:
#         # If peek fails for any reason, still fail gracefully
#         return [], []

#     q_emb = ollama_embed([question])[0]

#     print("RETRIEVED METAS:", metas[:3])
#     print("RETRIEVED DOC PREVIEW:", [d[:120] for d in docs[:3]])

#     res = col.query(
#         query_embeddings=[q_emb],
#         n_results=k,
#         include=["documents", "metadatas"]
#     )

#     docs = (res.get("documents") or [[]])[0] or []
#     metas = (res.get("metadatas") or [[]])[0] or []
#     return docs, metas

def retrieve(group_id: int, question: str, k: int = 25):
    col = get_collection(group_id)

    # Vector retrieval
    q_emb = ollama_embed([question])[0]
    res = col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []

    # If vector search didn't return good stuff, fallback to keyword scan
    if len(docs) < 3:
        # For your specific queries, these keywords are perfect
        keywords = ["sample space", "event", "events", "probability"]
        docs2, metas2 = keyword_fallback(col, keywords, limit=10)

        # Merge fallback results (avoid empty)
        if docs2:
            docs = docs2 + docs
            metas = metas2 + metas
    
    print("RETRIEVED SOURCES:", [(m.get("source"), m.get("chunk_index")) for m in metas[:5]], flush=True)
    print("DOC PREVIEW:", [d[:120] for d in docs[:2]], flush=True)

    return docs, metas

def build_messages(question: str, retrieved_docs: list[str], metas: list[dict], history: list[dict]) -> list[dict]:
    blocks = []
    for d, m in zip(retrieved_docs, metas):
        blocks.append(
            f"[Source: {m.get('source')} | Chunk: {m.get('chunk_index')}]\n{d}"
        )

    context = "\n\n---\n\n".join(blocks) if blocks else "No relevant context found in uploaded PDFs."

    # system = {
    #     "role": "system",
    #     "content": (
    #         "You are a helpful assistant for answering questions from PDFs. "
    #         "Use the provided context. If context is insufficient, say so clearly."
    #     )
    # }

    system = {
    "role": "system",
    "content": """
    You are a document-grounded question answering assistant.

    You must answer ONLY from the provided document context.

    ## Rules
    - Stick strictly to the document.
    - Do NOT use outside knowledge.
    - Do NOT guess.
    - Do NOT add information that is not explicitly supported by the context.
    - If the answer is missing, unclear, or not explicitly stated in the context, reply exactly:

    ❌ The answer was not found in the uploaded documents.

    ## Output format
    Return the answer in markdown.

    Use this structure when possible:

    ## 📘 Answer
    - Give the answer clearly.

    ## 🔍 Key Points
    - Use bullet points.
    - Keep them short and faithful to the document.

    ## 📝 Example
    - Include an example only if it is supported by the document.

    ## 📚 Source
    - State that the answer is based on the uploaded document context.

    Do not include any information that is not present in the context.
    """
    }

    msgs = [system]
    msgs.extend([m for m in history[-8:] if m["role"] == "user"])
    msgs.append({
    "role": "user",
    "content": f"""
    Answer the question using ONLY the document context below.

    ## Document Context
    {context}

    ## Question
    {question}

    ## Important
    - Stick strictly to the document.
    - Do not use outside knowledge.
    - Do not guess.
    - Return the answer in markdown format.
    - If the answer is not explicitly supported by the document, reply exactly:

    ❌ The answer was not found in the uploaded documents.
    """
    })
    return msgs

def stream_answer_with_citations(group_id: int, question: str, history: list[dict], k: int = 25):
    docs, metas = retrieve(group_id, question, k=k)

    if not docs or not metas:
        def fallback():
            yield "❌ The answer was not found in the uploaded documents."
        return fallback(), []

    messages = build_messages(question, docs, metas, history)
    token_stream = ollama_chat_stream(messages)
    return token_stream, metas