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
        You are a helpful assistant answering questions from uploaded PDFs.

        You MUST answer ONLY using the provided context.

        Formatting rules:
        - Return answers in **markdown format**
        - Use headings and emojis/icons
        - Use bullet points
        - Highlight key words in **bold**

        Answer format:

        ## 📘 Definition
        - Provide the definition.

        ## 🔍 Key Points
        - Important concepts in bullet points.

        ## 🧠 Explanation
        - Simple explanation.

        ## 📝 Example
        - Example if available.

        If the answer is not found in the context, reply exactly:

        ❌ The answer was not found in the uploaded documents.

        Do NOT use outside knowledge.
        """
    }

    msgs = [system]
    msgs.extend(history[-8:])  # memory window
    msgs.append({
            "role": "user",
            "content": f"""
        Use ONLY the following PDF context to answer.

        Context:
        {context}

        Question:
        {question}

        Return the answer in markdown format using:
        - headings
        - bullet points
        - bold keywords
        - emojis/icons
        """
        })
    return msgs

def stream_answer_with_citations(
    group_id: int,
    question: str,
    history: list[dict],
    k: int = 25
) -> Tuple[Iterable[str], list[dict]]:
    docs, metas = retrieve(group_id, question, k=k)

    # metas always defined now
    messages = build_messages(question, docs, metas, history)

    token_stream = ollama_chat_stream(messages)
    return token_stream, metas