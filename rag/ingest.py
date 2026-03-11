from pathlib import Path
import fitz

from qdrant_client.models import PointStruct

from rag.text_utils import clean_text, chunk_pages
from rag.qdrant_store import get_client, ensure_collection
from rag.embedder import embed_documents


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        text = clean_text(text)
        pages.append({
            "page_num": page_num,
            "text": text
        })

    return pages


def ingest_pdf_to_group(group_id: int, document_id: int, original_filename: str, pdf_path: Path, *, embed_documents):
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing file: {pdf_path}")

    page_entries = extract_pdf_pages(pdf_path)
    if not page_entries:
        raise ValueError("Could not extract any text from this PDF.")

    chunks = chunk_pages(page_entries, chunk_size=1800, overlap=500)
    if not chunks:
        raise ValueError("No chunks created from PDF text.")

    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_documents(chunk_texts)
    vector_size = len(embeddings[0])

    client = get_client()
    collection_name = ensure_collection(group_id, vector_size)

    points = []
    for c, emb in zip(chunks, embeddings):
        point_id = int(document_id * 1_000_000 + c["chunk_index"])
        payload = {
            "source": original_filename,
            "document_id": document_id,
            "chunk_index": c["chunk_index"],
            "page_start": c["page_start"] if c["page_start"] is not None else -1,
            "page_end": c["page_end"] if c["page_end"] is not None else -1,
            "section_title": c["section_title"] or "",
            "char_length": c["char_length"],
            "preview": c["preview"],
            "text": c["text"],
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload=payload
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return len(chunks)