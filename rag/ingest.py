from pathlib import Path
from pypdf import PdfReader
from rag.text_utils import clean_text, chunk_text
from rag.chroma_store import get_collection

def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n".join(pages))

def ingest_pdf_to_group(group_id: int, document_id: int, original_filename: str, pdf_path: Path, *, embed_fn):
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing file: {pdf_path}")

    text = extract_pdf_text(pdf_path)
    if not text:
        raise ValueError("Could not extract any text from this PDF (maybe scanned image-only PDF).")

    chunks = chunk_text(text, chunk_size=900, overlap=120)
    if not chunks:
        raise ValueError("No chunks created from PDF text.")

    embeddings = embed_fn(chunks)

    col = get_collection(group_id)
    ids = [f"doc_{document_id}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": original_filename, "document_id": document_id, "chunk_index": i}
        for i in range(len(chunks))
    ]

    col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)