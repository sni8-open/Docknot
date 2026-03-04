from rag.chroma_store import get_collection

def delete_doc_vectors(group_id: int, document_id: int) -> int:
    """
    Deletes all chunk vectors for this document_id by scanning ids.
    For class projects, this is reliable and simple.
    """
    col = get_collection(group_id)
    data = col.get(include=["ids"])
    ids = data.get("ids", [])
    prefix = f"doc_{document_id}::chunk::"
    to_delete = [i for i in ids if i.startswith(prefix)]

    if to_delete:
        col.delete(ids=to_delete)

    return len(to_delete)