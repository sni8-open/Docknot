from rag.chroma_store import get_collection

def delete_doc_vectors(group_id: int, document_id: int) -> int:
    """
    Delete all chunk vectors for a document by matching chunk id prefix.
    Chunk ids look like: doc_<document_id>::chunk::<i>
    """
    col = get_collection(group_id)

    # ids are returned by default
    data = col.get()
    ids = data.get("ids", [])

    prefix = f"doc_{document_id}::chunk::"
    to_delete = [i for i in ids if i.startswith(prefix)]

    if to_delete:
        col.delete(ids=to_delete)

    return len(to_delete)