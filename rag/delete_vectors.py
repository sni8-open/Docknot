from rag.qdrant_store import get_client, collection_name_for_group
from qdrant_client.models import Filter, FieldCondition, MatchValue


def delete_doc_vectors(group_id: int, document_id: int) -> int:
    client = get_client()
    collection_name = collection_name_for_group(group_id)

    # If collection does not exist, nothing to delete
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return 0

    flt = Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id)
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=flt,
        with_payload=False,
        limit=10000
    )

    point_ids = [p.id for p in points]

    if point_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )

    return len(point_ids)