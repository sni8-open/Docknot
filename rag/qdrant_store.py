from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import QDRANT_PATH, QDRANT_COLLECTION_PREFIX


@lru_cache(maxsize=1)
def get_client():
    return QdrantClient(path=str(QDRANT_PATH))


def collection_name_for_group(group_id: int):
    return f"{QDRANT_COLLECTION_PREFIX}{group_id}"


def ensure_collection(group_id: int, vector_size: int):
    client = get_client()
    name = collection_name_for_group(group_id)

    existing = [c.name for c in client.get_collections().collections]

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    return name