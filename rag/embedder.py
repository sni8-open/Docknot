from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME, EMBED_DEVICE


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = get_embedder()
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]
    return vec.tolist()