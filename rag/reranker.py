from functools import lru_cache
from sentence_transformers import CrossEncoder
from config import RERANK_MODEL_NAME, RERANK_DEVICE


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL_NAME, device=RERANK_DEVICE)


def rerank(query: str, docs: list[str], metas: list[dict], top_k: int = 8):
    if not docs:
        return [], []

    model = get_reranker()

    pairs = [[query, d] for d in docs]

    scores = model.predict(pairs)

    ranked = list(zip(scores, docs, metas))
    ranked.sort(key=lambda x: x[0], reverse=True)

    ranked = ranked[:top_k]

    docs_sorted = [r[1] for r in ranked]
    metas_sorted = [r[2] for r in ranked]

    return docs_sorted, metas_sorted