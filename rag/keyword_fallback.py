def keyword_fallback(col, keywords: list[str], limit: int = 10):
    data = col.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    keywords = [k.lower().strip() for k in keywords if k.strip()]
    hits = []

    for d, m in zip(docs, metas):
        if not d:
            continue

        dl = d.lower()
        score = sum(1 for k in keywords if k in dl)

        if score > 0:
            hits.append((score, d, m))

    hits.sort(key=lambda x: x[0], reverse=True)
    hits = hits[:limit]

    return [h[1] for h in hits], [h[2] for h in hits]


