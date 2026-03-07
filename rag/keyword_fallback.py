def keyword_fallback(col, keywords: list[str], limit: int = 8):
    data = col.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    keys = [k.lower() for k in keywords if k]
    hits = []

    for d, m in zip(docs, metas):
        if not d:
            continue
        dl = d.lower()
        if any(k in dl for k in keys):
            hits.append((d, m))
            if len(hits) >= limit:
                break

    return [h[0] for h in hits], [h[1] for h in hits]