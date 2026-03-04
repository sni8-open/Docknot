import json
import requests
from typing import Iterable
from config import OLLAMA_BASE, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL

def ollama_embed(texts: list[str], model: str = OLLAMA_EMBED_MODEL) -> list[list[float]]:
    base = OLLAMA_BASE.rstrip("/").replace("/api", "")
    embs = []
    for t in texts:
        r = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=120
        )
        r.raise_for_status()
        embs.append(r.json()["embedding"])

    print("OLLAMA_BASE (embed):", OLLAMA_BASE)
    print("POST:", f"{OLLAMA_BASE.rstrip('/')}/api/embeddings")
    return embs

def _messages_to_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            parts.append(f"SYSTEM:\n{content}\n")
        elif role == "user":
            parts.append(f"USER:\n{content}\n")
        elif role == "assistant":
            parts.append(f"ASSISTANT:\n{content}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)

def ollama_chat_stream(messages, model=OLLAMA_LLM_MODEL):

    prompt = "\n".join([m["content"] for m in messages])

    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True,
        timeout=300
    )

    r.raise_for_status()

    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "response" in data:
            yield data["response"]
    
    base = OLLAMA_BASE.rstrip("/")
    print("OLLAMA_BASE (chat):", base)
    print("TRY CHAT:", f"{base}/api/chat")
    print("TRY GENERATE:", f"{base}/api/generate")