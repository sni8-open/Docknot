import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from config import CHROMA_DIR

def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def collection_name_for_group(group_id: int) -> str:
    return f"group_{group_id}"

def get_collection(group_id: int):
    client = get_client()
    return client.get_or_create_collection(name=collection_name_for_group(group_id))