import sqlite3
from config import DB_PATH

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(group_id) REFERENCES groups(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        role TEXT NOT NULL,     -- 'user' | 'assistant'
        content TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(group_id) REFERENCES groups(id)
    )
    """)

    conn.commit()
    conn.close()

def create_group(name: str):
    name = (name or "").strip()
    if not name:
        return
    conn = get_conn()
    conn.execute("INSERT OR IGNORE INTO groups(name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def list_groups():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM groups ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows

def get_group(group_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM groups WHERE id=?", (group_id,)).fetchone()
    conn.close()
    return row

def add_document_returning_id(group_id: int, filename: str, filepath: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents(group_id, filename, filepath) VALUES (?,?,?)",
        (group_id, filename, filepath)
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id

def list_documents(group_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM documents WHERE group_id=? ORDER BY uploaded_at DESC",
        (group_id,)
    ).fetchall()
    conn.close()
    return rows

def get_document(doc_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    conn.close()
    return row

def delete_document_row(doc_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()

def add_message(group_id: int, role: str, content: str):
    role = (role or "").strip()
    content = (content or "").strip()
    if role not in ("user", "assistant") or not content:
        return
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages(group_id, role, content) VALUES (?,?,?)",
        (group_id, role, content)
    )
    conn.commit()
    conn.close()

def get_messages(group_id: int, limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE group_id=? ORDER BY id DESC LIMIT ?",
        (group_id, limit)
    ).fetchall()
    conn.close()
    return list(reversed(rows))