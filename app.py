import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from dotenv import load_dotenv

from config import UPLOAD_DIR, FLASK_SECRET_KEY
from db.models import (
    init_db, create_group, list_groups, get_group,
    add_document_returning_id, list_documents, get_document, delete_document_row,
    add_message, get_messages
)
from rag.ingest import ingest_pdf_to_group
from rag.ollama_client import ollama_embed
from rag.delete_vectors import delete_doc_vectors
from rag.qa_ollama import stream_answer_with_citations

load_dotenv()

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# @app.before_first_request
# def _init():
#     init_db()

@app.before_request
def _init_once():
    if not getattr(app, "_db_inited", False):
        init_db()
        app._db_inited = True

@app.get("/")
def index():
    groups = list_groups()
    return render_template("index.html", groups=groups)

@app.post("/groups/create")
def groups_create():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Group name cannot be empty.", "danger")
        return redirect(url_for("index"))
    create_group(name)
    flash(f"Created group: {name}", "success")
    return redirect(url_for("index"))

@app.get("/groups/<int:group_id>")
def group_page(group_id: int):
    group = get_group(group_id)
    if not group:
        flash("Group not found.", "danger")
        return redirect(url_for("index"))

    docs = list_documents(group_id)
    messages = get_messages(group_id, limit=50)
    return render_template("group.html", group=group, docs=docs, messages=messages)

@app.post("/groups/<int:group_id>/upload")
def upload_pdf(group_id: int):
    group = get_group(group_id)
    if not group:
        flash("Group not found.", "danger")
        return redirect(url_for("index"))

    f = request.files.get("pdf")
    if not f or f.filename.strip() == "":
        flash("No file selected.", "danger")
        return redirect(url_for("group_page", group_id=group_id))

    if not f.filename.lower().endswith(".pdf"):
        flash("Only PDF files are allowed.", "danger")
        return redirect(url_for("group_page", group_id=group_id))

    save_path = UPLOAD_DIR / f"{group_id}__{f.filename}"
    f.save(save_path)

    try:
        doc_id = add_document_returning_id(group_id, f.filename, str(save_path))
        n_chunks = ingest_pdf_to_group(
            group_id=group_id,
            document_id=doc_id,
            original_filename=f.filename,
            pdf_path=save_path,
            embed_fn=ollama_embed
        )
        flash(f"Uploaded & indexed {f.filename} ({n_chunks} chunks).", "success")
    except Exception as e:
        flash(f"Upload/index failed: {e}", "danger")

    return redirect(url_for("group_page", group_id=group_id))

@app.post("/groups/<int:group_id>/docs/<int:doc_id>/delete")
def delete_doc(group_id: int, doc_id: int):
    doc = get_document(doc_id)
    if not doc or doc["group_id"] != group_id:
        flash("Document not found.", "danger")
        return redirect(url_for("group_page", group_id=group_id))

    try:
        deleted_chunks = delete_doc_vectors(group_id, doc_id)

        path = Path(doc["filepath"])
        if path.exists():
            path.unlink()

        delete_document_row(doc_id)
        flash(f"Deleted PDF and removed {deleted_chunks} vector chunks.", "success")
    except Exception as e:
        flash(f"Delete failed: {e}", "danger")

    return redirect(url_for("group_page", group_id=group_id))

@app.post("/groups/<int:group_id>/chat_stream")
def chat_stream(group_id: int):
    group = get_group(group_id)
    if not group:
        return jsonify({"error": "Group not found"}), 404

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Store user message
    add_message(group_id, "user", question)

    # Conversation memory
    msgs = get_messages(group_id, limit=30)
    history = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] in ("user", "assistant")]

    def event_stream():
        try:
            token_stream, citations = stream_answer_with_citations(group_id, question, history, k=5)

            full_answer = []
            for tok in token_stream:
                full_answer.append(tok)
                yield f"data: {json.dumps({'type':'token','value':tok})}\n\n"

            answer_text = "".join(full_answer).strip()
            if answer_text:
                add_message(group_id, "assistant", answer_text)
            else:
                add_message(group_id, "assistant", "(No response generated.)")

            # send citations at end
            yield f"data: {json.dumps({'type':'citations','value':citations})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"

        except Exception as e:
            err = str(e)
            add_message(group_id, "assistant", f"Error: {err}")
            yield f"data: {json.dumps({'type':'error','value':err})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)