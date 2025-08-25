from flask import Flask, request, jsonify
import os
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = "AIzaSyD_oebgqWRp8P7QA2_XsoS5pIqh2DENaWY"
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = 384

faiss_index = faiss.IndexFlatL2(EMBED_DIM)
documents: list[dict] = []   

def extract_text_pymupdf(filepath: str) -> str:
    text = []
    doc = fitz.open(filepath)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
    words = text.split()
    chunks = []
    if not words:
        return chunks
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def add_to_index(chunks: list[dict], source_path: str):
    global documents, faiss_index
    if not chunks:
        return 0
 
    texts = [c["text"] for c in chunks]
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    faiss_index.add(embs.astype(np.float32))
    
    start_id = len(documents)
    for i, ch in enumerate(chunks):
        documents.append({
            "text": ch["text"],
            "source": source_path,
            "page": ch.get("page"),
            "line": ch.get("line"),
            "chunk_id": start_id + i
        })
    return len(chunks)

def retrieve(query: str, k: int = 5) -> list[dict]:
    if faiss_index.ntotal == 0:
        return []
    q = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = faiss_index.search(q, min(k, faiss_index.ntotal))
    hits = []
    for idx, dist in zip(I[0], D[0]):
        if 0 <= idx < len(documents):
            rec = documents[idx]
            hits.append({
                "text": rec["text"],
                "source": rec["source"],
                "page": rec.get("page"),
                "line": rec.get("line"),
                "score": float(dist)
            })
    return hits

def build_prompt(context_chunks: list[dict], query: str, target_words: int | None = 100) -> str:
    ctx = "\n\n".join(
        [f"(Chunk #{c['chunk_id']} | Source: {os.path.basename(c['source'])})\n{c['text']}" for c in context_chunks]
    )
    target = f" around {target_words} words" if target_words else ""
    prompt = f"""
You are a research assistant for scientific documents. 
Answer the question using ONLY the provided context. 
If the information is not present in the context, reply exactly:
"The information is not in the provided document."

When you state facts, cite the chunk numbers like [chunk:12] corresponding to the chunk identifiers in the context.
Keep equations in LaTeX if present.

Context:
{ctx}

Question:
{query}

Answer{target}:
"""
    return prompt.strip()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "docs_indexed": len(documents), "faiss_ntotal": faiss_index.ntotal})

@app.route("/reset", methods=["POST"])
def reset():
    global documents, faiss_index
    documents = []
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    return jsonify({"message": "Index cleared", "docs_indexed": 0})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file field 'file'"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, f.filename)
    f.save(path)

    text = extract_text_pymupdf(path)
    if not text or not text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    added = add_to_index(chunks, source_path=path)

    return jsonify({
        "message": "File processed",
        "chunks_indexed": added,
        "sample_chunks": chunks[:3]
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    k = int(data.get("k", 5))
    target_words = data.get("target_words", 100)  

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if faiss_index.ntotal == 0:
        return jsonify({"error": "No documents in index. Upload a PDF first."}), 400

    topk = retrieve(query, k=k)
    if not topk:
        return jsonify({"error": "Nothing retrieved for this query."}), 404

    prompt = build_prompt(topk, query, target_words=target_words)

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        answer = (resp.text or "").strip()
    except Exception as e:
        return jsonify({"error": f"Gemini error: {str(e)}"}), 500

    if "The information is not in the provided document." in answer:
        return jsonify({
            "query": query,
            "retrieved_context": topk,
            "answer": "The information is not in the provided document.",
            "note": "Try uploading more relevant PDFs or ask a narrower question."
        })

    return jsonify({
        "query": query,
        "retrieved_context": topk,
        "answer": answer
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
