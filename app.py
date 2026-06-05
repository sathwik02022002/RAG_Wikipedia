"""
Flask API wrapper around the RAG pipeline.

Run:
    pip install flask flask-cors pdfplumber faiss-cpu scikit-learn numpy requests
    export LLM_API_KEY="sk-..."
    python app.py

Then open index.html in your browser.
"""

import os, re, sys, math, heapq
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import requests

app = Flask(__name__)
CORS(app)  # allows the HTML file to call this server

# ── Data classes ──────────────────────────────────────────────

@dataclass
class Chunk:
    id: int
    text: str
    metadata: Dict = field(default_factory=dict)

# ── PDF extraction ────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    def clean(text):
        if not text: return ""
        text = re.sub(r'\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}\s+Machine learning - Wikipedia', '', text)
        text = re.sub(r'https://en\.wikipedia\.org/wiki/Machine_learning\s+\d+/\d+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [l for l in text.split('\n') if not re.fullmatch(r'\s*\d+\s*', l)]
        return '\n'.join(lines).strip()

    pages = []
    with pdfplumber.open(path) as pdf:
        print(f"[PDF] {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages):
            raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            cleaned = clean(raw)
            if cleaned:
                pages.append(cleaned)
            print(f"  Page {i+1}/{len(pdf.pages)}", end='\r')
    print(f"\n[PDF] Done — {len(' '.join(pages).split())} words extracted")
    return "\n\n".join(pages)

# ── Chunking ──────────────────────────────────────────────────

def make_chunks(text: str) -> List[Chunk]:
    """Semantic section chunking — best strategy from evaluation."""
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    merged, buf = [], ""
    for p in paras:
        if buf:
            if len(buf.split()) < 50:
                buf += " " + p
                continue
            merged.append(buf)
            buf = p
        else:
            buf = p
    if buf:
        merged.append(buf)
    return [Chunk(i, t) for i, t in enumerate(merged)]

# ── Embedding (LSA) ───────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, n=128):
        self.dim = n
        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=n, random_state=42)

    def fit_encode(self, texts):
        mat = self.vec.fit_transform(texts)
        self.svd.fit(mat)
        return normalize(self.svd.transform(mat).astype(np.float32), norm='l2')

    def encode(self, texts):
        mat = self.vec.transform(texts)
        return normalize(self.svd.transform(mat).astype(np.float32), norm='l2')

# ── BM25 ──────────────────────────────────────────────────────

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def _tok(self, t):
        return re.findall(r'\b[a-z]{2,}\b', t.lower())

    def fit(self, chunks):
        self.chunks = chunks
        self.N = len(chunks)
        self.toks = [self._tok(c.text) for c in chunks]
        self.df = defaultdict(int)
        for t in self.toks:
            for w in set(t): self.df[w] += 1
        self.avgdl = sum(len(t) for t in self.toks) / self.N
        self.tf = [Counter(t) for t in self.toks]

    def search(self, query, k=10):
        qt = self._tok(query)
        scores = []
        for i in range(self.N):
            dl = sum(self.tf[i].values())
            s = 0.0
            for w in qt:
                if w not in self.df: continue
                f = self.tf[i].get(w, 0)
                idf = math.log((self.N - self.df[w] + .5) / (self.df[w] + .5) + 1)
                s += idf * f*(self.k1+1) / (f + self.k1*(1-self.b+self.b*dl/self.avgdl))
            scores.append((s, i))
        top = heapq.nlargest(k, scores, key=lambda x: x[0])
        return [(self.chunks[i], sc) for sc, i in top if sc > 0]

# ── Retrieval ─────────────────────────────────────────────────

def hybrid_search(query, chunks, embed, vs_index, bm25, k=5):
    # Dense
    q_vec = embed.encode([query])
    sc, ix = vs_index.search(q_vec, k*2)
    dense = [(chunks[i], float(s)) for s, i in zip(sc[0], ix[0]) if i >= 0]
    # Sparse
    sparse = bm25.search(query, k*2)
    # RRF
    scores, cmap = defaultdict(float), {}
    for r, (c, _) in enumerate(dense):
        scores[c.id] += 0.6 / (61 + r); cmap[c.id] = c
    for r, (c, _) in enumerate(sparse):
        scores[c.id] += 0.4 / (61 + r); cmap[c.id] = c
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cmap[cid] for cid, _ in ranked[:k]]

# ── Global pipeline state ─────────────────────────────────────

print("\n[Startup] Loading pipeline...")

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Machine_learning_-_Wikipedia.pdf")

# Also check current working directory (for Render)
if not os.path.exists(PDF_PATH):
    PDF_PATH = "Machine_learning_-_Wikipedia.pdf"

if not os.path.exists(PDF_PATH):
    print(f"\n[Error] PDF not found: {PDF_PATH}")
    print("Place Machine_learning_-_Wikipedia.pdf next to app.py")
    sys.exit(1)

raw_text = extract_pdf(PDF_PATH)
chunks   = make_chunks(raw_text)
print(f"[Pipeline] {len(chunks)} chunks created")

embed = EmbeddingModel(n=128)
embs  = embed.fit_encode([c.text for c in chunks])

vs_index = faiss.IndexFlatIP(128)
vs_index.add(embs)

bm25 = BM25()
bm25.fit(chunks)

LLM_API_KEY      = os.environ.get("LLM_API_KEY", "")
LLM_ENDPOINT     = LLM_ENDPOINT
LLM_MODEL        = os.environ.get("LLM_MODEL", "")
print("[Pipeline] Ready ✓\n")

# ── Flask routes ──────────────────────────────────────────────

@app.route("/ask", methods=["POST"])
def ask():
    if not pipeline_ready:
        return jsonify({"error": "PDF not loaded. Add Machine_learning_-_Wikipedia.pdf to your repo."}), 503

    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    top_chunks = hybrid_search(question, chunks, embed, vs_index, bm25_idx, k=5)
    context    = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c.text}" for i, c in enumerate(top_chunks))

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        payload = {
            "model": LLM_MODEL,
            "max_tokens": 512,
            "messages": [
                {"role": "system",
                 "content": "You are an ML assistant. Answer using ONLY the provided context. Be clear and concise."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
        }
        r = requests.post(
            LLM_ENDPOINT,
            headers=headers, json=payload, timeout=30
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sources = [c.text[:200] + "..." for c in top_chunks]
    return jsonify({"answer": answer, "sources": sources})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if pipeline_ready else "pdf_missing",
        "chunks": len(chunks),
        "pipeline_ready": pipeline_ready
    })

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    print("Open index.html in your browser\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
