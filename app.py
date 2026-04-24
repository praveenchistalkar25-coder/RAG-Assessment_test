# project layout suggestion:
# ├── RAG-Project2/
# │   ├── dataloading.py
# │   ├── chunking.py
# │   ├── embedding.py
# │   ├── retrival.py
# │   ├── app.py
# │   ├── requirements.txt
# │   └── .env

# -------------------------------
# dataloading.py
# -------------------------------
import os

def load_documents(folder_path: str) -> dict:
    """
    Load all text documents from a folder.
    For PDFs, XLSX, DOCX you can extend with PyMuPDF, pandas, etc.
    Here we assume plain text extraction already done.
    """
    docs = {}
    for fname in os.listdir(folder_path):
        if fname.endswith((".pdf", ".txt", ".xlsx")):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8", errors="ignore") as f:
                docs[fname] = f.read()
    return docs


# -------------------------------
# chunking.py
# -------------------------------
def chunk_text(doc_name: str, text: str):
    """
    Chunk documents based on type.
    """
    chunks = []
    if "SOW" in doc_name or "Proposal" in doc_name:
        parts = text.split("## ")
        chunks = [{"doc": doc_name, "chunk_id": i, "text": p.strip()} for i, p in enumerate(parts) if p.strip()]
    elif "CaseStudy" in doc_name:
        parts = text.split("## ")
        chunks = [{"doc": doc_name, "chunk_id": i, "text": p.strip()} for i, p in enumerate(parts) if p.strip()]
    elif "Regression_Test_Plan" in doc_name:
        rows = text.split("CHK-")
        chunks = [{"doc": doc_name, "chunk_id": i, "text": "CHK-" + r.strip()} for i, r in enumerate(rows) if r.strip()]
    else:
        words = text.split()
        for i in range(0, len(words), 500):
            chunk = " ".join(words[i:i+500])
            chunks.append({"doc": doc_name, "chunk_id": i, "text": chunk})
    return chunks


# -------------------------------
# embedding.py
# -------------------------------
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding)


# -------------------------------
# retrival.py
# -------------------------------
import numpy as np

vector_store = []

def add_to_store(chunks, embed_fn):
    for ch in chunks:
        emb = embed_fn(ch["text"])
        vector_store.append({**ch, "embedding": emb})

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, embed_fn, top_k: int = 3):
    q_emb = embed_fn(query)
    scored = [(cosine_similarity(q_emb, item["embedding"]), item) for item in vector_store]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# -------------------------------
# app.py
# -------------------------------
from dataloading import load_documents
from chunking import chunk_text
from embedding import embed_text
from retrival import add_to_store, retrieve

if __name__ == "__main__":
    folder = "./documents"  # put your PDFs/XLSX/TXT here
    docs = load_documents(folder)

    for name, text in docs.items():
        chunks = chunk_text(name, text)
        add_to_store(chunks, embed_text)

    results = retrieve("AI solutions delivered for healthcare clients", embed_text)

    for score, item in results:
        print(f"Source: {item['doc']} | Chunk ID: {item['chunk_id']} | Score: {score:.4f}")
        print(item["text"][:400], "...\n")

