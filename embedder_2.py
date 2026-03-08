"""
Stage-2: Milvus Vector Store Builder
--------------------------------------
Reads chunk metadata produced by the stage-1 embedder, wraps each chunk as
a LangChain Document, computes embeddings via HuggingFace, and upserts
everything into a Milvus Lite collection stored as a local .db file.

Milvus Lite runs entirely in-process — no Docker, no server required.
"""

import json
import pathlib

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# ── Configuration ─────────────────────────────────────────────────────────
CHUNK_META_PATH  = pathlib.Path("data/chunk_metadata.json")
MILVUS_DB_PATH   = "data/milvus_bank.db"      # Milvus Lite: just a file
COLLECTION_NAME  = "bank_knowledge"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_chunks(path: pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_documents(chunks: list[dict]) -> list[Document]:
    """Wrap each chunk dict in a LangChain Document."""
    return [
        Document(
            page_content=ch["text"],
            metadata={
                "chunk_id": ch["id"],
                "question": ch["question"],
                "answer":   ch["answer"],
                "product":  ch.get("product", "Unknown"),
                "sheet":    ch.get("sheet",   "Unknown"),
            },
        )
        for ch in chunks
    ]


# ── Main builder ──────────────────────────────────────────────────────────

def build_milvus_store():
    chunks = _load_chunks(CHUNK_META_PATH)
    docs   = _to_documents(chunks)
    print(f"[milvus] {len(docs)} documents ready for insertion")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"[milvus] embedding model '{EMBEDDING_MODEL}' loaded")

    vec_store = Milvus.from_documents(
        docs,
        embedder,
        connection_args={"uri": MILVUS_DB_PATH},
        collection_name=COLLECTION_NAME,
        drop_old=True,          # rebuild from scratch each run
    )

    total = vec_store.col.num_entities
    print(f"[milvus] persisted {total} vectors  ->  {MILVUS_DB_PATH}")
    return vec_store


if __name__ == "__main__":
    build_milvus_store()
