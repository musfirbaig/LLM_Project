"""
Stage-1 Embedding Pipeline
---------------------------
Reads the cleaned QA-pair JSON, normalises every entry, converts each pair
into an embeddable text chunk, generates dense vectors with a
SentenceTransformer encoder, and persists a FAISS index alongside the
corresponding chunk metadata.
"""

import json
import re
import pathlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────
QA_SOURCE_FILE    = pathlib.Path("data/all_qa_pairs.json")
INDEX_DEST        = pathlib.Path("data/faiss_index.bin")
METADATA_DEST     = pathlib.Path("data/chunk_metadata.json")
ENCODER_MODEL_ID  = "all-MiniLM-L6-v2"


# ── Text pre-processing ─────────────────────────────────────────────────

def _sanitise(raw_text: str) -> str:
    """Strip pipe-table artefacts, collapse whitespace, remove bullets."""
    result = re.sub(r"\s*\|\s*", " ", raw_text)          # pipe separators
    result = re.sub(r"\s+", " ", result)                  # excess spaces
    result = re.sub(r"^[^\w\s]+\s*", "", result, flags=re.MULTILINE)  # leading bullets/nums
    return result.strip()


# ── Pipeline stages ──────────────────────────────────────────────────────

def ingest_qa_pairs(src: pathlib.Path) -> list[dict]:
    """Read QA pairs from *src*, discarding entries with blank fields."""
    with open(src, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    filtered = [
        rec for rec in raw
        if rec.get("question", "").strip() and rec.get("answer", "").strip()
    ]
    print(f"[load] kept {len(filtered)} of {len(raw)} entries")
    return filtered


def assemble_chunks(qa_records: list[dict]) -> list[dict]:
    """Turn each QA record into a single text chunk with metadata."""
    output = []
    for pos, rec in enumerate(qa_records):
        q = _sanitise(rec["question"])
        a = _sanitise(rec["answer"])
        output.append({
            "id":       pos,
            "text":     f"Question: {q}\nAnswer: {a}",
            "question": q,
            "answer":   a,
            "product":  rec.get("product", "Unknown"),
            "sheet":    rec.get("sheet", "Unknown"),
        })
    print(f"[chunk] assembled {len(output)} passages")
    return output


def encode_passages(
    chunks: list[dict],
    model_id: str = ENCODER_MODEL_ID,
):
    """Vectorise chunk texts and return (matrix, encoder)."""
    print(f"[encode] initialising '{model_id}' ...")
    encoder = SentenceTransformer(model_id)

    passages = [c["text"] for c in chunks]
    print(f"[encode] processing {len(passages)} passages ...")
    matrix = encoder.encode(passages, show_progress_bar=True, convert_to_numpy=True)

    print(f"[encode] matrix shape: {matrix.shape}")
    return matrix, encoder


def write_faiss_artefacts(
    matrix: np.ndarray,
    chunks: list[dict],
    idx_path: pathlib.Path = INDEX_DEST,
    meta_path: pathlib.Path = METADATA_DEST,
) -> faiss.Index:
    """Construct a flat-L2 FAISS index and persist everything to disk."""
    dim = matrix.shape[1]
    store = faiss.IndexFlatL2(dim)
    store.add(matrix)

    faiss.write_index(store, str(idx_path))
    print(f"[write] index   -> {idx_path}")

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=4)
    print(f"[write] metadata -> {meta_path}")
    return store


# ── Orchestrator ─────────────────────────────────────────────────────────

def run():
    records    = ingest_qa_pairs(QA_SOURCE_FILE)
    chunks     = assemble_chunks(records)
    matrix, _  = encode_passages(chunks)
    store      = write_faiss_artefacts(matrix, chunks)

    print(
        f"\n[summary] chunks={len(chunks)}  dim={matrix.shape[1]}  "
        f"vectors={matrix.shape}  stored={store.ntotal}"
    )


if __name__ == "__main__":
    run()
