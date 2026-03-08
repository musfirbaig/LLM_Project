"""
Standalone Similarity Search CLI (Milvus backend)
---------------------------------------------------
Loads the Milvus Lite collection and the SentenceTransformer embedding
model, then provides an interactive prompt where the user can type
natural-language queries and see the closest matching QA chunks ranked
by relevance score.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# ── Settings ───────────────────────────────────────────────────────────
MILVUS_DB_PATH   = "data/milvus_bank.db"
COLLECTION_NAME  = "bank_knowledge"
MODEL_ID         = "all-MiniLM-L6-v2"
DEFAULT_K        = 3


def _bootstrap():
    """Load the embedding model and connect to the Milvus collection."""
    embed_fn = HuggingFaceEmbeddings(model_name=MODEL_ID)
    store = Milvus(
        embed_fn,
        connection_args={"uri": MILVUS_DB_PATH},
        collection_name=COLLECTION_NAME,
    )
    return store


def find_similar(query: str, store: Milvus, k: int = DEFAULT_K):
    """Search *store* for the top-k passages closest to *query*."""
    hits = store.similarity_search_with_score(query, k=k)

    for rank, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata
        print(f"\n--- Result {rank}  (score: {score:.4f}) ---")
        print(f"  Product : {meta.get('product', 'N/A')}")
        print(f"  Q : {meta.get('question', '')}")
        print(f"  A : {meta.get('answer', '')[:200]}...")


def main():
    print("Loading Milvus collection and embedding model ...")
    store = _bootstrap()
    print("Ready.  Type 'exit' to quit.\n")

    while True:
        user_input = input("Search: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        find_similar(user_input, store)


if __name__ == "__main__":
    main()
