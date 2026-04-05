"""
RAG Inference Engine (Milvus backend)
---------------------------------------
Loads the Milvus Lite vector store, connects to a locally-running
Qwen 3.5 (4B) model via Ollama, and runs a retrieval-augmented QA
chain that fetches the top-k relevant knowledge chunks before generating
a final answer.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import OllamaLLM

from guardrails import inspect_query, post_filter, retrieval_confidence_from_distance

# ── Configuration ───────────────────────────────────────────────────────
GENERATIVE_MODEL  = "qwen3.5:4b"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
MILVUS_DB_PATH    = "data/milvus_bank.db"
COLLECTION_NAME   = "bank_knowledge"
TOP_K_RESULTS     = 3
MIN_RETRIEVAL_CONFIDENCE = 0.35


# ── Chain setup ─────────────────────────────────────────────────────────

def initialise_components() -> tuple[OllamaLLM, Milvus]:
    """Create generator and vector store clients."""
    generator = OllamaLLM(model=GENERATIVE_MODEL)
    embed_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    knowledge_base = Milvus(
        embed_fn,
        connection_args={"uri": MILVUS_DB_PATH},
        collection_name=COLLECTION_NAME,
    )
    return generator, knowledge_base


def _build_prompt(question: str, contexts: list[str]) -> str:
    merged_context = "\n\n".join(contexts)
    return (
        "You are a caring customer support assistant for NUST Bank.\n"
        "Follow these rules:\n"
        "1) Use only the provided context.\n"
        "2) If the answer is not in context, say you do not have enough information.\n"
        "3) Be concise, polite, and practical.\n\n"
        f"Context:\n{merged_context}\n\n"
        f"Customer question: {question}\n\n"
        "Answer:"
    )


def ask(question: str) -> dict:
    """Answer a question using retrieval + guarded generation."""
    gate = inspect_query(question)
    if not gate.allowed:
        return {
            "answer": gate.reason,
            "guardrail_blocked": True,
            "sources": [],
        }

    generator, knowledge_base = initialise_components()
    hits = knowledge_base.similarity_search_with_score(question, k=TOP_K_RESULTS)

    if not hits:
        return {
            "answer": "I could not find relevant bank information for that question.",
            "guardrail_blocked": False,
            "sources": [],
        }

    best_distance = min(score for _, score in hits)
    confidence = retrieval_confidence_from_distance(best_distance)
    if confidence < MIN_RETRIEVAL_CONFIDENCE:
        return {
            "answer": "I do not have enough relevant bank information to answer that confidently.",
            "guardrail_blocked": False,
            "confidence": confidence,
            "sources": [],
        }

    contexts = [doc.page_content for doc, _ in hits]
    prompt = _build_prompt(question, contexts)
    raw_answer = generator.invoke(prompt)
    safe_answer = post_filter(raw_answer)

    source_payload = []
    for doc, score in hits:
        source_payload.append(
            {
                "score": float(score),
                "product": doc.metadata.get("product", "N/A"),
                "question": doc.metadata.get("question", ""),
            }
        )

    return {
        "answer": safe_answer,
        "guardrail_blocked": False,
        "confidence": confidence,
        "sources": source_payload,
    }


if __name__ == "__main__":
    sample_query = "how do i open an account?"
    result = ask(sample_query)
    print("response:", result)
