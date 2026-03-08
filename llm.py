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
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# ── Configuration ───────────────────────────────────────────────────────
GENERATIVE_MODEL  = "qwen3.5:4b"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
MILVUS_DB_PATH    = "data/milvus_bank.db"
COLLECTION_NAME   = "bank_knowledge"
TOP_K_RESULTS     = 3


# ── Chain setup ─────────────────────────────────────────────────────────

def initialise_qa_chain() -> RetrievalQA:
    """Connect the Milvus retriever and Ollama LLM into a single QA chain."""
    generator = OllamaLLM(model=GENERATIVE_MODEL)

    embed_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    knowledge_base = Milvus(
        embed_fn,
        connection_args={"uri": MILVUS_DB_PATH},
        collection_name=COLLECTION_NAME,
    )

    retriever = knowledge_base.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS},
    )

    chain = RetrievalQA.from_chain_type(
        llm=generator,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
    )
    return chain


def ask(question: str) -> dict:
    """Build the chain and run *question* through it."""
    chain = initialise_qa_chain()
    return chain.invoke({"query": question})


if __name__ == "__main__":
    sample_query = "how do i open an account?"
    result = ask(sample_query)
    print("response:", result)
