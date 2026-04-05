# NUST Bank RAG QA System

> **Course:** Large Language Models (LLM) — Spring 2026, NUST  
> **Group:** Muhammad Musfir Baig (409968) · Awais Nazir (406270)

An AI-powered customer support assistant that answers questions about NUST Bank products using **Retrieval-Augmented Generation (RAG)** with a **fine-tuned Qwen 3.5-4B** language model.

📖 **[Full Documentation → DOCUMENTATION.md](./DOCUMENTATION.md)**  
🚀 **[Setup Guide → SETUP_GUIDE.md](./SETUP_GUIDE.md)**

---

## How It Works

```
NUST Bank-Product-Knowledge.xlsx
              │
   format_for_finetuning.py  →  all_qa_pairs.json
              │
         embedder.py          →  chunk_metadata.json + faiss_index.bin
              │
         embedder_2.py        →  data/milvus_bank.db  (Milvus Lite)
              │
   ┌──────────┴──────────┐
   │     Query time       │
   │  User Question       │
   │       ↓              │
   │  Guardrails (pre)    │
   │       ↓              │
   │  Milvus search (k=3) │
   │       ↓              │
   │  Qwen 3.5-4B + LoRA  │
   │       ↓              │
   │  Guardrails (post)   │
   │       ↓              │
   │  Final Answer        │
   └──────────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build knowledge base (first time only)
python data/format_for_finetuning.py
python embedder.py
python embedder_2.py

# 3. Launch the web interface
streamlit run streamlit_app.py
# → Opens at http://localhost:8501
```

> ⚠️ First launch downloads `Qwen/Qwen3.5-4B` (~8 GB). CPU inference ~30-60s/answer. GPU inference ~2-5s/answer.

---

## Features

| Feature | Description |
|---------|-------------|
| 🤖 Fine-tuned LLM | Qwen 3.5-4B + LoRA adapter trained on NUST Bank data |
| 🔍 RAG Pipeline | Milvus Lite vector store + `all-MiniLM-L6-v2` embeddings |
| 🛡️ Guardrails | Prompt injection, harmful content, confidence-based filtering |
| 📄 Real-time Updates | Upload JSON or add Q&A entries — indexed instantly (no restart) |
| 💻 GPU + CPU | Auto-detects hardware; 4-bit quantised on GPU, bfloat16 on CPU |
| 🌐 Web UI | Chat interface + document upload + manual entry form |

---

## Upload Format

The **Upload Data** tab accepts a JSON array:

```json
[
  {
    "question": "How do I open a savings account?",
    "answer": "Visit any NUST Bank branch with your CNIC.",
    "product": "Savings Account"
  }
]
```

---

## Project Files

| File | Purpose |
|------|---------|
| `llm.py` | RAG inference engine (Qwen 3.5 + LoRA, GPU/CPU auto-detect) |
| `embedder.py` | Build FAISS index from Q&A pairs |
| `embedder_2.py` | Build Milvus Lite vector store |
| `guardrails.py` | Safety layer (pre + post generation) |
| `search.py` | CLI similarity search tool (no LLM) |
| `streamlit_app.py` | Web interface |
| `qwen3.5_banking_lora/` | Fine-tuned LoRA adapter weights |
| `data/` | Knowledge base files and data scripts |
| `scripts/` | Fine-tuning and validation scripts |
| `DOCUMENTATION.md` | Full architecture & component docs |
| `SETUP_GUIDE.md` | Step-by-step local + Colab setup |
