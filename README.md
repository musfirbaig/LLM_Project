# NUST Bank Product Knowledge -- RAG QA System

A Retrieval-Augmented Generation pipeline that answers banking product questions using a locally-hosted Qwen 3.5 LLM grounded in a FAISS vector store built from the bank's product knowledge base.

## How It Works

```
NUST Bank-Product-Knowledge.xlsx
            |
   format_for_finetuning.py      (extract Q&A pairs)
            |
     all_qa_pairs.json
            |
       embedder.py               (raw FAISS index + metadata)
            |
       embedder_2.py             (LangChain vector store)
            |
         llm.py                  (RAG inference via Qwen 3.5)
```

## Files

| File | Description |
|------|-------------|
| `embedder.py` | Stage-1 embedder -- loads QA pairs, normalises text, encodes with `all-MiniLM-L6-v2`, writes a FAISS index and chunk metadata. |
| `embedder_2.py` | Stage-2 embedder -- converts chunk metadata into a LangChain-compatible FAISS vector store. |
| `llm.py` | RAG inference engine -- loads the vector store, retrieves top-3 chunks, generates an answer with Qwen 3.5 via Ollama. |
| `search.py` | Standalone CLI similarity search over the raw FAISS index (no LLM). |
| `funds_transer_app_features_faq.json` | Supplementary mobile-app FAQ data. |
| `requirements.txt` | Python dependencies. |

### `data/` directory

| File | Description |
|------|-------------|
| `format_for_finetuning.py` | Extracts QA pairs from the Excel workbook into JSON and JSONL fine-tuning formats. |
| `inspect_data.py` | Diagnostic utility that prints shape, types, and sample rows for every Excel sheet. |
| `all_qa_pairs.json` | Extracted QA pairs consumed by the embedding pipeline. |
| `chunk_metadata.json` | Chunk-level metadata produced by `embedder.py`. |
| `finetuning_data.jsonl` | Alpaca-style instruction format (for optional fine-tuning). |
| `finetuning_data_chat.jsonl` | OpenAI chat-completion format (for optional fine-tuning). |
| `vectorstore/` | LangChain FAISS index (`index.faiss` + `index.pkl`). |

## Getting Started

```bash
# 1  Install dependencies
pip install -r requirements.txt

# 2  Extract QA pairs from the Excel file
python data/format_for_finetuning.py

# 3  Build embeddings (run in order)
python embedder.py
python embedder_2.py

# 4  Start Ollama (in a separate terminal)
ollama serve
ollama pull qwen3.5:4b

# 5  Run RAG QA
python llm.py
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with the `qwen3.5:4b` model pulled
