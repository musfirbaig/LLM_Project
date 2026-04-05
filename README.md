# NUST Bank Product Knowledge -- RAG QA System

A Retrieval-Augmented Generation pipeline that answers banking product questions using a locally-hosted Qwen 3.5 LLM grounded in a **Milvus Lite** vector store built from the bank's product knowledge base.

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
| `embedder_2.py` | Stage-2 embedder -- converts chunk metadata into a Milvus Lite collection (`data/milvus_bank.db`). |
| `llm.py` | RAG inference engine -- loads Milvus, retrieves top-3 chunks, generates an answer with Qwen 3.5 via Ollama. |
| `guardrails.py` | Input/output safety layer used by `llm.py` (prompt-injection and harmful query checks + safe fallbacks). |
| `search.py` | Standalone CLI similarity search over the Milvus collection (no LLM). |
| `streamlit_app.py` | Web interface for asking questions and uploading new QA JSON data. |
| `funds_transer_app_features_faq.json` | Supplementary mobile-app FAQ data. |
| `requirements.txt` | Python dependencies. |
| `scripts/validate_finetuning_data.py` | Validates chat JSONL fine-tuning data and writes deterministic train/val/test splits. |
| `scripts/train_qlora_qwen.py` | QLoRA fine-tuning script for Qwen-style chat models using HuggingFace + PEFT. |

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
python embedder.py        # produces chunk_metadata.json
python embedder_2.py      # produces data/milvus_bank.db  (Milvus Lite)

# 4  Start Ollama (in a separate terminal)
ollama serve
ollama pull qwen3.5:4b

# 5  Run RAG QA
python llm.py

# 6  (Optional) Validate and split fine-tuning data
python scripts/validate_finetuning_data.py

# 7  (Optional) Fine-tune with QLoRA (Colab/local GPU)
python scripts/train_qlora_qwen.py \
     --model-id Qwen/Qwen2.5-3B-Instruct \
     --train-file data/splits/train.jsonl \
     --val-file data/splits/val.jsonl \
     --output-dir checkpoints/qwen_lora

# 8  (Optional) Run Streamlit interface
streamlit run streamlit_app.py
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) with the `qwen3.5:4b` model pulled

## Upload JSON Format (for `streamlit_app.py`)

The upload tab accepts a JSON array with objects containing at least `question` and `answer`:

```json
[
     {
          "question": "How can I open a savings account?",
          "answer": "You can open an account by visiting any NUST Bank branch with your CNIC.",
          "product": "Savings Account"
     }
]
```
