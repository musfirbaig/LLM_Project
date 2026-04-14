# Project Understanding: NUST Bank Product Knowledge QA System

## What This Project Is

This is a Retrieval-Augmented Generation (RAG) system for answering customer questions about NUST Bank products.

The pipeline is split into two parts:

1. Local retrieval and safety checks run on your machine.
2. Answer generation can run locally or on a remote Colab GPU server exposed through ngrok.

That lets the app work on a CPU-only laptop while still using the fine-tuned Qwen model on a GPU when available.

## Data Source

The knowledge base comes from:

- NUST Bank-Product-Knowledge.xlsx
- funds_transer_app_features_faq.json

The extraction script turns these into structured Q&A pairs, then the embedding pipeline stores them in Milvus Lite.

## Pipeline Overview

### Step 1: Data Extraction

data/format_for_finetuning.py parses the spreadsheet and FAQ JSON, then writes:

- data/all_qa_pairs.json
- data/finetuning_data.jsonl
- data/finetuning_data_chat.jsonl

### Step 2: Embedding and Vector Store

embedder.py creates a FAISS index and metadata files for inspection.

embedder_2.py writes the production vector store to data/milvus_bank.db using Milvus Lite.

### Step 3: Query-Time RAG

llm.py does the following:

1. Checks the user question against the guardrails.
2. Retrieves the top matching chunks from Milvus Lite.
3. Computes a retrieval confidence score.
4. Generates the final answer with Qwen 3.5 + LoRA.
5. Applies a post-generation safety filter.

If NUST_BANK_REMOTE_LLM_URL is set, step 4 is sent to the remote Colab API instead of loading the model locally.

## Model Runtime

| Property | Value |
|----------|-------|
| Base model | Qwen/Qwen3.5-4B |
| Fine-tuning method | LoRA via PEFT / Unsloth |
| Local runtime | HuggingFace Transformers + PEFT |
| Remote runtime | HTTP inference server on Colab, exposed with ngrok |
| Local fallback | CPU bfloat16 or GPU float16 + 4-bit quantisation |

## Remote GPU Setup

The Colab side runs remote_llm_server.py, which exposes:

- GET /health
- POST /ask

The local app sends the retrieved context and question to /ask, so the remote server only handles generation.

Set this on the local machine before running the app:

```powershell
$env:NUST_BANK_REMOTE_LLM_URL = "https://xxxx.ngrok-free.app"
```

Optional shared-secret protection:

```powershell
$env:NUST_BANK_REMOTE_LLM_TOKEN = "your_shared_secret"
```

## How the Pieces Fit Together

```text
NUST Bank-Product-Knowledge.xlsx
            |
            v
  format_for_finetuning.py
            |
            v
    all_qa_pairs.json
            |
            v
      embedder.py
            |
            v
    faiss_index.bin + chunk_metadata.json
            |
            v
      embedder_2.py
            |
            v
     data/milvus_bank.db
            |
            v
         llm.py
      /           \
 local model   remote Colab API
      \           /
       v         v
     final answer
```

## What To Use For What

- Use streamlit_app.py for the web UI.
- Use llm.py for retrieval + answer generation.
- Use remote_llm_server.py on Colab when you want GPU inference remotely.
- Use search.py when you only want similarity search with no LLM.

## Practical Notes

- Keep the Colab runtime alive while the local app is using the remote API.
- The local machine still needs the Milvus Lite database and embedding model files.
- If the remote server is unavailable, llm.py will fail fast instead of silently pretending it has a GPU.
- streamlit_app.py does not need special changes; it calls ask() and inherits the remote/local switch automatically.
