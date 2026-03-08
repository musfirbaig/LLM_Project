# Project Understanding: NUST Bank Product Knowledge QA System

## What This Project Is

This is a **Retrieval-Augmented Generation (RAG)** system built for answering customer questions about NUST Bank's products. Instead of training a model from scratch, RAG works by:

1. Storing all the bank's FAQ knowledge in a searchable vector database
2. When a user asks a question, finding the most relevant stored answers
3. Feeding those relevant answers as context to a language model (Qwen 3.5) which generates a final human-friendly response

---

## Data Source

The primary knowledge base is an Excel file: `NUST Bank-Product-Knowledge.xlsx`
- Contains **34+ sheets**, each representing a different bank product (e.g., Little Champs Account, Savings Account, etc.)
- Each sheet has Q&A pairs in a semi-structured layout (merged cells, multi-row answers, side-by-side columns)
- There's also a supplementary JSON file (`funds_transer_app_features_faq.json`) with mobile app FAQs

---

## Complete Pipeline Flow (Step by Step)

### Step 0: Data Inspection (`data/inspect_data.py`)
- **Purpose:** Utility script to peek into the Excel file
- **What it does:** Loads every sheet using pandas, prints shape, column names, data types, null counts, unique values, and sample rows
- **When to run:** Only when you want to understand the raw Excel structure — not part of the main pipeline

### Step 1: Extract Q&A Pairs from Excel (`data/format_for_finetuning.py`)
- **Input:** `NUST Bank-Product-Knowledge.xlsx` + `funds_transer_app_features_faq.json`
- **Output:** Three files:
  - `data/all_qa_pairs.json` — All extracted Q&A pairs (used by the embedding pipeline)
  - `data/finetuning_data.jsonl` — Alpaca instruction format (for potential fine-tuning)
  - `data/finetuning_data_chat.jsonl` — OpenAI chat format (for potential fine-tuning)

**How it works:**
1. Opens the Excel workbook with `openpyxl`
2. Skips non-relevant sheets (Main, Rate Sheet, empty sheets)
3. For each sheet, reads all cell values row-by-row
4. Uses a heuristic (`is_question()`) to detect which rows are questions vs. answers — checks for "?", or if the text starts with "What", "How", "Is", etc.
5. Groups consecutive answer rows under the preceding question
6. Also loads the JSON FAQ file and merges those Q&A pairs in
7. Writes everything to the three output formats

### Step 2: Build Raw FAISS Index (`embedder.py`)
- **Input:** `data/all_qa_pairs.json`
- **Output:** `data/faiss_index.bin` + `data/chunk_metadata.json`

**How it works:**
1. **Load & Clean:** Reads the JSON, filters out entries with empty question or answer
2. **Normalize:** Removes pipe-delimited table formatting, collapses whitespace, strips bullets/numbers
3. **Chunking:** Each Q&A pair becomes one chunk — the text is formatted as `"Question: ...\nAnswer: ..."`
4. **Embedding:** Uses the `all-MiniLM-L6-v2` model from SentenceTransformers to convert each chunk into a 384-dimensional vector
5. **FAISS Index:** Creates a flat L2 (Euclidean distance) index and adds all vectors
6. **Save:** Writes the binary FAISS index and a JSON metadata file (so you can look up the original text for any vector)

### Step 3: Build Milvus Vector Store (`embedder_2.py`)
- **Input:** `data/chunk_metadata.json` (from Step 2)
- **Output:** `data/milvus_bank.db` — a Milvus Lite database file (single file on disk, no server)

**How it works:**
1. Reads the chunk metadata JSON
2. Wraps each chunk as a LangChain `Document` object (with page_content + metadata)
3. Uses `HuggingFaceEmbeddings` (same `all-MiniLM-L6-v2` model) to embed all documents
4. Creates a **Milvus Lite** collection called `bank_knowledge` inside the `.db` file
5. Upserts all vectors and metadata — this is what `llm.py` and `search.py` load at query time

**Why two embedding steps?** Step 2 still produces `chunk_metadata.json` (flat JSON, useful for inspection). Step 3 puts everything into Milvus so that both the RAG chain and the search CLI share one consistent vector store.

**What is Milvus Lite?** It is an embedded variant of Milvus that runs entirely inside the Python process — no Docker, no service to start: it's just a `.db` file, similar to SQLite.

### Step 4: RAG Inference (`llm.py`)
- **Input:** User query + `data/milvus_bank.db`
- **Output:** Generated answer

**How it works:**
1. Loads the Qwen 3.5 (4B parameter) model via Ollama (Ollama must be running as a local server)
2. Opens the Milvus Lite database and connects to the `bank_knowledge` collection
3. Creates a `RetrievalQA` chain using the "stuff" strategy:
   - **Retrieval:** For any user query, finds the top-3 most similar chunks from Milvus
   - **Stuffing:** Concatenates those 3 chunks into a single context prompt
   - **Generation:** Sends context + question to Qwen 3.5, which generates the final answer
4. Currently hardcoded to ask: "how do i open an account?"

### Bonus: Interactive Search (`search.py`)
- Opens the Milvus Lite collection with the same embedding model
- Provides an interactive CLI loop where you type a query and get the top-3 nearest chunks with relevance scores
- No LLM involved — pure vector similarity search
- Useful for debugging and verifying the embeddings are working

---

## Embedding Model Details

| Property | Value |
|----------|-------|
| Model | `all-MiniLM-L6-v2` (SentenceTransformers / HuggingFace) |
| Vector dimension | 384 |
| Type | Sentence embedding (bi-encoder) |
| Size | ~80MB |
| Speed | Very fast, runs on CPU easily |

**Note:** The README mentions "Qwen latest model for embedding" but the code actually uses `all-MiniLM-L6-v2` for embeddings. Qwen 3.5 (4B) is used only as the **generative LLM** via Ollama for answer generation. The embeddings are NOT produced by Qwen.

---

## LLM Model Details

| Property | Value |
|----------|-------|
| Model | `qwen3.5:4b` |
| Runtime | Ollama (local inference server) |
| Role | Answer generation (takes retrieved context + question, produces answer) |
| Requirement | Ollama must be installed and running (`ollama serve`) |

---

## Architecture Diagram

```
NUST Bank-Product-Knowledge.xlsx
            |
            v
  format_for_finetuning.py
            |
            v
    all_qa_pairs.json ──────────────────┐
            |                           |
            v                           v
      embedder.py                  (finetuning JSONL outputs)
            |
            v
    faiss_index.bin  +  chunk_metadata.json
                                   |
                                   v
                             embedder_2.py
                                   |
                                   v
                          data/milvus_bank.db        <-- Milvus Lite
                         (bank_knowledge collection)
                            /              \
                           v                v
                       llm.py           search.py
              (Qwen 3.5 via Ollama)  (direct similarity)
                           |
                           v
                   Generated Answer
```

---

## How to Run on Google Colab

Since Ollama (needed for Qwen 3.5) runs as a local server, Colab needs a special setup. Here's how:

### Cell 1: Install System Dependencies & Ollama
```python
# Install Ollama on Colab
!curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server in background
import subprocess
process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for it to start
import time
time.sleep(5)

# Pull the Qwen model
!ollama pull qwen3.5:4b
```

### Cell 2: Install Python Dependencies
```python
!pip install langchain-classic langchain-core langchain-community langchain-ollama \
             langchain-huggingface langchain-text-splitters \
             "pymilvus>=2.4.0" "langchain-milvus>=0.1.0" \
             faiss-cpu sentence-transformers huggingface_hub ollama openpyxl
```

### Cell 3: Upload Your Files
```python
# Option A: Upload from local machine
from google.colab import files

# Upload these files:
# - NUST Bank-Product-Knowledge.xlsx
# - funds_transer_app_features_faq.json
uploaded = files.upload()

# Option B: If using Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# Then copy files from your Drive path
```

### Cell 4: Create Directory Structure
```python
import os
os.makedirs('data', exist_ok=True)
# Milvus Lite writes its .db file directly into data/ — no sub-dir needed
```

### Cell 5: Run the Data Extraction
```python
# Run format_for_finetuning.py
# (Make sure NUST Bank-Product-Knowledge.xlsx is in the parent directory or adjust the path)
%run data/format_for_finetuning.py
```

### Cell 6: Run Embedder Step 1
```python
%run embedder.py
```

### Cell 7: Run Embedder Step 2
```python
%run embedder_2.py
```

### Cell 8: Run RAG Query
```python
%run llm.py
```

### Cell 9: (Optional) Interactive Query
```python
from langchain_ollama import OllamaLLM
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

llm = OllamaLLM(model="qwen3.5:4b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vec_db = Milvus(
    embeddings,
    connection_args={"uri": "data/milvus_bank.db"},
    collection_name="bank_knowledge",
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=vec_db.as_retriever(search_kwargs={"k": 3}),
)

# Ask your own question:
query = "What is Little Champs Account?"
result = qa_chain.invoke({"query": query})
print(result)
```

### Important Colab Notes
- Colab free tier has limited RAM — the 4B Qwen model should fit but performance may be slow
- If Ollama crashes, restart the runtime and re-run the `ollama serve` cell
- All uploaded files are lost when the runtime disconnects; use Google Drive mount for persistence
- The embedding step (Step 2, embedder.py) downloads the `all-MiniLM-L6-v2` model (~80MB) the first time

---

## File Summary Table

| File | Role | Run Order |
|------|------|-----------|
| `data/inspect_data.py` | Inspect Excel structure (optional) | -- |
| `data/format_for_finetuning.py` | Extract Q&A from Excel | 1st |
| `embedder.py` | Build raw FAISS index + metadata | 2nd |
| `embedder_2.py` | Build LangChain vector store | 3rd |
| `llm.py` | RAG inference with Qwen 3.5 | 4th |
| `search.py` | Direct similarity search (standalone) | Anytime after 2nd |
