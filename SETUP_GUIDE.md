# NUST Bank RAG System — Setup & Run Guide

Complete instructions for running the NUST Bank QA system **locally** (CPU) and on **Google Colab** (GPU).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Setup (CPU)](#local-setup-cpu)
4. [Google Colab Setup (GPU)](#google-colab-setup-gpu)
5. [Using the Application](#using-the-application)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
NUST Bank-Product-Knowledge.xlsx
            |
            v
  format_for_finetuning.py
            |
            v
    all_qa_pairs.json
            |
            v
      embedder.py    ←  all-MiniLM-L6-v2 (embedding model)
            |
            v
    faiss_index.bin  +  chunk_metadata.json
                                   |
                                   v
                             embedder_2.py
                                   |
                                   v
                          data/milvus_bank.db
                         (bank_knowledge collection)
                                   |
                                   v
              ┌────────────────────┴────────────────────┐
              │            llm.py                       │
              │   Qwen 3.5-4B + Banking LoRA            │
              │   (native HuggingFace / PEFT)           │
              └────────────────────┬────────────────────┘
                                   |
                                   v
                         streamlit_app.py
                         (Web UI on localhost)
```

**Key change**: The system now uses the **fine-tuned Qwen 3.5-4B + LoRA adapter** loaded
directly via HuggingFace `transformers` + `peft`. Ollama is **no longer required**.

| Component         | Technology                          |
|-------------------|-------------------------------------|
| Embedding Model   | `all-MiniLM-L6-v2` (384-dim)       |
| Vector Store      | Milvus Lite (single `.db` file)     |
| Generative LLM    | `Qwen/Qwen3.5-4B` + LoRA adapter   |
| LLM Runtime       | HuggingFace Transformers + PEFT     |
| Web UI            | Streamlit                           |
| Guardrails        | Custom regex + post-filter          |

**Hardware auto-detection:**

| Environment | dtype    | Quantisation        | RAM / VRAM required |
|-------------|----------|---------------------|---------------------|
| GPU (CUDA)  | float16  | 4-bit (bitsandbytes)| ~5-6 GB VRAM        |
| CPU only    | bfloat16 | None                | ~8-10 GB RAM        |

---

## Prerequisites

- **Python 3.10+**
- **Git** (for cloning the repository)
- **~15 GB disk space** (model weights + vector stores)
- **8+ GB RAM** (for CPU inference)

The base model (`Qwen/Qwen3.5-4B`) is downloaded automatically from HuggingFace
on first run (~8 GB, cached in `~/.cache/huggingface/`).

---

## Local Setup (CPU)

> **Note:** CPU inference works but is slower (~30-60 seconds per answer). Good for
> development and testing.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LLM_Project.git
cd LLM_Project
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On Windows without a GPU, `torch` will install the CPU-only version
> automatically. This is correct — no need for CUDA.

### Step 4 — Build the Knowledge Base (first time only)

If you **don't already** have `data/all_qa_pairs.json`, `data/faiss_index.bin`,
`data/chunk_metadata.json`, and `data/milvus_bank.db`, run the data pipeline:

```bash
# Step 4a — Extract Q&A pairs from Excel
python data/format_for_finetuning.py

# Step 4b — Build FAISS index + chunk metadata
python embedder.py

# Step 4c — Build Milvus Lite vector store
python embedder_2.py
```

If these files already exist in `data/`, you can **skip this step**.

### Step 5 — Test the Model (optional)

```bash
python llm.py
```

This will:
1. Download the base model `Qwen/Qwen3.5-4B` from HuggingFace (~8 GB, first time only)
2. Load the LoRA adapter from `qwen3.5_banking_lora/`
3. Ask "how do i open an account?" and print the answer

Expected output:
```
[llm] Loading base model  Qwen/Qwen3.5-4B  (CPU · bfloat16) …
[llm] Merging LoRA adapter from  qwen3.5_banking_lora  …
[llm] Model ready in 45.2s

--- Query: 'how do i open an account?' ---

Answer : To open an account at NUST Bank, you can visit ...
```

### Step 6 — Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**. On first launch, the model loads
(~1-2 minutes on CPU). Subsequent queries are faster.

---

## Google Colab Setup (GPU + ngrok API)

> **Recommended:** Use a Colab runtime with **GPU** (T4 or better) for fast inference
> (~2-5 seconds per answer with 4-bit quantisation).

This mode runs the model on Colab, exposes it through ngrok, and lets your local
pipeline call the remote API through `NUST_BANK_REMOTE_LLM_URL`.

### Step 1 — Open a New Colab Notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

**Change runtime to GPU:** `Runtime → Change runtime type → T4 GPU`

### Step 2 — Upload Project Files

Upload these files/folders to Colab's `/content/` directory:

```
/content/
├── qwen3.5_banking_lora/    ← entire folder (adapter weights)
├── data/
│   ├── all_qa_pairs.json
│   ├── chunk_metadata.json
│   ├── faiss_index.bin
│   └── milvus_bank.db
├── llm.py
├── guardrails.py
├── embedder.py
├── embedder_2.py
├── search.py
├── streamlit_app.py
├── requirements.txt
├── NUST Bank-Product-Knowledge.xlsx
└── funds_transer_app_features_faq.json
```

**Option A — Upload from local machine:**
```python
from google.colab import files
# Upload one file at a time, or zip and upload
uploaded = files.upload()
```

**Option B — Upload via Google Drive (recommended for large files):**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to working directory
!cp -r /content/drive/MyDrive/LLM_Project/* /content/
```

**Option C — Clone from GitHub:**
```python
!git clone https://github.com/YOUR_USERNAME/LLM_Project.git
%cd LLM_Project
```

### Step 3 — Install Dependencies

```python
%%bash
pip install -q -r requirements.txt

# Also install bitsandbytes for GPU 4-bit quantisation
pip install -q "bitsandbytes>=0.43.1"

# Install ngrok helper for the public tunnel
pip install -q pyngrok
```

### Step 4 — Build Knowledge Base (if not uploaded)

```python
import os
os.makedirs('data', exist_ok=True)

# Only run these if data files don't already exist
!python data/format_for_finetuning.py
!python embedder.py
!python embedder_2.py
```

### Step 5 — Start the Remote API Server

Run the HTTP server in one Colab cell:

```python
import subprocess
import sys

process = subprocess.Popen(
  [sys.executable, "remote_llm_server.py", "--host", "0.0.0.0", "--port", "8000"],
  stdout=subprocess.PIPE,
  stderr=subprocess.STDOUT,
  text=True,
)
print("Remote server started on port 8000")
```

### Step 6 — Expose the Server with ngrok

```python
from pyngrok import ngrok

public_url = ngrok.connect(8000).public_url
print(public_url)
```

Keep this URL; it becomes your local pipeline endpoint.

### Step 7 — Configure the Local App

Set this environment variable on your local machine before starting Streamlit or your pipeline:

```bash
set NUST_BANK_REMOTE_LLM_URL=https://xxxx.ngrok-free.app
```

Optional security token, if you set `NUST_BANK_REMOTE_LLM_TOKEN` on Colab:

```bash
set NUST_BANK_REMOTE_LLM_TOKEN=your_shared_secret
```

### Step 8 — Test Inference

```python
from llm import ask

result = ask("How do I open a savings account?")
print("Answer:", result["answer"])
print("Confidence:", result.get("confidence"))
print("Sources:", result.get("sources"))
```

If the environment variable is set correctly, `llm.py` will retrieve context locally,
send the prompt to Colab, and return the generated answer without loading the model on your laptop.

Expected output (with GPU):
```
[llm] Loading base model  Qwen/Qwen3.5-4B  (GPU (Tesla T4) · 4-bit quantised) …
[llm] Merging LoRA adapter from  qwen3.5_banking_lora  …
[llm] Model ready in 12.3s

Answer: To open a savings account at NUST Bank ...
```

### Step 6 — Run Streamlit on Colab (Optional)

Streamlit doesn't natively run on Colab, but you can use a tunnel:

```python
# Install localtunnel
!npm install -g localtunnel

# Start Streamlit in background
import subprocess
proc = subprocess.Popen(
    ['streamlit', 'run', 'streamlit_app.py', '--server.port', '8501'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Create tunnel
import time
time.sleep(5)
!npx localtunnel --port 8501
```

This will print a public URL (e.g., `https://xyz.loca.lt`) — open it in your browser.

> **Note:** The tunnel URL changes each time. Enter the Colab IP when prompted by
> localtunnel.

### Alternative: Direct Python Querying on Colab

If you don't need the Streamlit UI, you can run queries directly:

```python
from llm import ask

# Interactive query loop
while True:
    q = input("Ask: ")
    if q.lower() in ("exit", "quit"):
        break
    result = ask(q)
    print(f"\n{result['answer']}\n")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print("---")
```

---

## Using the Application

### Tab 1: Ask Questions 💬

Type any question about NUST Bank products in the chat input. The system:
1. Checks the query against **guardrails** (blocks harmful / injection attempts)
2. **Retrieves** the top-3 most relevant Q&A chunks from Milvus
3. **Generates** an answer using the fine-tuned Qwen 3.5 model
4. **Post-filters** the answer for safety

### Tab 2: Upload Data 📄

Upload a JSON file containing new Q&A pairs to bulk-add to the knowledge base:

```json
[
  {
    "question": "What fees apply to international transfers?",
    "answer": "International wire transfers incur a fee of Rs. 2,500 per transaction.",
    "product": "Funds Transfer"
  },
  {
    "question": "Can I set up recurring payments?",
    "answer": "Yes, you can set up standing instructions through Internet Banking.",
    "product": "Internet Banking"
  }
]
```

After upload, the FAISS index and Milvus vector store are **automatically rebuilt**.
New entries are immediately searchable.

### Tab 3: Add Q&A Pair ➕

Manually add a single Q&A entry via a form. Fill in:
- **Product / Category** — e.g., "Home Loan", "Savings Account"
- **Question** — The customer question
- **Answer** — The bank's official answer

Click **Add & Rebuild Index** — the entry is saved and indexed instantly.

### Real-Time Updates (Requirement #6)

Both Tab 2 and Tab 3 support **real-time knowledge updates**:

1. New data is appended to `data/all_qa_pairs.json`
2. `embedder.py` re-generates the FAISS index + metadata
3. `embedder_2.py` re-builds the Milvus vector store
4. The next query immediately uses the updated knowledge base

No restart required — new information is available for the very next query.

---

## Troubleshooting

### Model download is slow

The base model `Qwen/Qwen3.5-4B` is ~8 GB. First download may take 10-30 minutes
depending on your connection. The model is cached in `~/.cache/huggingface/` so
subsequent runs are instant.

### Out of memory on CPU

If you get OOM errors on CPU, try:
1. Close other applications to free RAM
2. Reduce `MAX_NEW_TOKENS` in `llm.py` (e.g., from 512 to 256)
3. Ensure you have at least 10 GB free RAM

### `bitsandbytes` error on CPU / Windows

This is expected — `bitsandbytes` is only for GPU quantisation. The code auto-detects
and skips it when no GPU is available. You do not need to install it on CPU.

### Streamlit shows "Loading model" for a long time

First model load takes 1-2 minutes on CPU (30s on GPU). This happens only once —
subsequent queries reuse the cached model.

### CUDA out of memory on Colab

If the T4 GPU runs out of VRAM:
1. Restart the runtime (`Runtime → Restart runtime`)
2. Make sure no other notebooks are using the GPU
3. The 4-bit quantisation should fit in ~5 GB VRAM

### Import errors

Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

For GPU environments, also install:
```bash
pip install bitsandbytes>=0.43.1
```

### Vector store not found

If you see errors about `milvus_bank.db`, run the data pipeline:
```bash
python data/format_for_finetuning.py
python embedder.py
python embedder_2.py
```
