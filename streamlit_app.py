"""
Streamlit Interface — NUST Bank RAG Assistant
-----------------------------------------------
Three-tab UI:
  1. Chat-style Q&A powered by the fine-tuned Qwen 3.5 + LoRA model
  2. Bulk JSON upload for new QA data
  3. Manual single-entry Q&A form

Sidebar shows model status, device info, and knowledge-base statistics.
All new data is instantly indexed and available for queries (Requirement #6).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from llm import ask, load_model, get_device_info

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_QA_PATH  = PROJECT_ROOT / "data" / "all_qa_pairs.json"
REQUIRED_KEYS = {"question", "answer"}


# ── Data helpers ─────────────────────────────────────────────────────────

def _load_all_qa() -> list[dict[str, Any]]:
    if not ALL_QA_PATH.exists():
        return []
    with ALL_QA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_all_qa(records: list[dict[str, Any]]) -> None:
    ALL_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ALL_QA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


def _normalize_new_records(payload: Any) -> list[dict[str, str]]:
    """Validate and normalise a list of QA dicts from uploaded JSON."""
    if not isinstance(payload, list):
        raise ValueError("Uploaded JSON must be a list of QA objects.")

    clean: list[dict[str, str]] = []
    for idx, rec in enumerate(payload, start=1):
        if not isinstance(rec, dict):
            raise ValueError(f"Entry {idx} is not a JSON object.")
        if not REQUIRED_KEYS.issubset(rec.keys()):
            raise ValueError(f"Entry {idx} must include keys: question, answer.")

        q = str(rec.get("question", "")).strip()
        a = str(rec.get("answer", "")).strip()
        if not q or not a:
            raise ValueError(f"Entry {idx} has empty question / answer.")

        clean.append(
            {
                "question": q,
                "answer": a,
                "product": str(rec.get("product", "Uploaded Document")).strip()
                    or "Uploaded Document",
                "sheet": str(rec.get("sheet", "uploaded_json")).strip()
                    or "uploaded_json",
            }
        )
    return clean


def _rebuild_indexes() -> tuple[bool, str]:
    """Re-run both embedding pipelines so new data becomes searchable."""
    try:
        subprocess.run(
            [sys.executable, "embedder.py"],
            cwd=str(PROJECT_ROOT), check=True, capture_output=True,
        )
        subprocess.run(
            [sys.executable, "embedder_2.py"],
            cwd=str(PROJECT_ROOT), check=True, capture_output=True,
        )
        return True, "Index rebuild complete — new data is now available for queries."
    except subprocess.CalledProcessError as exc:
        return False, f"Rebuild failed: {exc}"


# ── Cached model warm-up ────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading fine-tuned model (first time may take 1-2 min) …")
def _warm_model():
    """Pre-load model + tokenizer on first Streamlit run."""
    return load_model()


# ── Sidebar ──────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🏦 NUST Bank Assistant")
        st.markdown("---")

        # ── Model info ──
        st.markdown("### ⚙️ Model")
        st.markdown("**Qwen 3.5-4B + Banking LoRA**")
        st.info(f"**Device:** {get_device_info()}")

        # ── Knowledge base stats ──
        st.markdown("---")
        st.markdown("### 📊 Knowledge Base")
        qa_data = _load_all_qa()
        col1, col2 = st.columns(2)
        col1.metric("Q&A Pairs", len(qa_data))
        products = sorted({r.get("product", "Unknown") for r in qa_data})
        col2.metric("Products", len(products))

        if products:
            with st.expander("View products"):
                for p in products:
                    st.write(f"• {p}")

        # ── Quick tips ──
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.caption(
            "• Ask about any NUST Bank product.\n"
            "• Upload new Q&A JSON or add entries manually — they're indexed instantly.\n"
            "• Guard-rails block harmful / off-topic queries automatically."
        )


# ── Tab 1: Chat Q&A ─────────────────────────────────────────────────────

def _render_query_tab() -> None:
    st.subheader("💬 Ask a Question")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.write(
                            f"{i}. **[{src.get('product', 'N/A')}]** "
                            f"{src.get('question', '')}"
                        )
            if msg.get("confidence") is not None:
                st.caption(f"Retrieval confidence: {msg['confidence']:.3f}")

    # Chat input
    if query := st.chat_input("e.g.  How do I open a savings account?"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context & generating answer …"):
                result = ask(query)

            answer = result.get("answer", "No answer returned.")
            st.markdown(answer)

            if result.get("guardrail_blocked"):
                st.warning("This query was blocked by the safety guardrails.")

            confidence = result.get("confidence")
            if confidence is not None:
                st.caption(f"Retrieval confidence: {confidence:.3f}")

            sources = result.get("sources", [])
            if sources:
                with st.expander("📎 Sources"):
                    for i, src in enumerate(sources, 1):
                        st.write(
                            f"{i}. **[{src.get('product', 'N/A')}]** "
                            f"{src.get('question', '')}"
                        )

        # Persist in session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "confidence": confidence,
                "sources": sources,
            }
        )


# ── Tab 2: JSON bulk upload ─────────────────────────────────────────────

def _render_upload_tab() -> None:
    st.subheader("📄 Upload New QA Data (JSON)")
    st.caption(
        "Upload a JSON file containing an array of objects. "
        "Each object must have `question` and `answer` keys. "
        "Optional keys: `product`, `sheet`."
    )

    st.markdown(
        """
        **Example format:**
        ```json
        [
          {
            "question": "What is the interest rate on savings?",
            "answer": "The current interest rate is 5% per annum.",
            "product": "Savings Account"
          }
        ]
        ```
        """
    )

    uploaded = st.file_uploader("Choose a JSON file", type=["json"])
    if uploaded is None:
        return

    if st.button("🔄 Ingest & Rebuild Index", type="primary", key="btn_json_upload"):
        try:
            payload = json.loads(uploaded.read().decode("utf-8"))
            new_rows = _normalize_new_records(payload)

            existing = _load_all_qa()
            existing.extend(new_rows)
            _save_all_qa(existing)

            with st.spinner("Rebuilding vector index — please wait …"):
                ok, msg = _rebuild_indexes()

            if ok:
                st.success(f"✅ Added **{len(new_rows)}** record(s). {msg}")
                st.balloons()
            else:
                st.error(msg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")


# ── Tab 3: Manual single-entry form ─────────────────────────────────────

def _render_add_qa_tab() -> None:
    st.subheader("➕ Add a Single Q&A Entry")
    st.caption(
        "Add one knowledge entry at a time. "
        "The new entry will be indexed immediately and available for queries."
    )

    with st.form("add_qa_form", clear_on_submit=True):
        product = st.text_input(
            "Product / Category",
            placeholder="e.g., Savings Account, Home Loan, Credit Card",
        )
        question = st.text_area(
            "Question",
            placeholder="What is the minimum balance requirement?",
            height=80,
        )
        answer = st.text_area(
            "Answer",
            placeholder="The minimum balance for a Basic Savings Account is Rs. 1,000 …",
            height=120,
        )

        submitted = st.form_submit_button("➕ Add & Rebuild Index", type="primary")

        if submitted:
            q = question.strip()
            a = answer.strip()
            if not q or not a:
                st.error("Both **Question** and **Answer** are required.")
            else:
                new_record = {
                    "question": q,
                    "answer": a,
                    "product": product.strip() or "Manual Entry",
                    "sheet": "manual_entry",
                }

                existing = _load_all_qa()
                existing.append(new_record)
                _save_all_qa(existing)

                with st.spinner("Rebuilding vector index …"):
                    ok, msg = _rebuild_indexes()

                if ok:
                    st.success(f"✅ Q&A pair added! {msg}")
                    st.balloons()
                else:
                    st.error(msg)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NUST Bank RAG Assistant",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 NUST Bank Product Knowledge Assistant")
    st.caption(
        "RAG-powered Q&A with a fine-tuned Qwen 3.5 model  ·  "
        "Upload or add documents for instant knowledge updates"
    )

    # Pre-load the model (cached after first call)
    _warm_model()

    # Sidebar
    _render_sidebar()

    # Main content
    tab_query, tab_upload, tab_add = st.tabs(
        ["💬 Ask Questions", "📄 Upload Data", "➕ Add Q&A Pair"]
    )

    with tab_query:
        _render_query_tab()
    with tab_upload:
        _render_upload_tab()
    with tab_add:
        _render_add_qa_tab()


if __name__ == "__main__":
    main()
