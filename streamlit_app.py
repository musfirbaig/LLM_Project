"""Streamlit interface for bank QA + dataset upload."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from llm import ask

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_QA_PATH = PROJECT_ROOT / "data" / "all_qa_pairs.json"


REQUIRED_KEYS = {"question", "answer"}


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
    if not isinstance(payload, list):
        raise ValueError("Uploaded JSON must be a list of QA objects")

    clean: list[dict[str, str]] = []
    for idx, rec in enumerate(payload, start=1):
        if not isinstance(rec, dict):
            raise ValueError(f"Entry {idx} is not a JSON object")
        if not REQUIRED_KEYS.issubset(rec.keys()):
            raise ValueError(f"Entry {idx} must include keys: question, answer")

        q = str(rec.get("question", "")).strip()
        a = str(rec.get("answer", "")).strip()
        if not q or not a:
            raise ValueError(f"Entry {idx} has empty question/answer")

        clean.append(
            {
                "question": q,
                "answer": a,
                "product": str(rec.get("product", "Uploaded Document")).strip() or "Uploaded Document",
                "sheet": str(rec.get("sheet", "uploaded_json")).strip() or "uploaded_json",
            }
        )
    return clean


def _rebuild_indexes() -> tuple[bool, str]:
    try:
        subprocess.run([sys.executable, "embedder.py"], cwd=str(PROJECT_ROOT), check=True)
        subprocess.run([sys.executable, "embedder_2.py"], cwd=str(PROJECT_ROOT), check=True)
        return True, "Index rebuild complete. New data is now available for queries."
    except subprocess.CalledProcessError as exc:
        return False, f"Rebuild failed: {exc}"


def _render_query_tab() -> None:
    st.subheader("Ask NUST Bank Assistant")
    query = st.text_input("Enter your question", placeholder="How do I open a savings account?")

    if st.button("Get Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Retrieving and generating answer..."):
            result = ask(query)

        st.markdown("### Response")
        st.write(result.get("answer", "No answer returned."))

        confidence = result.get("confidence")
        if confidence is not None:
            st.caption(f"Retrieval confidence: {confidence:.3f}")

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for idx, src in enumerate(sources, start=1):
                st.write(f"{idx}. [{src.get('product', 'N/A')}] {src.get('question', '')}")


def _render_upload_tab() -> None:
    st.subheader("Upload New QA Data (JSON)")
    st.caption("Expected format: a JSON array of objects with question and answer fields.")

    uploaded = st.file_uploader("Choose a JSON file", type=["json"])
    if uploaded is None:
        return

    if st.button("Ingest and Rebuild Index", type="primary"):
        try:
            payload = json.loads(uploaded.read().decode("utf-8"))
            new_rows = _normalize_new_records(payload)

            existing = _load_all_qa()
            existing.extend(new_rows)
            _save_all_qa(existing)

            ok, msg = _rebuild_indexes()
            if ok:
                st.success(f"Added {len(new_rows)} records. {msg}")
            else:
                st.error(msg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="NUST Bank RAG Assistant", page_icon="bank", layout="wide")
    st.title("NUST Bank Product Knowledge Assistant")
    st.caption("RAG-powered QA with upload-to-index workflow")

    tab_query, tab_upload = st.tabs(["Ask Questions", "Upload Data"])
    with tab_query:
        _render_query_tab()
    with tab_upload:
        _render_upload_tab()


if __name__ == "__main__":
    main()
