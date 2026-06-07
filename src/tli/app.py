"""Streamlit web UI for the Temporal Literature Investigator.

Launch:  tli web        (or:  streamlit run src/tli/app.py)

Ask a question and get a grounded answer from the same retrieval + synthesis
pipeline as the CLI. The answer pane shows Claude's response; an expander shows
exactly which books, profiles, and events were retrieved.
"""
from __future__ import annotations

import streamlit as st

from tli import config
from tli.ingest import load_books
from tli.rag import answer
from tli.store import get_collection

SAMPLES = [
    "How did 19th-century Russian turmoil shape Dostoevsky's novels?",
    "Which novels in the collection engage with World War II, and how?",
    "ادبیات آمریکای لاتین چگونه تحت تأثیر دیکتاتوری‌های نظامی قرار گرفت؟",
    "What was happening in Colombia around One Hundred Years of Solitude?",
]


@st.cache_data(show_spinner=False)
def _corpus_meta():
    return len(load_books())


@st.cache_data(show_spinner=False)
def _index_meta():
    try:
        coll = get_collection()
        return coll.count(), dict(coll.metadata or {})
    except Exception as e:  # noqa: BLE001
        return None, {"error": str(e)}


def _sources_panel(hits: list[dict]) -> None:
    order = {"book": 0, "profile": 1, "event": 2}
    labels = {"book": "📖 Book review", "profile": "🧭 Book profile",
              "event": "🏛️ Historical event"}
    for h in sorted(hits, key=lambda x: order.get(x["meta"].get("type"), 9)):
        m = h["meta"]
        sim = 1.0 - float(h["distance"])  # cosine distance -> similarity
        head = labels.get(m.get("type"), m.get("type", "?"))
        title = m.get("title") or m.get("slug") or ""
        meta_bits = " · ".join(str(x) for x in
                               [m.get("country"), m.get("year"), m.get("setting_place")]
                               if x)
        st.markdown(f"**{head}** — {title}  \n"
                    f"<span style='color:gray'>{meta_bits} · similarity {sim:.2f}"
                    f"{' · ' + m.get('slug') if m.get('slug') else ''}</span>",
                    unsafe_allow_html=True)
        st.caption(h["doc"][:400] + ("…" if len(h["doc"]) > 400 else ""))


def main() -> None:
    st.set_page_config(page_title="Temporal Literature Investigator",
                       page_icon="📚", layout="wide")
    st.title("📚 Temporal Literature Investigator")
    st.caption("Exploring history through literature — ask how events shaped the "
               "novels of an era, or vice versa.")

    n_books = _corpus_meta()
    n_chunks, idx_meta = _index_meta()

    with st.sidebar:
        st.subheader("Status")
        st.write(f"LLM backend: `{config.LLM_BACKEND}`")
        if n_chunks is None:
            st.error("Index not built. Run `tli index` first.")
        else:
            st.write(f"Index: **{n_chunks}** chunks")
            st.write(f"Embedder: `{idx_meta.get('embed_model', '?')}`")
        st.write(f"Corpus: {n_books} books")

    # Question form (only submits on button press).
    if "question" not in st.session_state:
        st.session_state.question = ""
    st.write("**Try one:**")
    cols = st.columns(len(SAMPLES))
    for col, s in zip(cols, SAMPLES):
        if col.button(s, use_container_width=True):
            st.session_state.question = s

    with st.form("ask"):
        question = st.text_area("Your question (English or Persian)",
                                value=st.session_state.question, height=90)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        if n_chunks is None:
            st.error("No index found — build it with `tli index`.")
            return
        with st.spinner("Retrieving and synthesizing…"):
            text, hits = answer(question.strip())
        st.markdown(text)
        with st.expander(f"🔍 Retrieved context ({len(hits)} chunks)"):
            _sources_panel(hits)


# `streamlit run` executes the module top-to-bottom (no __main__ guard needed).
main()
