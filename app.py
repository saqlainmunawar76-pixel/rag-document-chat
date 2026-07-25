"""
Chat With Your Document — RAG Mini-Project
Generative AI & Prompt Engineering Internship — NeuroFive Solutions

A polished Streamlit app that lets you upload any PDF (resume, class notes,
report) and ask questions grounded ONLY in that document's actual content,
with an optional side-by-side comparison against a plain (non-grounded) prompt.
Includes Q&A history and a downloadable results file.
"""

import streamlit as st
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
import io
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat With Your Document | NeuroFive RAG",
    page_icon="📄",
    layout="centered",
)

# ── Custom styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0A0F1F 0%, #121A33 100%);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #5EEAD4, #7DD3FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9AA7C7 !important;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #1B2440;
        border: 1px solid #2A3760;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.85rem;
        color: #C7D0E8 !important;
        margin-top: 8px;
        line-height: 1.6;
    }
    .answer-box {
        background-color: #16233F;
        border-left: 4px solid #5EEAD4;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 12px;
        color: #F3F6FC !important;
        font-size: 1rem;
        line-height: 1.65;
    }
    .plain-answer-box {
        background-color: #201B33;
        border-left: 4px solid #F5A623;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 12px;
        color: #F3F6FC !important;
        font-size: 1rem;
        line-height: 1.65;
    }
    .answer-box b, .plain-answer-box b, .answer-box strong, .plain-answer-box strong {
        color: #5EEAD4 !important;
    }
    .section-label {
        color: #E8ECF4 !important;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .history-q {
        color: #7DD3FC !important;
        font-weight: 600;
    }
    .history-a {
        color: #C7D0E8 !important;
        font-size: 0.9rem;
    }
    .stTextInput input {
        color: #F3F6FC !important;
        background-color: #161F3D !important;
        border: 1px solid #2A3760 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1424 0%, #161F3D 100%) !important;
        border-right: 1px solid #2A3760;
    }
    section[data-testid="stSidebar"] * {
        color: #E8ECF4 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #5EEAD4 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2A3760 !important;
    }
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] small {
        color: #7C89AC !important;
    }

    /* ── Expander (history items) ───────────────────────────── */
    section[data-testid="stSidebar"] details {
        background-color: #1B2440 !important;
        border: 1px solid #2A3760 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] summary {
        color: #E8ECF4 !important;
    }

    /* ── File uploader ───────────────────────────────────────── */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #161F3D !important;
        border: 1.5px dashed #2A3760 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #C7D0E8 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #161F3D !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #1B2440 !important;
        color: #F3F6FC !important;
        border: 1px solid #2A3760 !important;
    }

    /* ── Uploaded file chip ─────────────────────────────────── */
    [data-testid="stFileUploaderFile"] {
        background-color: #1B2440 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: #F3F6FC !important;
    }

    /* ── Toggles / labels in main area ──────────────────────── */
    .stApp label, .stApp p, .stApp span {
        color: #E8ECF4;
    }
    [data-testid="stWidgetLabel"] p {
        color: #C7D0E8 !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stDownloadButton button, .stButton button[kind="secondary"] {
        background-color: #1B2440 !important;
        color: #F3F6FC !important;
        border: 1px solid #2A3760 !important;
    }

    /* ── Success / info banners ─────────────────────────────── */
    [data-testid="stAlert"] {
        background-color: #16233F !important;
        color: #F3F6FC !important;
    }
    [data-testid="stAlert"] * {
        color: #F3F6FC !important;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Backend setup ──────────────────────────────────────────────────
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)
GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RAG_SYSTEM_PROMPT = """
You are a document Q&A assistant. Answer the user's question using ONLY the 
provided context below. Do not use any outside knowledge.

If the answer is not clearly present in the context, say exactly: 
"I cannot find this in the document." Do not guess or make anything up.
Be concise and direct.
"""


def extract_text(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(texts: list[str]) -> np.ndarray:
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings])


def top_k_chunks(query: str, chunks: list[str], chunk_embeddings: np.ndarray, k: int = 3):
    query_embedding = embed([query])[0]
    similarities = chunk_embeddings @ query_embedding / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_indices]


def answer_with_rag(question: str, chunks, chunk_embeddings):
    relevant_chunks = top_k_chunks(question, chunks, chunk_embeddings, k=3)
    context = "\n---\n".join(relevant_chunks)
    full_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text, relevant_chunks


def answer_plain(question: str) -> str:
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def build_history_export(history: list[dict]) -> str:
    lines = [
        "# Chat With Your Document — Q&A Results",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Document: {st.session_state.get('filename', 'N/A')}",
        "",
        "---",
        "",
    ]
    for i, entry in enumerate(history, 1):
        lines.append(f"## Question {i}: {entry['question']}")
        lines.append("")
        lines.append(f"**Grounded (RAG) Answer:**  \n{entry['rag_answer']}")
        lines.append("")
        if entry.get("plain_answer"):
            lines.append(f"**Plain Prompt Answer:**  \n{entry['plain_answer']}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ── Session state init ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ How it works")
    st.markdown("""
    1. Your PDF is split into chunks
    2. Each chunk is embedded (Gemini embeddings)
    3. Your question is matched to the most relevant chunks (cosine similarity)
    4. The model answers **only** from those chunks
    """)
    st.markdown("---")

    st.markdown("### 🕘 Q&A History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{len(st.session_state.history) - i + 1}: {entry['question'][:40]}..."):
                st.markdown(f'<p class="history-q">Q: {entry["question"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="history-a">📄 {entry["rag_answer"]}</p>', unsafe_allow_html=True)
                if entry.get("plain_answer"):
                    st.markdown(f'<p class="history-a">🧠 {entry["plain_answer"]}</p>', unsafe_allow_html=True)

        st.markdown("---")
        export_text = build_history_export(st.session_state.history)
        st.download_button(
            "⬇️ Download all results (.md)",
            data=export_text,
            file_name="rag_qa_results.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No questions asked yet.")

    st.markdown("---")
    st.markdown("Built for the **Generative AI & Prompt Engineering Internship** @ NeuroFive Solutions")

# ── Main content ───────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Chat With Your Document</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a resume, report, or notes — ask questions grounded in the actual content. Built on Retrieval-Augmented Generation (RAG).</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF (resume, class notes, or report — 3-10 pages)", type="pdf")

if uploaded_file:
    if "chunks" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Reading and indexing document..."):
            raw_text = extract_text(io.BytesIO(uploaded_file.read()))
            chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
            chunk_embeddings = embed(chunks)
            st.session_state.chunks = chunks
            st.session_state.chunk_embeddings = chunk_embeddings
            st.session_state.filename = uploaded_file.name
        st.success(f"✅ Indexed {len(st.session_state.chunks)} chunks from **{uploaded_file.name}**")

    col1, col2 = st.columns(2)
    with col1:
        compare_mode = st.toggle("Compare with plain prompt", value=True)
    with col2:
        show_sources = st.toggle("Show retrieved sources", value=False)

    question = st.text_input("Ask a question about the document:", placeholder="e.g. What skills are listed?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving relevant sections and answering..."):
            rag_answer, sources = answer_with_rag(question, st.session_state.chunks, st.session_state.chunk_embeddings)

        plain_answer = None
        if compare_mode:
            with st.spinner("Getting plain prompt answer for comparison..."):
                plain_answer = answer_plain(question)

        st.session_state.history.append({
            "question": question,
            "rag_answer": rag_answer,
            "plain_answer": plain_answer,
        })

        st.markdown('<p class="section-label">📄 Grounded Answer (from your document)</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{rag_answer}</div>', unsafe_allow_html=True)

        if show_sources:
            st.markdown('<div class="source-box"><b>Retrieved chunks used:</b><br><br>' + "<br><br>".join(sources) + '</div>', unsafe_allow_html=True)

        if compare_mode:
            st.markdown('<p class="section-label">🧠 Plain Prompt Answer (model\'s own knowledge, no document)</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="plain-answer-box">{plain_answer}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Upload a PDF to get started.")
