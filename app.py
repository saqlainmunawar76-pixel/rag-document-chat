"""
Chat With Your Document — RAG Mini-Project
Generative AI & Prompt Engineering Internship — NeuroFive Solutions

A polished Streamlit app that lets you upload any PDF (resume, class notes,
report) and ask questions grounded ONLY in that document's actual content,
with an optional side-by-side comparison against a plain (non-grounded) prompt.
"""

import streamlit as st
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
import io

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
        color: #9AA7C7;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #161F3D;
        border: 1px solid #2A3760;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #9AA7C7;
        margin-top: 8px;
    }
    .answer-box {
        background-color: #161F3D;
        border-left: 3px solid #5EEAD4;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .plain-answer-box {
        background-color: #161F3D;
        border-left: 3px solid #F5A623;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
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


# ── UI ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Chat With Your Document</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a resume, report, or notes — ask questions grounded in the actual content. Built on Retrieval-Augmented Generation (RAG).</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ How it works")
    st.markdown("""
    1. Your PDF is split into chunks
    2. Each chunk is embedded (Gemini embeddings)
    3. Your question is matched to the most relevant chunks (cosine similarity)
    4. The model answers **only** from those chunks
    """)
    st.markdown("---")
    st.markdown("Built for the **Generative AI & Prompt Engineering Internship** @ NeuroFive Solutions")

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

    compare_mode = st.toggle("Also show a plain prompt answer (no document) for comparison", value=True)
    show_sources = st.toggle("Show retrieved source chunks", value=False)

    question = st.text_input("Ask a question about the document:", placeholder="e.g. What skills are listed?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving relevant sections and answering..."):
            rag_answer, sources = answer_with_rag(question, st.session_state.chunks, st.session_state.chunk_embeddings)

        st.markdown("#### 📄 Grounded Answer (from your document)")
        st.markdown(f'<div class="answer-box">{rag_answer}</div>', unsafe_allow_html=True)

        if show_sources:
            st.markdown('<div class="source-box"><b>Retrieved chunks used:</b><br>' + "<br><br>".join(sources) + '</div>', unsafe_allow_html=True)

        if compare_mode:
            with st.spinner("Getting plain prompt answer for comparison..."):
                plain_answer = answer_plain(question)
            st.markdown("#### 🧠 Plain Prompt Answer (model's own knowledge, no document)")
            st.markdown(f'<div class="plain-answer-box">{plain_answer}</div>', unsafe_allow_html=True)
else:
    st.info("👆 Upload a PDF to get started.")
