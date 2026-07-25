# 📄 Chat With Your Document — RAG Mini-Project

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF — a resume, class notes, or a report — and ask questions that are answered **strictly from that document's content**, instead of the model guessing from its training data.

Built for the **Generative AI & Prompt Engineering Internship** at [NeuroFive Solutions](https://neurofivesolutions.com).

🔗 **Live demo:** [Insert your Streamlit app link here after deploying]

---

## What This Solves

Large language models don't know your private data — your resume, your class notes, your company's internal report. Ask a plain LLM a question about a document it's never seen, and it will either say it can't help, or worse, **hallucinate** a plausible-sounding but false answer.

RAG fixes this by:
1. Breaking your document into small chunks
2. Converting each chunk into a numerical "embedding" (a vector that captures its meaning)
3. When you ask a question, embedding the question too, and finding the chunks most semantically similar to it
4. Feeding **only those relevant chunks** to the model as context, and instructing it to answer from that context alone — flagging when the answer isn't there instead of guessing

---

## How It Works (Architecture)

```
PDF Upload
    │
    ▼
Text Extraction (pypdf)
    │
    ▼
Chunking (500 chars, 50 char overlap — keeps context from being cut mid-sentence)
    │
    ▼
Embedding (Gemini text-embedding-004) — each chunk becomes a vector
    │
    ▼
Stored in-memory as a NumPy array (no vector database needed for this scale)
    │
    ▼
User asks a question
    │
    ▼
Question embedded → Cosine similarity search against all chunk vectors (NumPy)
    │
    ▼
Top 3 most relevant chunks retrieved
    │
    ▼
Gemini (gemini-2.5-flash) answers using ONLY those chunks as context
    │
    ▼
If the answer isn't in the retrieved context, the model says so explicitly
   instead of hallucinating
```

**Bonus feature:** a toggle lets you see the plain (non-grounded) answer side-by-side, so you can directly compare quality and spot hallucinations.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| LLM | `gemini-2.5-flash` (Gemini API) | Fast, free-tier friendly |
| Embeddings | `text-embedding-004` (Gemini API) | Same provider, no extra API key needed |
| Vector search | Plain NumPy (cosine similarity) | No external vector DB needed at this scale — simpler, zero extra dependencies |
| PDF parsing | `pypdf` | Lightweight, no external binaries |
| UI | Streamlit | Fast to build, easy to deploy |
| SDK | `google-genai` | Official current SDK (not the deprecated `google-generativeai`) |

---

## Running Locally

1. Clone this repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Get a free Gemini API key: https://aistudio.google.com/apikey

3. Create `.streamlit/secrets.toml` in the project folder:
```toml
GEMINI_API_KEY = "your_key_here"
```

4. Run:
```bash
streamlit run app.py
```

5. Open the local URL Streamlit gives you, upload a PDF, and start asking questions.

---

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. **New app** → select this repo → main file: `app.py`
4. Under **Advanced settings → Secrets**, add:
```toml
GEMINI_API_KEY = "your_key_here"
```
5. Deploy — you'll get a live public URL in a couple of minutes

---

## Test Results & Observations

*(Fill in after testing with your own document — resume, notes, or report)*

| # | Question | RAG Answer | Plain Prompt Answer | Hallucination Flagged? |
|---|----------|------------|----------------------|--------------------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Summary: How grounding changed answer quality

*(Write 3-4 sentences here after testing — e.g. did the plain prompt guess or refuse? Did RAG correctly say "I cannot find this in the document" for anything not actually present? Was the RAG answer more specific/accurate?)*

---

## Project Structure

```
rag-document-chat/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

**Submitted as part of the Generative AI & Prompt Engineering Internship at NeuroFive Solutions**
