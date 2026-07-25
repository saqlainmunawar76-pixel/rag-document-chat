# 📄 Chat With Your Document — RAG Mini-Project

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF — a resume, class notes, or a report — and ask questions that are answered **strictly from that document's content**, instead of the model guessing from its training data.

Built for the **Generative AI & Prompt Engineering Internship** at [NeuroFive Solutions](https://neurofivesolutions.com).

🔗 **Live demo:** https://saqlain-rag-document-chat.streamlit.app

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

Tested using a sample document on AI automation and RAG concepts. Five questions were asked directly against the app, with "Compare with plain prompt" enabled to see both answers side by side.

| # | Question | RAG Answer (grounded) | Plain Prompt Answer (no document) | Hallucination Flagged? |
|---|----------|------------------------|-------------------------------------|--------------------------|
| 1 | What is Retrieval-Augmented Generation (RAG)? | "RAG is a process that combines document retrieval with large language models." (short, matches document's exact definition) | Gave a full generic explanation with an "open-book exam" analogy, a 4-step breakdown, and use cases — accurate about RAG in general, but not sourced from *this* document | No — both answers were factually correct; the plain answer just wasn't grounded in the specific document |
| 2 | What are embeddings? | "Embeddings are used to convert text into vectors. They are stored in a vector database to enable semantic search." (matches document's exact phrasing) | Gave a long generic explanation with a "concept space" analogy and multiple embedding types — correct in general, but far more generic than what's actually in the document | No — no false claims, but the plain answer diverged from the document's specific framing |
| 3 | How does AI automation work? | "AI automation works by connecting various tools to automate workflows. Specifically, it integrates: n8n, APIs, Large Language Models (LLMs)." (correctly pulled the specific tools named in the document) | Gave a generic 5-step AI automation framework (data ingestion, perception, decision-making, action, feedback loop) with no mention of n8n or the document's specific integrations | **Yes, partially** — the plain answer's 5-step framework is not hallucinated as false, but it invents a structure never mentioned in the document and omits the specific tools (n8n, APIs) that were the actual answer |
| 4 | What is Retrieval-Augmented Generation (RAG)? *(repeated)* | Same short, document-grounded answer returned consistently | Plain answer varied in wording/structure across repeat runs (different analogies each time) | No |
| 5 | What are embeddings? *(repeated)* | Same short, document-grounded answer returned consistently | Plain answer again varied in structure and examples used | No |

### Summary: How grounding changed answer quality

The RAG-grounded answers were consistently short, precise, and matched the document's actual wording almost verbatim — the same question asked twice returned the same answer both times. The plain prompt answers, by contrast, were longer and generally accurate about the underlying concepts (RAG, embeddings, AI automation are well-known topics the model already knows), but they were **not actually sourced from the document** — they varied in structure between repeat runs and, in the "AI automation" case, presented a generic 5-step framework instead of the document's specific answer (n8n + APIs + LLMs). This is the core value of RAG in practice: for well-known concepts a plain prompt can sound convincing without ever having read your document, which is risky for anything document-specific (a company's actual process, a specific number, a specific policy) — RAG guarantees the answer is traceable back to the source content instead of the model's general training knowledge.

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
