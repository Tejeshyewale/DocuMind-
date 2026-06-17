# DocuMind — RAG Document Q&A

Upload a PDF, ask questions in plain language, get answers grounded in the
document itself (with the source page numbers) instead of a generic LLM guess.

**Stack:** Python · LangChain · Gemini API · FAISS · Streamlit

## How it works

```
PDF Upload
   ↓
PyPDFLoader (extract text per page)
   ↓
RecursiveCharacterTextSplitter (chunk into ~1000-char pieces)
   ↓
GoogleGenerativeAIEmbeddings (chunk -> vector)
   ↓
FAISS (store + similarity search)
   ↓
User question -> top-k relevant chunks retrieved
   ↓
Gemini chat model answers using ONLY those chunks as context
   ↓
Answer + source page numbers shown in Streamlit
```

This is Retrieval-Augmented Generation (RAG): instead of asking the LLM to
answer from memory, you hand it the relevant slice of *your* document, which
cuts down hallucination and lets you cite a source page.

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Gemini API key:** https://aistudio.google.com/apikey

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.

## Folder structure

```
DocuMind/
├── app.py                  # Streamlit UI — wires everything together
├── requirements.txt
├── .env.example
├── utils/
│   ├── pdf_loader.py        # Step: PDF -> text per page
│   ├── embeddings.py         # Step: text -> vectors (Gemini)
│   ├── vector_store.py       # Step: chunking + FAISS build/load
│   └── rag_pipeline.py       # Step: retrieve chunks + ask Gemini
├── data/                    # (optional) put sample PDFs here
└── faiss_index/             # FAISS index gets saved here after processing
```

## A note on model names

Google renames/retires Gemini model IDs every few months. This project reads
the chat and embedding model names from `.env` (`GEMINI_CHAT_MODEL`,
`GEMINI_EMBEDDING_MODEL`) instead of hardcoding them, so if a model gets
deprecated you only need to update `.env` — check current options at
https://ai.google.dev/gemini-api/docs/models, no code change needed.

## Roadmap (good next features for your resume)

- [ ] Multiple PDF upload + cross-document search
- [ ] Persist FAISS index per-document so you don't re-embed on every restart
- [ ] Move the RAG logic behind a FastAPI backend, keep Streamlit as the client
- [ ] Add simple auth (so it's not a fully open demo)
- [ ] Containerize with Docker, deploy to Render/AWS
- [ ] Evaluate retrieval quality with RAGAS to back up "accuracy" claims on your resume
