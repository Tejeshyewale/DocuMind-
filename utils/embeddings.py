"""
Step 4 of the RAG pipeline: turn text chunks into vectors using a local
Hugging Face embedding model. Model name is configurable via env var.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load a local HuggingFace embedding model.
    Configurable via HF_EMBEDDING_MODEL environment variable.
    """
    model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)
