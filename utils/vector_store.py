"""
Steps 3 and 5 of the RAG pipeline:
- split_documents: break pages into overlapping chunks (Step 3)
- build_vector_store / load_vector_store: FAISS index management (Step 5)
- merge_vector_stores: Merge multiple FAISS indexes for multi-document support
"""

import hashlib
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embedding_model


def calculate_md5(file_bytes: bytes) -> str:
    """Calculate the MD5 hash of file content to cache vector store indexes."""
    return hashlib.md5(file_bytes).hexdigest()


def get_faiss_index_path(file_hash: str) -> str:
    """Get the local storage directory for a specific file hash's index."""
    return os.path.join("faiss_index", file_hash)


def split_documents(pages, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split loaded pages into smaller overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(pages)


def build_vector_store(chunks, save_path: str):
    """Embed chunks and build a fresh FAISS index, persisted to disk."""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    return vector_store


def load_vector_store(save_path: str):
    """Load a previously saved FAISS index back from disk."""
    embeddings = get_embedding_model()
    return FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )


def merge_vector_stores(stores):
    """
    Merge a list of FAISS vector store instances into a single combined store.
    To avoid mutating the cached/in-memory individual vector stores,
    we safely clone the first vector store via serialization before merging the rest.
    """
    if not stores:
        return None
    
    combined = None
    for store in stores:
        if combined is None:
            try:
                # Clone the first index safely to avoid modifying the original cached store
                serialized = store.serialize_to_bytes()
                combined = FAISS.deserialize_from_bytes(
                    serialized,
                    get_embedding_model(),
                    allow_dangerous_deserialization=True
                )
            except Exception:
                # Fallback to direct merge if serialization is not supported
                combined = store
        else:
            combined.merge_from(store)
            
    return combined
