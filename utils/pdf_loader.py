"""
Step 2 of the RAG pipeline: load a PDF and turn it into LangChain Document
objects (one per page), each carrying page_content + metadata (page number and source).
"""

import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str, original_filename: str = None):
    """
    Load a PDF from disk and return a list of Document objects,
    one per page, with metadata['page'] set.
    If original_filename is provided, metadata['source'] is set to it.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Clean up the metadata['source'] so citations are user-friendly
    target_source = original_filename if original_filename else os.path.basename(file_path)
    for page in pages:
        page.metadata["source"] = target_source
        
    return pages
