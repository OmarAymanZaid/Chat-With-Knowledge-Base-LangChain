import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from loguru import logger

def load_document(file_path: str) -> list[Document]:
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found at {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8").load()
    else:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Extension '{ext}' is not supported.")
