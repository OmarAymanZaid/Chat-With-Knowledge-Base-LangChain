from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from loguru import logger

from src.features.ingestion.loaders import load_document
from src.features.ingestion.splitters import get_text_splitter
from src.helpers.configs import Settings


def run_ingestion(
    file_path: str,
    vectorstore: VectorStore,
    config: Settings,
) -> int:
    logger.info(f"Starting ingestion for file: {file_path}")

    # 1. Load document
    docs = load_document(file_path)

    # 2. Split document into chunks
    splitter = get_text_splitter(config)
    chunks = splitter.split_documents(docs)
    logger.info(
        f"Split {len(docs)} document page(s) into {len(chunks)} chunk(s)."
    )

    # 3. Add to the injected VectorStore instance directly
    vectorstore.add_documents(chunks)
    logger.info(f"Successfully ingested {len(chunks)} chunk(s) into Chroma.")

    return len(chunks)
