from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from loguru import logger

from src.helpers.configs import Settings


def build_retriever(
    vectorstore: VectorStore,
    config: Settings,
    search_type: str | None = None,
    k: int | None = None,
) -> VectorStoreRetriever:
    """Builds and configures a LangChain VectorStoreRetriever from an injected VectorStore instance."""
    stype = search_type or getattr(config, "RETRIEVAL_SEARCH_TYPE", "similarity")
    top_k = k or getattr(config, "RETRIEVAL_K", 4)

    logger.info(
        f"Initializing retriever with search_type='{stype}' and top_k={top_k}"
    )

    return vectorstore.as_retriever(
        search_type=stype,
        search_kwargs={"k": top_k},
    )
