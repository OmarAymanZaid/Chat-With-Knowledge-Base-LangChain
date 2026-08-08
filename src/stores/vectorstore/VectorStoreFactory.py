from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from loguru import logger

from src.helpers.configs import Settings
from src.enums.enums import VectorStoreProvider


class VectorStoreFactory:

    def __init__(self, config: Settings):
        self.config = config

    def create_vectorstore(
        self,
        embedding_model: Embeddings,
        provider: str | None = None,
        collection_name: str | None = None,
    ) -> VectorStore:
        provider = provider or self.config.VECTORSTORE_PROVIDER
        collection = collection_name or self.config.VECTORSTORE_COLLECTION_NAME

        if provider == VectorStoreProvider.CHROMA.value:
            return Chroma(
                collection_name=collection,
                embedding_function=embedding_model,
                persist_directory=self.config.VECTORSTORE_PERSIST_DIR,
            )

        logger.error(f"Unsupported VectorStore provider: {provider}")
        raise ValueError(f"Provider '{provider}' is not supported.")
