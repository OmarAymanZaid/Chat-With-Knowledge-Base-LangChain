from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "LangChain Engine"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    LOG_LEVEL: str = "INFO"

    # Telemetry
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "default-chain-engine"

    # Generation Specs
    GENERATION_PROVIDER: str = "OPENAI"
    GENERATION_MODEL_NAME: str = "gpt-4o-mini"
    GENERATION_DEFAULT_MAX_TOKENS: int = 200
    GENERATION_DEFAULT_TEMPERATURE: float = 0.1
    INPUT_DEFAULT_MAX_CHARACTERS: int = 1024

    # Embedding Specs
    EMBEDDING_PROVIDER: str = "COHERE"
    EMBEDDING_MODEL_NAME: str = "embed-multilingual-light-v3.0"
    EMBEDDING_MODEL_SIZE: int = 384

    # Keys and URLs
    OPENAI_API_KEY: str | None = None
    OPENAI_API_URL: str | None = None
    GOOGLE_API_KEY: str | None = None


    # VectorStore Specs
    VECTORSTORE_PROVIDER: str = "CHROMA"
    VECTORSTORE_COLLECTION_NAME: str = "rag_knowledge_base"
    VECTORSTORE_PERSIST_DIR: str = "src/assets/vectorstores"

    # Retrieval Specs
    RETRIEVAL_SEARCH_TYPE: str = "similarity"
    RETRIEVAL_K: int = 4

    @property
    def is_local(self) -> bool:
        return self.ENVIRONMENT == "local"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
