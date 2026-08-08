from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from src.features.generation.prompts import get_rag_prompt


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Assembles the LCEL RAG chain via Dependency Injection."""
    logger.info("Assembling LCEL RAG chain...")

    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
