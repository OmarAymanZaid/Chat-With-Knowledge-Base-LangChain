from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are a helpful and precise assistant for question-answering tasks.
Use the following retrieved context to answer the question. 
If you do not know the answer based on the context, say that you don't know. 
Keep the answer concise and accurate.

Context:
{context}"""

def get_rag_prompt() -> ChatPromptTemplate:
    """Returns the ChatPromptTemplate for RAG generation."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
