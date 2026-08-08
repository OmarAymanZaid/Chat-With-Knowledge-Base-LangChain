from typing import Any
import gradio as gr
from loguru import logger

from src.features.generation.chain import build_rag_chain
from src.features.ingestion.pipeline import run_ingestion
from src.features.retrieval.retriever import build_retriever
from src.helpers.configs import Settings
from src.stores.llm.LLMFactory import LLMProviderFactory
from src.stores.vectorstore.VectorStoreFactory import VectorStoreFactory


def initialize_rag_system():
    """Boots up core configuration and infrastructure primitives once."""
    logger.info("Initializing RAG Core Infrastructure...")
    config = Settings()

    # 1. Instantiate factories and models
    llm_factory = LLMProviderFactory(config)
    llm = llm_factory.create_llm()
    embedding_model = llm_factory.create_embedding_model()

    # 2. Instantiate VectorStore
    vectorstore_factory = VectorStoreFactory(config)
    vectorstore = vectorstore_factory.create_vectorstore(
        embedding_model=embedding_model
    )

    # 3. Instantiate Retriever & Chain
    retriever = build_retriever(vectorstore, config)
    rag_chain = build_rag_chain(llm, retriever)

    return config, vectorstore, rag_chain


# Global initializations
config, vectorstore, rag_chain = initialize_rag_system()


def handle_file_upload(file_obj: Any) -> str:
    """Handles document ingestion via the UI."""
    if file_obj is None:
        return "Please select a file to upload."

    try:
        num_chunks = run_ingestion(
            file_path=file_obj.name, vectorstore=vectorstore, config=config
        )
        return f"✅ Successfully ingested file into {num_chunks} chunks!"
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        return f"❌ Ingestion failed: {str(e)}"


def handle_user_query(message: str, history: Any) -> str:
    """Handles query submission through the RAG chain."""
    if not message.strip():
        return "Please enter a valid prompt."

    try:
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return f"❌ Error generating response: {str(e)}"


# Build Gradio UI
with gr.Blocks(title="RAG Knowledge Base") as demo:
    gr.Markdown("# 📚 Chat With Knowledge Base")
    gr.Markdown(
        "Upload your documents (PDF, Markdown, TXT) and ask questions based on your ingested knowledge base."
    )

    with gr.Row():
        # Left Panel: Ingestion
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ingest Documents")
            file_input = gr.File(
                label="Select Document",
                file_types=[".pdf", ".md", ".txt"],
            )
            upload_btn = gr.Button("Ingest Document", variant="primary")
            status_output = gr.Textbox(
                label="Ingestion Status", interactive=False
            )

            upload_btn.click(
                fn=handle_file_upload,
                inputs=[file_input],
                outputs=[status_output],
                api_name=False,
            )

        # Right Panel: Retrieval & Generation Chat
        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat with your Data")
            gr.ChatInterface(
                fn=handle_user_query,
                type="messages",
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)