import os
from src.helpers.configs import Settings
from src.stores.llm.LLMFactory import LLMProviderFactory
from src.stores.vectorstore.VectorStoreFactory import VectorStoreFactory
from src.features.ingestion.pipeline import run_ingestion
from src.features.retrieval.retriever import build_retriever
from src.features.generation.chain import build_rag_chain

# 1. Boot up configurations and core instances once
config = Settings()

llm_factory = LLMProviderFactory(config)
llm = llm_factory.create_llm()
embedding_model = llm_factory.create_embedding_model()

vectorstore_factory = VectorStoreFactory(config)
vectorstore = vectorstore_factory.create_vectorstore(embedding_model=embedding_model)

# 2. Phase 1: Ingestion (Data Phase)
# Example: run_ingestion("data/sample.pdf", vectorstore, config)

# 3. Phase 2: Retrieval (Openbook Phase)
retriever = build_retriever(vectorstore, config)

# 4. Phase 3: Generation (Augmentation & Inference Phase)
rag_chain = build_rag_chain(llm, retriever)

# 5. Execute
# response = rag_chain.invoke("What is the core topic of the ingested document?")
# print(response)
