from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.helpers.configs import Settings

def get_text_splitter(config: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
