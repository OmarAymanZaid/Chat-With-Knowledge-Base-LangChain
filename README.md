# Chat-With-Knowledge-Base-LangChain

A lightweight **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **Google Gemini**, and **ChromaDB**. The application allows you to ask questions about your own documents by retrieving relevant information before generating an answer.

This project is designed as a learning resource for understanding the core components of a RAG pipeline using LangChain.

---

## Features

* Load text documents from a local directory
* Split documents into manageable chunks
* Generate embeddings for semantic search
* Store embeddings in a Chroma vector database
* Retrieve relevant context for user queries
* Generate grounded responses using Google Gemini
* Easily replace the knowledge base with your own documents

---

## Tech Stack

* Python
* LangChain
* Google Gemini
* ChromaDB
* Python Dotenv

---

## Project Structure

```text
simple-rag/
│
├── app.py                 # Chat interface
├── ingest.py              # Creates the vector database
├── requirements.txt
├── .env
│
├── data/                  # Knowledge base
│   ├── ai.txt
│   ├── python.txt
│   └── ...
│
└── chroma_db/             # Generated vector database
```

---

## How It Works

```text
Documents
    │
    ▼
Document Loader
    │
    ▼
Text Splitter
    │
    ▼
Embeddings
    │
    ▼
Chroma Vector Database
    │
    ▼
Retriever
    │
    ▼
Google Gemini
    │
    ▼
Answer
```

When a question is asked, the chatbot retrieves the most relevant document chunks from the vector database and provides them to the language model as context. This helps produce responses based on your documents instead of relying solely on the model's general knowledge.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/simple-rag.git
cd simple-rag
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5. Add your documents

Place your text files inside the `data/` directory.

Example:

```text
data/
├── ai.txt
├── python.txt
└── machine_learning.txt
```

### 6. Build the vector database

```bash
python ingest.py
```

### 7. Start the chatbot

```bash
python app.py
```

---

## Example

```text
You: What is Retrieval-Augmented Generation?

AI:
Retrieval-Augmented Generation (RAG) is a technique that retrieves relevant information from external documents before generating a response, allowing the language model to answer using your own knowledge base.
```

---

## Learning Objectives

This project demonstrates the fundamentals of building a RAG application with LangChain, including:

* Document loading
* Text chunking
* Embedding generation
* Vector databases
* Semantic search
* Retrievers
* Prompting
* Retrieval-Augmented Generation (RAG)

---

## Future Improvements

* Support PDF documents
* Support Word documents
* Display source citations
* Conversation memory
* Streaming responses
* Web interface with Streamlit or React
* Hybrid search
* Multiple knowledge bases
* Agent-based retrieval
* Chat history persistence

---

## License

This project is licensed under the MIT License.

You can easily evolve this README as the project grows by updating the **Features**, **Project Structure**, and **Future Improvements** sections without needing to rewrite the rest.

