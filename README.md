# RAG Document Assistant

A basic **Retrieval-Augmented Generation (RAG) system** for PDF documents using **vector databases** and **local LLMs**.  
This project demonstrates a complete pipeline: document ingestion, vector search, LLM integration, and an API for querying documents. Also includes a Streamlit chat interface that allows users to interact with the RAG system and ask questions about uploaded documents.  

Built with **Python**, **FastAPI**, **PGVector**, **Ollama LLMs** and **Streamlit**.

---

## Features

- Ingest PDF documents and split them into semantic chunks
- Generate embeddings using **nomic-embed-text** model
- Store and search embeddings in **PostgreSQL + PGVector**
- Retrieval-Augmented Generation (RAG) using **llama3.2**
- **REST API** for asking questions
- **Upload endpoint** for adding new documents dynamically
- Local, free LLM usage (no OpenAI API required)
- **Streamlit** chat interface

---

## Architecture

```markdown
User  
↓  
Streamlit Chat UI  
↓  
FastAPI /ask endpoint  
↓  
Retriever (PGVector)  
↓  
Relevant Document Chunks  
↓  
LLM (Ollama llama3.2)  
↓  
Answer
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/StijnVanRiet/rag-document-assistant.git
cd rag-document-assistant
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Start PostgreSQL with PGVector:

```bash
docker compose up -d
```

4. Install Ollama models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

5. Create documents folder with files for retrieval

6. Initial ingestion of documents:

```bash
python -m app.ingest
```

7. Start API:

```bash
uvicorn app.main:app --reload 
```

8. Open browser to test API:

<http://127.0.0.1:8000/docs>

9. Start UI:

```bash
streamlit run ui/chat.py
```

10. Open browser to test UI:

<http://localhost:8501>
