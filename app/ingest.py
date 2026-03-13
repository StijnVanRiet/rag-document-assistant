from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from pathlib import Path
import re
import os

from app.config import DB_CONNECTION, COLLECTION_NAME


def clean_text(text: str) -> str:
    """
    Clean problematic characters from PDF text.
    """
    text = text.replace("\x00", "")  # remove NUL characters
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


def load_pdf(path: str):
    """
    Load a PDF, clean text, and set metadata.
    """
    loader = PyPDFLoader(path)
    pdf_docs = loader.load()
    filename = os.path.basename(path)

    for doc in pdf_docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = filename
        doc.metadata["page"] = doc.metadata.get("page", 0)

    return pdf_docs


def split_and_ingest(documents, collection_name=COLLECTION_NAME):
    """
    Split documents into chunks, create embeddings, and store in PGVector.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    print(f"Created {len(split_docs)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=DB_CONNECTION,
    )


def ingest_folder(folder_path=COLLECTION_NAME):
    """
    Ingest all PDF files in a folder.
    """
    all_docs = []
    for file in Path(folder_path).glob("*.pdf"):
        print(f"Loading {file}")
        docs = load_pdf(str(file))
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages")
    split_and_ingest(all_docs)

    print("Documents ingested successfully")


def ingest_file(path):
    """
    Ingest a single PDF file.
    """
    docs = load_pdf(path)
    split_and_ingest(docs)
    print(f"{os.path.basename(path)} ingested successfully")


if __name__ == "__main__":
    ingest_folder()
