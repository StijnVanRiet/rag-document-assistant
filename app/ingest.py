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
    text = text.strip()
    return text

# TODO: duplicate code
def load_documents():
    docs = []
    for file in Path(COLLECTION_NAME).glob("*.pdf"):

        print(f"Loading {file}")

        loader = PyPDFLoader(str(file))
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.page_content = clean_text(doc.page_content)
            # set source metadata of each page to filename
            doc.metadata["source"] = file.name
            # page number
            doc.metadata["page"] = doc.metadata.get("page", 0)

        docs.extend(pdf_docs)

    return docs


def ingest():
    documents = load_documents()

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = splitter.split_documents(documents)

    print(f"Created {len(split_docs)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )

    print("Documents ingested successfully")


def ingest_file(path):
    loader = PyPDFLoader(path)
    pdf_docs = loader.load()
    filename = os.path.basename(path)
    for doc in pdf_docs:
        doc.page_content = clean_text(doc.page_content)
        # set source metadata of each page to filename
        doc.metadata["source"] = filename
        # page number
        doc.metadata["page"] = doc.metadata.get("page", 0)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = splitter.split_documents(pdf_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )

    print(f"{filename} ingested successfully")


if __name__ == "__main__":
    ingest()
