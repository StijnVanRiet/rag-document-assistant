from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from pathlib import Path
import re

from app.config import DB_CONNECTION, COLLECTION_NAME


def clean_text(text: str) -> str:
    if text:
        text = text.replace("\x00", "")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
    return text


def load_documents():
    docs = []
    for file in Path(COLLECTION_NAME).glob("*.pdf"):

        print(f"Loading {file}")

        loader = PyPDFLoader(str(file))
        pdf_docs = loader.load()

        for doc in pdf_docs:
            # remove NUL characters
            doc.page_content = clean_text(doc.page_content)
            # set source metadata of each page to filename
            doc.metadata["source"] = file.name

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
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PGVector.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DB_CONNECTION,
    )


if __name__ == "__main__":
    ingest()
