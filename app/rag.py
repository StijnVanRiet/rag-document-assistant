from langchain_community.vectorstores.pgvector import PGVector
from langchain_ollama import OllamaEmbeddings, ChatOllama

from app.config import DB_CONNECTION, COLLECTION_NAME


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    connection_string=DB_CONNECTION,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

retriever = vector_store.as_retriever()

llm = ChatOllama(model="llama3.2", temperature=0)


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
            You are an AI assistant answering questions using the provided context.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

    response = llm.invoke(prompt)

    return response.content
