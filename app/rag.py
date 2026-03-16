from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings, ChatOllama

from app.config import DB_CONNECTION, COLLECTION_NAME


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_CONNECTION,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # top k chunks

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
