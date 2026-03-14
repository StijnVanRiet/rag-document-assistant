import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Document Assistant")

st.title("RAG Document Assistant")
st.write("Ask questions about your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
prompt = st.chat_input("Ask a question about the documents...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # call API
    response = requests.post(API_URL, json={"question": prompt})

    answer = response.json()["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

uploaded_file = st.file_uploader("Upload a PDF")

if uploaded_file:

    files = {"file": uploaded_file.getvalue()}

    r = requests.post("http://localhost:8000/upload", files={"file": uploaded_file})

    st.success("Document uploaded and indexed.")
