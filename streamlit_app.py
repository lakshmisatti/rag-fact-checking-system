import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

# Title
st.title("🔍 RAG Fact Checking System")

# Input
query = st.text_input("Enter your question:")

# Sample documents
docs = [
    "Paris is the capital of France",
    "The Earth revolves around the Sun",
    "Water boils at 100 degrees Celsius"
]

# Build vector DB
embedding = HuggingFaceEmbeddings()
db = FAISS.from_texts(docs, embedding)

if query:
    retrieved_docs = db.similarity_search(query)

    # LLM response
    llm = OpenAI()
    response = llm.predict(f"Answer based on: {retrieved_docs} Question: {query}")

    st.subheader("Answer:")
    st.write(response)

    st.subheader("Retrieved Documents:")
    for doc in retrieved_docs:
        st.write("-", doc.page_content)
