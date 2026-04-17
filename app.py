from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

# Sample documents
docs = ["Paris is the capital of France", "The Earth revolves around the Sun"]

# Create embeddings
embedding = HuggingFaceEmbeddings()
db = FAISS.from_texts(docs, embedding)

# Query
query = "What is the capital of France?"
retrieved_docs = db.similarity_search(query)

# Generate answer
llm = OpenAI()
response = llm.predict(f"Answer based on: {retrieved_docs} Question: {query}")

print(response)
