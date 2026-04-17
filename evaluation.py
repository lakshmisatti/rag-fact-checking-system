from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Sample dataset
queries = [
    "What is the capital of France?",
    "What does Earth revolve around?"
]

ground_truth = [
    "Paris",
    "Sun"
]

docs = [
    "Paris is the capital of France",
    "The Earth revolves around the Sun"
]

embedding = HuggingFaceEmbeddings()
db = FAISS.from_texts(docs, embedding)

correct = 0

for i, query in enumerate(queries):
    retrieved_docs = db.similarity_search(query)
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

    if ground_truth[i].lower() in retrieved_text.lower():
        correct += 1

accuracy = correct / len(queries)

print(f"Retrieval Accuracy: {accuracy}")
