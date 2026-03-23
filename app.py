from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data (you can change this)
documents = [
    "Artificial Intelligence is the future of technology",
    "Machine Learning is a subset of AI",
    "Python is widely used for AI and data science",
    "Transformers are powerful models for NLP",
    "FAISS is used for similarity search"
]

# Convert documents into embeddings
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Search function
def search(query):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k=1)
    return documents[indices[0][0]]

# Run loop
while True:
    query = input("\nEnter your question (or 'exit'): ")
    if query.lower() == "exit":
        break
    
    result = search(query)
    print("Best Match:", result)