from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ✅ Create FastAPI app (IMPORTANT)
app = FastAPI()

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dataset
documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence by machines",
    "Machine Learning is a subset of Artificial Intelligence that allows systems to learn from data",
    "Python is widely used for Artificial Intelligence and data science",
    "Transformers are deep learning models used in natural language processing",
    "FAISS is a library used for fast similarity search in vector databases"
]

# Create embeddings
doc_embeddings = model.encode(documents)

# Normalize embeddings
faiss.normalize_L2(doc_embeddings)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

# Request model
class QueryRequest(BaseModel):
    query: str

# Search function
def search(query, k=2):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(documents[i])

    return results

# API route
@app.post("/search")
def search_api(request: QueryRequest):
    return {
        "query": request.query,
        "results": search(request.query)
    }

# Home route
@app.get("/")
def home():
    return {"message": "API is working 🚀"}