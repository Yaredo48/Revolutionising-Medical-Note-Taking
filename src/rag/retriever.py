import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "medical-rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # retrieve top 5 chunks

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve_chunks(query, top_k=TOP_K):
    """
    Retrieve top-k chunks for a given query.
    """
    query_vec = embed_model.encode([query]).astype(np.float32)[0]
    result = index.query(vector=query_vec.tolist(), top_k=top_k, include_metadata=True)

    # Extract text from metadata
    retrieved_chunks = [match["metadata"]["text"] for match in result["matches"]]
    return retrieved_chunks


if __name__ == "__main__":
    test_query = "Patient with fever, cough, and sore throat"
    chunks = retrieve_chunks(test_query)
    print("Retrieved Chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk[:200]}...")  # print first 200 chars
