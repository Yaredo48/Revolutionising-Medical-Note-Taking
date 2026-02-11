import os
import uuid
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pinecone import Pinecone, ServerlessSpec
from src.embeddings.chunker import load_and_chunk

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "medical-rag"

client = InferenceClient(token=HF_API_KEY)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # embedding dimension of all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(INDEX_NAME)

def get_embeddings(texts):
    """
    Generate embeddings via Hugging Face Inference API
    """
    embeddings = []
    for t in texts:
        emb = client.text_embedding(model="sentence-transformers/all-MiniLM-L6-v2", 
                                    inputs=t)
        embeddings.append(emb)
    return np.array(embeddings).astype("float32")

def create_embeddings_and_upload():
    chunks = load_and_chunk()
    print(f"Generating embeddings for {len(chunks)} chunks via HF API...")
    embeddings = get_embeddings(chunks)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} vectors to Pinecone.")

if __name__ == "__main__":
    create_embeddings_and_upload()
