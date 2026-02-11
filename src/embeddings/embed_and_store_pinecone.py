import os
import uuid
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from src.embeddings.chunker import load_and_chunk

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

INDEX_NAME = "medical-rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings_and_upload():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading and chunking documents...")
    chunks = load_and_chunk()

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        print("Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embeddings.shape[1],
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

    index = pc.Index(INDEX_NAME)

    print("Uploading vectors to Pinecone...")

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)

    print("Upload complete!")
    print(f"Uploaded {len(vectors)} vectors.")


if __name__ == "__main__":
    create_embeddings_and_upload()
