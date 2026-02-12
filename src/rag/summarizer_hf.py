import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from src.rag.retriever import retrieve_chunks  # keep Pinecone retriever

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(token=HF_API_KEY)

def summarize_chunks(chunks):
    context = "\n\n".join(chunks)
    prompt = f"""
You are a medical assistant. Generate a structured medical note from the following information:

{context}

Output format:
Patient Summary:
Symptoms:
Diagnosis:
Plan:
Sources:
"""
    response = client.text_generation(
        model="bigscience/bloomz-560m", 
        inputs=prompt, 
        parameters={"max_new_tokens": 200}
    )
    return response  # text_generation returns a string directly

if __name__ == "__main__":
    test_query = "Patient with fever, cough, and sore throat"
    chunks = retrieve_chunks(test_query)
    note = summarize_chunks(chunks)
    print(note)
