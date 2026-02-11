from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Example free Hugging Face LLM
MODEL_NAME = "bigscience/bloomz-560m"  # small-medium model for testing

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def summarize_chunks(chunks):
    """
    Generate structured medical note from retrieved chunks
    """
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
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


if __name__ == "__main__":
    from retriever import retrieve_chunks

    test_query = "Patient with fever, cough, and sore throat"
    chunks = retrieve_chunks(test_query)
    note = summarize_chunks(chunks)
    print("Generated Medical Note:\n")
    print(note)
