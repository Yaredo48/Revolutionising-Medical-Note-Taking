import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(token=HF_API_KEY)

def transcribe_audio(audio_file):
    """
    Transcribe audio via Hugging Face API
    """
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    result = client.audio_to_text(model="openai/whisper-small", 
                                  audio=audio_bytes)
    return result["text"]

if __name__ == "__main__":
    transcription = transcribe_audio("data/raw/audio/test_patient.wav")
    print(transcription)
