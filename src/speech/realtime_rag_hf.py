from src.speech.stt_hf import transcribe_audio
from src.rag.retriever import retrieve_chunks
from src.rag.summarizer_hf import summarize_chunks

def process_audio_to_note(audio_file):
    transcription = transcribe_audio(audio_file)
    print("Transcribed Text:\n", transcription)
    chunks = retrieve_chunks(transcription)
    note = summarize_chunks(chunks)
    return note

if __name__ == "__main__":
    note = process_audio_to_note("data/raw/audio/test_patient.wav")
    print("\nGenerated Medical Note:\n")
    print(note)
