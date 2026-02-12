import os
import asyncio
from fastapi import FastAPI, WebSocket
from speech.stt_hf import transcribe_audio
from rag.retriever import retrieve_chunks
from rag.summarizer_hf import summarize_chunks

app = FastAPI(title="Medical RAG Streaming API")

# Dictionary to store session memory
session_memory = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in session_memory:
        session_memory[session_id] = []  # Initialize memory

    try:
        while True:
            data = await websocket.receive_bytes()  # receive audio chunk
            # Save chunk temporarily
            temp_file = f"temp_{session_id}.wav"
            with open(temp_file, "wb") as f:
                f.write(data)

            # Transcribe audio chunk
            transcription = transcribe_audio(temp_file)
            session_memory[session_id].append(transcription)

            # Retrieve relevant chunks from Pinecone
            combined_context = " ".join(session_memory[session_id])
            retrieved_chunks = retrieve_chunks(combined_context)

            # Summarize with context + retrieved chunks
            note = summarize_chunks(retrieved_chunks)

            # Send back updated note
            await websocket.send_json({
                "transcription": transcription,
                "note": note
            })

            # Delete temp file
            os.remove(temp_file)

    except Exception as e:
        await websocket.close()
        print(f"Session {session_id} closed: {e}")
