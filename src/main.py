import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.speech.realtime_rag_hf import process_audio_to_note

app = FastAPI(title="Medical RAG API")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Upload audio file, return structured medical note
    """
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        # Process audio → RAG → structured note
        note = process_audio_to_note(temp_file)

        # Delete temp file
        os.remove(temp_file)

        return JSONResponse(content={"note": note})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Medical RAG API is running"}
