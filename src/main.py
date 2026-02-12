"""
Medical RAG (Retrieval-Augmented Generation) Application
Real-time speech-to-text and medical note generation.
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from speech.stt_hf import transcribe_audio
from rag.retriever import retrieve_chunks
from rag.summarizer_hf import summarize_chunks
from ingestion.pdf_parser import extract_text_from_pdf, ingest_pdfs

app = FastAPI(
    title="Medical RAG API",
    description="Real-time speech-to-text and medical note generation using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TranscriptionRequest(BaseModel):
    audio_path: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class NoteResponse(BaseModel):
    transcription: str
    retrieved_chunks: list
    summary: str

class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: list

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Medical RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/v1/transcribe", response_model=NoteResponse)
async def transcribe(request: TranscriptionRequest):
    """
    Transcribe audio file and generate medical note.
    """
    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Transcribe audio
    transcription = transcribe_audio(request.audio_path)
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(transcription)
    
    # Generate summary
    summary = summarize_chunks(retrieved_chunks)
    
    return NoteResponse(
        transcription=transcription,
        retrieved_chunks=retrieved_chunks,
        summary=summary
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base for relevant medical information.
    """
    retrieved_chunks = retrieve_chunks(request.query, top_k=request.top_k)
    return QueryResponse(
        query=request.query,
        retrieved_chunks=retrieved_chunks
    )

@app.get("/api/v1/ingest-pdfs")
async def trigger_pdf_ingestion():
    """
    Trigger PDF ingestion process.
    """
    try:
        ingest_pdfs()
        return {"status": "PDF ingestion completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/extract-pdf")
async def extract_pdf_text(pdf_path: str):
    """
    Extract text from a specific PDF file.
    """
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    try:
        text = extract_text_from_pdf(pdf_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
