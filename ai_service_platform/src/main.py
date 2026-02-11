"""
Main FastAPI application for AI Service Platform.
"""

import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.core.config import settings
from src.api.routes import (
    auth_router, tenant_router, user_router, apikey_router,
    service_router, usage_router
)
from src.api.middleware import RateLimitMiddleware
from src.db.database import Base, engine

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.observability.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting AI Service Platform...")
    
    # Initialize database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Service Platform...")


# Create FastAPI application
app = FastAPI(
    title="AI Service Platform",
    description="""
    Multi-tenant AI Service Provider Platform
    
    Features:
    - Multi-tenant architecture with tenant isolation
    - Multiple AI provider support (OpenAI, HuggingFace, Anthropic)
    - Vector database abstraction (Pinecone, Qdrant, Weaviate)
    - RAG (Retrieval-Augmented Generation) pipeline
    - Usage tracking and rate limiting
    - API key authentication
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(tenant_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")
app.include_router(apikey_router, prefix="/api/v1")
app.include_router(service_router, prefix="/api/v1")
app.include_router(usage_router, prefix="/api/v1")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.env
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI Service Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.env == "development" else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.env == "development"
    )
