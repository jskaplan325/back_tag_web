"""
Document Intelligence Dashboard - FastAPI Backend

Main application entry point.
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database.db import init_db
from .routers import documents, results, models, metrics, taxonomy, matters

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    init_db()
    yield
    # Shutdown (cleanup if needed)


app = FastAPI(
    title="Document Intelligence Dashboard",
    description="API for document tagging with LegalBERT and Florence-2",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and CRA defaults
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving page images
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Include routers
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(taxonomy.router, prefix="/api/taxonomy", tags=["Taxonomy"])
app.include_router(matters.router, prefix="/api/matters", tags=["Matters"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Document Intelligence Dashboard",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "uploads_dir": str(UPLOADS_DIR),
        "results_dir": str(RESULTS_DIR),
    }


@app.get("/api/llm-status")
async def llm_status():
    """Check available LLM backends for Smart Mode."""
    from .services.llm_tagger import (
        check_ollama_available, OLLAMA_MODEL, OLLAMA_BASE_URL,
        GEMINI_AVAILABLE, get_available_backend
    )
    import requests

    result = {
        "smart_mode_available": False,
        "preferred_backend": None,
        "ollama": {
            "available": False,
            "url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "models_installed": []
        },
        "gemini": {
            "available": False,
            "api_key_set": bool(os.getenv('GOOGLE_API_KEY'))
        }
    }

    # Check Ollama
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            result["ollama"]["available"] = True
            result["ollama"]["models_installed"] = [m.get('name', '') for m in models]
    except Exception:
        pass

    # Check Gemini
    result["gemini"]["available"] = GEMINI_AVAILABLE and result["gemini"]["api_key_set"]

    # Set overall status
    result["smart_mode_available"] = result["ollama"]["available"] or result["gemini"]["available"]
    result["preferred_backend"] = get_available_backend()

    return result
