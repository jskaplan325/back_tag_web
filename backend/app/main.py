"""
Document Intelligence Dashboard - FastAPI Backend

Main application entry point.
"""
# Python 3.9 compatibility shim for huggingface_hub
# packages_distributions() was added in Python 3.10
import sys
if sys.version_info < (3, 10):
    import importlib.metadata
    if not hasattr(importlib.metadata, 'packages_distributions'):
        def _packages_distributions():
            """Fallback implementation of packages_distributions for Python 3.9."""
            pkg_to_dist = {}
            for dist in importlib.metadata.distributions():
                if dist.files:
                    for file in dist.files:
                        name = file.parts[0] if file.parts else ''
                        if name:
                            pkg_to_dist.setdefault(name, []).append(dist.metadata['Name'])
            return pkg_to_dist
        importlib.metadata.packages_distributions = _packages_distributions

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database.db import init_db
from .routers import documents, results, models, metrics, taxonomy, matters, feedback, annotations

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
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(annotations.router, prefix="/api", tags=["Annotations"])


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
