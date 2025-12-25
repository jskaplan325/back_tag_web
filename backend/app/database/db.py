"""
Database setup and connection management.

Uses SQLite for local development, easily switchable to PostgreSQL for production.
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Generator

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.sqlite import JSON

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/dashboard.db")

# Ensure data directory exists
Path("./data").mkdir(exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Models
class Document(Base):
    """Uploaded document record."""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_size_bytes = Column(Integer)
    page_count = Column(Integer)
    word_count = Column(Integer)
    status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    error_message = Column(Text, nullable=True)

    # Relationships
    results = relationship("Result", back_populates="document", cascade="all, delete-orphan")


class Result(Base):
    """Processing result record."""
    __tablename__ = "results"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    processing_time_seconds = Column(Float)

    # Model info
    semantic_model = Column(String(255))
    vision_model = Column(String(255), nullable=True)
    vision_enabled = Column(Boolean, default=False)

    # Results summary
    tag_count = Column(Integer)
    average_confidence = Column(Float)

    # Full result JSON
    result_json = Column(JSON)

    # Visual pages detected
    visual_pages = Column(JSON, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="results")
    model_usages = relationship("ModelUsage", back_populates="result", cascade="all, delete-orphan")


class Model(Base):
    """Model registry entry."""
    __tablename__ = "models"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)  # e.g., "pile-of-law/legalbert-large-1.7M-2"
    type = Column(String(50), nullable=False)  # semantic, vision, ocr

    # HuggingFace metadata
    huggingface_url = Column(String(512))
    size_gb = Column(Float)
    description = Column(Text)
    downloads = Column(Integer, default=0)
    license = Column(String(100))
    last_updated = Column(DateTime)

    # Approval workflow
    approved = Column(Boolean, default=False)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    usages = relationship("ModelUsage", back_populates="model")


class ModelUsage(Base):
    """Track model usage per processing run."""
    __tablename__ = "model_usages"

    id = Column(String(36), primary_key=True)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    result_id = Column(String(36), ForeignKey("results.id"), nullable=False)
    used_at = Column(DateTime, default=datetime.utcnow)
    processing_time_seconds = Column(Float)

    # Relationships
    model = relationship("Model", back_populates="usages")
    result = relationship("Result", back_populates="model_usages")


class APICost(Base):
    """Track API costs (for LLM zone detection, etc.)."""
    __tablename__ = "api_costs"

    id = Column(String(36), primary_key=True)
    provider = Column(String(50), nullable=False)  # openai, gemini, anthropic
    model = Column(String(100), nullable=False)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    result_id = Column(String(36), ForeignKey("results.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
