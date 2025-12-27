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
class Matter(Base):
    """Matter/project grouping for documents."""
    __tablename__ = "matters"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    matter_type = Column(String(100), nullable=True)  # Funds, M&A, LevFin, etc.
    source_path = Column(String(512), nullable=True)  # Original folder path if bulk imported
    created_at = Column(DateTime, default=datetime.utcnow)
    document_count = Column(Integer, default=0)

    # Relationships
    documents = relationship("Document", back_populates="matter", cascade="all, delete-orphan")


class Document(Base):
    """Uploaded document record."""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    matter_id = Column(String(36), ForeignKey("matters.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_size_bytes = Column(Integer)
    page_count = Column(Integer)
    word_count = Column(Integer)
    status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    error_message = Column(Text, nullable=True)

    # Pipeline recommendation (from document analyzer)
    recommended_pipeline = Column(String(50), nullable=True)  # fast, zone, vision, ocr
    analysis_metadata = Column(JSON, nullable=True)  # Full analysis results

    # Relationships
    matter = relationship("Matter", back_populates="documents")
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

    # Extended model card info
    architecture = Column(Text)  # Model architecture description
    intended_use = Column(Text)  # Intended use cases
    limitations = Column(Text)  # Known limitations
    bias_risks = Column(Text)  # Bias and fairness concerns
    training_data = Column(Text)  # Training data description
    evaluation = Column(Text)  # Evaluation results/metrics
    model_card_json = Column(Text)  # Full model card as JSON

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


class AreaOfLaw(Base):
    """Area of Law for taxonomy organization."""
    __tablename__ = "areas_of_law"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    color = Column(String(20), default="#3b82f6")  # Tailwind blue-500
    icon = Column(String(50), default="Scale")  # Lucide icon name
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tags = relationship("Tag", back_populates="area_of_law", cascade="all, delete-orphan")


class Tag(Base):
    """Tag definition within an Area of Law."""
    __tablename__ = "tags"

    id = Column(String(36), primary_key=True)
    area_of_law_id = Column(String(36), ForeignKey("areas_of_law.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    patterns = Column(JSON, default=list)  # List of regex patterns
    semantic_examples = Column(JSON, default=list)  # List of example phrases
    threshold = Column(Float, default=0.45)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    area_of_law = relationship("AreaOfLaw", back_populates="tags")


class TagUsage(Base):
    """Track tag detection in documents."""
    __tablename__ = "tag_usages"

    id = Column(String(36), primary_key=True)
    tag_id = Column(String(36), ForeignKey("tags.id"), nullable=False)
    result_id = Column(String(36), ForeignKey("results.id"), nullable=False)
    confidence = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)


class TagFeedback(Base):
    """Human feedback on tag detections for learning loop."""
    __tablename__ = "tag_feedback"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    result_id = Column(String(36), ForeignKey("results.id"), nullable=False)
    tag_name = Column(String(100), nullable=False)

    # Original ML prediction
    original_confidence = Column(Float)
    original_detected = Column(Boolean, default=True)

    # Human feedback
    action = Column(String(20), nullable=False)  # 'confirmed', 'rejected', 'added'
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, default=datetime.utcnow)

    # Context for learning
    area_of_law = Column(String(100), nullable=True)
    matter_type = Column(String(100), nullable=True)

    # Relationships
    document = relationship("Document")
    result = relationship("Result")


class Annotation(Base):
    """User-created annotations for ML training data."""
    __tablename__ = "annotations"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)  # 1-indexed

    # Normalized bounding box (0.0-1.0 relative to page dimensions)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    # Tag association
    tag_name = Column(String(100), nullable=False)
    tag_id = Column(String(36), ForeignKey("tags.id"), nullable=True)
    area_of_law = Column(String(100), nullable=True)

    # Training metadata
    annotation_type = Column(String(20), default="positive")  # positive, negative, uncertain
    color = Column(String(20), default="green")  # green, yellow, red
    source = Column(String(20), default="human")  # human, ocr, ml

    # Audit
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relationships
    document = relationship("Document")
    tag = relationship("Tag")


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
