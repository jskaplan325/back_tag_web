"""
Results Router - View processing results.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database.db import get_db, Result, Document

router = APIRouter()


class ResultSummary(BaseModel):
    id: str
    document_id: str
    document_name: str
    processed_at: datetime
    processing_time_seconds: Optional[float]
    semantic_model: Optional[str]
    vision_enabled: bool
    tag_count: int
    average_confidence: Optional[float]

    class Config:
        from_attributes = True


class ResultDetail(BaseModel):
    id: str
    document_id: str
    document_name: str
    processed_at: datetime
    processing_time_seconds: Optional[float]
    semantic_model: Optional[str]
    vision_model: Optional[str]
    vision_enabled: bool
    tag_count: int
    average_confidence: Optional[float]
    result_json: Optional[Dict[str, Any]]
    visual_pages: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class TagDetail(BaseModel):
    tag_name: str
    confidence: float
    method: str
    pattern_match: bool
    semantic_score: float
    matched_patterns: List[str]


@router.get("", response_model=List[ResultSummary])
async def list_results(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all processing results."""
    results = db.query(Result).order_by(Result.processed_at.desc()).offset(skip).limit(limit).all()

    summaries = []
    for result in results:
        document = db.query(Document).filter(Document.id == result.document_id).first()
        summaries.append(ResultSummary(
            id=result.id,
            document_id=result.document_id,
            document_name=document.filename if document else "Unknown",
            processed_at=result.processed_at,
            processing_time_seconds=result.processing_time_seconds,
            semantic_model=result.semantic_model,
            vision_enabled=result.vision_enabled,
            tag_count=result.tag_count,
            average_confidence=result.average_confidence
        ))

    return summaries


@router.get("/{result_id}", response_model=ResultDetail)
async def get_result(result_id: str, db: Session = Depends(get_db)):
    """Get detailed result."""
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    document = db.query(Document).filter(Document.id == result.document_id).first()

    return ResultDetail(
        id=result.id,
        document_id=result.document_id,
        document_name=document.filename if document else "Unknown",
        processed_at=result.processed_at,
        processing_time_seconds=result.processing_time_seconds,
        semantic_model=result.semantic_model,
        vision_model=result.vision_model,
        vision_enabled=result.vision_enabled,
        tag_count=result.tag_count,
        average_confidence=result.average_confidence,
        result_json=result.result_json,
        visual_pages=result.visual_pages
    )


@router.get("/{result_id}/tags", response_model=List[TagDetail])
async def get_result_tags(result_id: str, db: Session = Depends(get_db)):
    """Get tag breakdown for a result."""
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    if not result.result_json or 'details' not in result.result_json:
        return []

    tags = []
    for tag_name, details in result.result_json['details'].items():
        tags.append(TagDetail(
            tag_name=tag_name,
            confidence=details.get('confidence', 0),
            method=details.get('method', 'unknown'),
            pattern_match=details.get('pattern_match', False),
            semantic_score=details.get('semantic_score', 0),
            matched_patterns=details.get('matched_patterns', [])
        ))

    # Sort by confidence descending
    tags.sort(key=lambda x: x.confidence, reverse=True)
    return tags


@router.get("/document/{document_id}", response_model=List[ResultSummary])
async def get_document_results(document_id: str, db: Session = Depends(get_db)):
    """Get all results for a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    results = db.query(Result).filter(Result.document_id == document_id).order_by(Result.processed_at.desc()).all()

    return [
        ResultSummary(
            id=r.id,
            document_id=r.document_id,
            document_name=document.filename,
            processed_at=r.processed_at,
            processing_time_seconds=r.processing_time_seconds,
            semantic_model=r.semantic_model,
            vision_enabled=r.vision_enabled,
            tag_count=r.tag_count,
            average_confidence=r.average_confidence
        )
        for r in results
    ]
