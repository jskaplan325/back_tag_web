"""
Feedback Router - Human-in-the-loop tag feedback for learning.
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database.db import get_db, Document, Result, Matter, TagFeedback

router = APIRouter()


class FeedbackSubmit(BaseModel):
    tag_name: str
    action: str  # 'confirmed', 'rejected'
    original_confidence: float
    reviewed_by: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    tag_name: str
    action: str
    original_confidence: float
    reviewed_at: datetime
    reviewed_by: Optional[str]


class AddTagRequest(BaseModel):
    tag_name: str
    reviewed_by: Optional[str] = None


@router.post("/documents/{document_id}/feedback", response_model=FeedbackResponse)
async def submit_tag_feedback(
    document_id: str,
    feedback: FeedbackSubmit,
    db: Session = Depends(get_db)
):
    """Submit feedback for a tag detection."""
    # Get document and latest result
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    result = db.query(Result).filter(Result.document_id == document_id).order_by(Result.processed_at.desc()).first()
    if not result:
        raise HTTPException(status_code=404, detail="No results found for document")

    # Get matter context
    matter = db.query(Matter).filter(Matter.id == doc.matter_id).first() if doc.matter_id else None

    # Check if feedback already exists for this tag
    existing = db.query(TagFeedback).filter(
        TagFeedback.document_id == document_id,
        TagFeedback.tag_name == feedback.tag_name
    ).first()

    if existing:
        # Update existing feedback
        existing.action = feedback.action
        existing.original_confidence = feedback.original_confidence
        existing.reviewed_by = feedback.reviewed_by
        existing.reviewed_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return FeedbackResponse(
            id=existing.id,
            tag_name=existing.tag_name,
            action=existing.action,
            original_confidence=existing.original_confidence,
            reviewed_at=existing.reviewed_at,
            reviewed_by=existing.reviewed_by
        )

    # Create new feedback
    fb = TagFeedback(
        id=str(uuid.uuid4()),
        document_id=document_id,
        result_id=result.id,
        tag_name=feedback.tag_name,
        original_confidence=feedback.original_confidence,
        original_detected=True,
        action=feedback.action,
        reviewed_by=feedback.reviewed_by,
        reviewed_at=datetime.utcnow(),
        area_of_law=None,  # TODO: Link when AoL taxonomy is connected
        matter_type=matter.matter_type if matter else None
    )

    db.add(fb)
    db.commit()
    db.refresh(fb)

    return FeedbackResponse(
        id=fb.id,
        tag_name=fb.tag_name,
        action=fb.action,
        original_confidence=fb.original_confidence,
        reviewed_at=fb.reviewed_at,
        reviewed_by=fb.reviewed_by
    )


@router.post("/documents/{document_id}/feedback/add-tag", response_model=FeedbackResponse)
async def add_missing_tag(
    document_id: str,
    request: AddTagRequest,
    db: Session = Depends(get_db)
):
    """Add a tag that was missed by the ML model."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    result = db.query(Result).filter(Result.document_id == document_id).order_by(Result.processed_at.desc()).first()
    if not result:
        raise HTTPException(status_code=404, detail="No results found for document")

    matter = db.query(Matter).filter(Matter.id == doc.matter_id).first() if doc.matter_id else None

    # Check if already added
    existing = db.query(TagFeedback).filter(
        TagFeedback.document_id == document_id,
        TagFeedback.tag_name == request.tag_name,
        TagFeedback.action == "added"
    ).first()

    if existing:
        return FeedbackResponse(
            id=existing.id,
            tag_name=existing.tag_name,
            action=existing.action,
            original_confidence=existing.original_confidence,
            reviewed_at=existing.reviewed_at,
            reviewed_by=existing.reviewed_by
        )

    fb = TagFeedback(
        id=str(uuid.uuid4()),
        document_id=document_id,
        result_id=result.id,
        tag_name=request.tag_name,
        original_confidence=0.0,  # ML didn't detect it
        original_detected=False,
        action="added",
        reviewed_by=request.reviewed_by,
        reviewed_at=datetime.utcnow(),
        area_of_law=None,
        matter_type=matter.matter_type if matter else None
    )

    db.add(fb)
    db.commit()
    db.refresh(fb)

    return FeedbackResponse(
        id=fb.id,
        tag_name=fb.tag_name,
        action=fb.action,
        original_confidence=fb.original_confidence,
        reviewed_at=fb.reviewed_at,
        reviewed_by=fb.reviewed_by
    )


@router.get("/documents/{document_id}/feedback", response_model=List[FeedbackResponse])
async def get_document_feedback(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get all feedback for a document."""
    feedbacks = db.query(TagFeedback).filter(TagFeedback.document_id == document_id).all()

    return [
        FeedbackResponse(
            id=fb.id,
            tag_name=fb.tag_name,
            action=fb.action,
            original_confidence=fb.original_confidence,
            reviewed_at=fb.reviewed_at,
            reviewed_by=fb.reviewed_by
        )
        for fb in feedbacks
    ]


@router.delete("/documents/{document_id}/feedback/{tag_name}")
async def undo_feedback(
    document_id: str,
    tag_name: str,
    db: Session = Depends(get_db)
):
    """Undo/remove feedback for a tag (revert to ML state)."""
    feedback = db.query(TagFeedback).filter(
        TagFeedback.document_id == document_id,
        TagFeedback.tag_name == tag_name
    ).first()

    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    db.delete(feedback)
    db.commit()

    return {"status": "ok", "message": f"Feedback for '{tag_name}' removed"}


class ReviewQueueItem(BaseModel):
    id: str
    type: str  # 'no_tags', 'low_confidence', 'failed'
    document_id: str
    document_name: str
    matter_id: Optional[str]
    matter_name: Optional[str]
    tag_name: Optional[str]  # For low_confidence items
    confidence: Optional[float]
    error_message: Optional[str]  # For failed items
    priority: int  # 1=high, 2=medium, 3=low


class ReviewQueueResponse(BaseModel):
    total_items: int
    items: List[ReviewQueueItem]
    by_type: dict


@router.get("/review-queue", response_model=ReviewQueueResponse)
async def get_review_queue(
    matter_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get items needing human review."""
    from ..database.db import Document, Result, Matter

    items = []

    # Build document query
    doc_query = db.query(Document)
    if matter_id:
        doc_query = doc_query.filter(Document.matter_id == matter_id)

    documents = doc_query.all()

    # Get all feedback to check what's already reviewed
    all_feedback = db.query(TagFeedback).all()
    reviewed_set = {(fb.document_id, fb.tag_name) for fb in all_feedback}

    for doc in documents:
        matter = db.query(Matter).filter(Matter.id == doc.matter_id).first() if doc.matter_id else None
        matter_name = matter.name if matter else None

        # 1. Failed documents not reviewed
        if doc.status == 'failed':
            if (doc.id, '__failed_reviewed__') not in reviewed_set:
                items.append(ReviewQueueItem(
                    id=f"failed_{doc.id}",
                    type='failed',
                    document_id=doc.id,
                    document_name=doc.filename,
                    matter_id=doc.matter_id,
                    matter_name=matter_name,
                    tag_name=None,
                    confidence=None,
                    error_message=doc.error_message,
                    priority=1
                ))
            continue

        # 2. Completed documents
        if doc.status == 'completed':
            result = db.query(Result).filter(Result.document_id == doc.id).order_by(Result.processed_at.desc()).first()

            if not result or not result.result_json:
                continue

            tags = result.result_json.get('tags', [])

            # 2a. Documents with 0 tags
            if len(tags) == 0:
                if (doc.id, '__no_tags__') not in reviewed_set:
                    items.append(ReviewQueueItem(
                        id=f"notags_{doc.id}",
                        type='no_tags',
                        document_id=doc.id,
                        document_name=doc.filename,
                        matter_id=doc.matter_id,
                        matter_name=matter_name,
                        tag_name=None,
                        confidence=0,
                        error_message=None,
                        priority=1
                    ))
            else:
                # 2b. Low confidence tags (<=70%) not reviewed
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', tag.get('tag_name', tag.get('tag', '')))
                        confidence = tag.get('confidence', tag.get('score', 0))
                    else:
                        continue

                    if confidence <= 0.70 and (doc.id, tag_name) not in reviewed_set:
                        items.append(ReviewQueueItem(
                            id=f"tag_{doc.id}_{tag_name}",
                            type='low_confidence',
                            document_id=doc.id,
                            document_name=doc.filename,
                            matter_id=doc.matter_id,
                            matter_name=matter_name,
                            tag_name=tag_name,
                            confidence=confidence,
                            error_message=None,
                            priority=2 if confidence >= 0.5 else 1
                        ))

    # Sort by priority, then by confidence (lowest first)
    items.sort(key=lambda x: (x.priority, x.confidence or 0))

    # Count by type
    by_type = {
        'failed': sum(1 for i in items if i.type == 'failed'),
        'no_tags': sum(1 for i in items if i.type == 'no_tags'),
        'low_confidence': sum(1 for i in items if i.type == 'low_confidence')
    }

    return ReviewQueueResponse(
        total_items=len(items),
        items=items[:limit],
        by_type=by_type
    )


@router.get("/feedback/stats")
async def get_feedback_stats(
    db: Session = Depends(get_db)
):
    """Get overall feedback statistics for learning insights."""
    all_feedback = db.query(TagFeedback).all()

    # Group by tag and matter type
    stats_by_tag = {}
    for fb in all_feedback:
        key = fb.tag_name
        if key not in stats_by_tag:
            stats_by_tag[key] = {
                "confirmed": 0,
                "rejected": 0,
                "added": 0,
                "by_matter_type": {}
            }

        stats_by_tag[key][fb.action] += 1

        # Track by matter type
        mt = fb.matter_type or "Unknown"
        if mt not in stats_by_tag[key]["by_matter_type"]:
            stats_by_tag[key]["by_matter_type"][mt] = {"confirmed": 0, "rejected": 0, "added": 0}
        stats_by_tag[key]["by_matter_type"][mt][fb.action] += 1

    # Calculate precision per tag
    output = []
    for tag_name, stats in stats_by_tag.items():
        total_reviewed = stats["confirmed"] + stats["rejected"]
        precision = stats["confirmed"] / total_reviewed if total_reviewed > 0 else None

        output.append({
            "tag_name": tag_name,
            "confirmed": stats["confirmed"],
            "rejected": stats["rejected"],
            "added": stats["added"],
            "precision": round(precision, 3) if precision else None,
            "by_matter_type": stats["by_matter_type"]
        })

    # Sort by total feedback count
    output.sort(key=lambda x: x["confirmed"] + x["rejected"] + x["added"], reverse=True)

    return {
        "total_feedback": len(all_feedback),
        "tags": output
    }
