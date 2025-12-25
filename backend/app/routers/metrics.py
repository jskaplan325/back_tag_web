"""
Metrics Router - Dashboard statistics and cost tracking.
"""
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from ..database.db import get_db, Document, Result, Model, ModelUsage, APICost

router = APIRouter()


class DashboardSummary(BaseModel):
    total_documents: int
    total_processed: int
    total_failed: int
    avg_confidence: float
    avg_processing_time: float
    total_tags_detected: int
    models_registered: int
    models_approved: int


class ProcessingTrend(BaseModel):
    date: str
    documents_processed: int
    avg_processing_time: float
    avg_confidence: float


class ModelUsageSummary(BaseModel):
    model_name: str
    model_type: str
    usage_count: int
    total_processing_time: float
    approved: bool


class CostSummary(BaseModel):
    provider: str
    total_cost: float
    total_tokens_in: int
    total_tokens_out: int
    request_count: int


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get overall dashboard statistics."""
    total_docs = db.query(Document).count()
    completed = db.query(Document).filter(Document.status == "completed").count()
    failed = db.query(Document).filter(Document.status == "failed").count()

    # Get average confidence and processing time from results
    results = db.query(Result).all()
    if results:
        avg_confidence = sum(r.average_confidence or 0 for r in results) / len(results)
        avg_time = sum(r.processing_time_seconds or 0 for r in results) / len(results)
        total_tags = sum(r.tag_count or 0 for r in results)
    else:
        avg_confidence = 0
        avg_time = 0
        total_tags = 0

    models_total = db.query(Model).count()
    models_approved = db.query(Model).filter(Model.approved == True).count()

    return DashboardSummary(
        total_documents=total_docs,
        total_processed=completed,
        total_failed=failed,
        avg_confidence=round(avg_confidence, 3),
        avg_processing_time=round(avg_time, 2),
        total_tags_detected=total_tags,
        models_registered=models_total,
        models_approved=models_approved
    )


@router.get("/processing", response_model=List[ProcessingTrend])
async def get_processing_trends(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get processing trends over time."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    results = db.query(Result).filter(Result.processed_at >= cutoff).all()

    # Group by date
    trends_by_date = {}
    for result in results:
        date_str = result.processed_at.strftime("%Y-%m-%d")
        if date_str not in trends_by_date:
            trends_by_date[date_str] = {
                "count": 0,
                "total_time": 0,
                "total_confidence": 0
            }
        trends_by_date[date_str]["count"] += 1
        trends_by_date[date_str]["total_time"] += result.processing_time_seconds or 0
        trends_by_date[date_str]["total_confidence"] += result.average_confidence or 0

    # Convert to list
    trends = []
    for date_str in sorted(trends_by_date.keys()):
        data = trends_by_date[date_str]
        trends.append(ProcessingTrend(
            date=date_str,
            documents_processed=data["count"],
            avg_processing_time=round(data["total_time"] / data["count"], 2) if data["count"] > 0 else 0,
            avg_confidence=round(data["total_confidence"] / data["count"], 3) if data["count"] > 0 else 0
        ))

    return trends


@router.get("/models", response_model=List[ModelUsageSummary])
async def get_model_usage_summary(db: Session = Depends(get_db)):
    """Get model usage breakdown."""
    models = db.query(Model).all()

    summaries = []
    for model in models:
        usages = db.query(ModelUsage).filter(ModelUsage.model_id == model.id).all()
        total_time = sum(u.processing_time_seconds or 0 for u in usages)

        summaries.append(ModelUsageSummary(
            model_name=model.name,
            model_type=model.type,
            usage_count=len(usages),
            total_processing_time=round(total_time, 2),
            approved=model.approved
        ))

    # Sort by usage count descending
    summaries.sort(key=lambda x: x.usage_count, reverse=True)
    return summaries


@router.get("/costs", response_model=List[CostSummary])
async def get_cost_summary(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get API cost breakdown by provider."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    costs = db.query(APICost).filter(APICost.created_at >= cutoff).all()

    # Group by provider
    by_provider = {}
    for cost in costs:
        if cost.provider not in by_provider:
            by_provider[cost.provider] = {
                "total_cost": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "count": 0
            }
        by_provider[cost.provider]["total_cost"] += cost.cost_usd or 0
        by_provider[cost.provider]["tokens_in"] += cost.tokens_in or 0
        by_provider[cost.provider]["tokens_out"] += cost.tokens_out or 0
        by_provider[cost.provider]["count"] += 1

    return [
        CostSummary(
            provider=provider,
            total_cost=round(data["total_cost"], 4),
            total_tokens_in=data["tokens_in"],
            total_tokens_out=data["tokens_out"],
            request_count=data["count"]
        )
        for provider, data in by_provider.items()
    ]


@router.get("/tags")
async def get_tag_statistics(db: Session = Depends(get_db)):
    """Get tag detection statistics."""
    results = db.query(Result).all()

    tag_stats = {}
    for result in results:
        if result.result_json and 'details' in result.result_json:
            for tag_name, details in result.result_json['details'].items():
                if tag_name not in tag_stats:
                    tag_stats[tag_name] = {
                        "detected_count": 0,
                        "total_count": 0,
                        "total_confidence": 0,
                        "pattern_matches": 0,
                        "semantic_only": 0
                    }

                tag_stats[tag_name]["total_count"] += 1

                confidence = details.get('confidence', 0)
                if confidence >= 0.45:  # Detection threshold
                    tag_stats[tag_name]["detected_count"] += 1
                    tag_stats[tag_name]["total_confidence"] += confidence

                    method = details.get('method', '')
                    if 'pattern' in method:
                        tag_stats[tag_name]["pattern_matches"] += 1
                    elif method == 'semantic':
                        tag_stats[tag_name]["semantic_only"] += 1

    # Calculate averages
    output = []
    for tag_name, stats in tag_stats.items():
        detection_rate = stats["detected_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
        avg_confidence = stats["total_confidence"] / stats["detected_count"] if stats["detected_count"] > 0 else 0

        output.append({
            "tag_name": tag_name,
            "detection_rate": round(detection_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "detected_count": stats["detected_count"],
            "total_documents": stats["total_count"],
            "pattern_matches": stats["pattern_matches"],
            "semantic_only": stats["semantic_only"]
        })

    # Sort by detection rate
    output.sort(key=lambda x: x["detection_rate"], reverse=True)
    return output
