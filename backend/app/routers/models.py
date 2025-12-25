"""
Models Router - Model registry with HuggingFace integration.
"""
import uuid
import httpx
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from ..database.db import get_db, Model, ModelUsage

router = APIRouter()


class ModelCreate(BaseModel):
    name: str  # HuggingFace model name, e.g., "pile-of-law/legalbert-large-1.7M-2"
    type: str  # semantic, vision, ocr


class ModelUpdate(BaseModel):
    approved: Optional[bool] = None
    approved_by: Optional[str] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    type: str
    huggingface_url: Optional[str]
    size_gb: Optional[float]
    description: Optional[str]
    downloads: Optional[int]
    license: Optional[str]
    last_updated: Optional[datetime]
    approved: bool
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    created_at: datetime
    usage_count: int = 0

    class Config:
        from_attributes = True


class ModelUsageStats(BaseModel):
    model_id: str
    model_name: str
    total_usages: int
    total_processing_time: float
    avg_processing_time: float
    last_used: Optional[datetime]


async def fetch_huggingface_info(model_name: str) -> dict:
    """Fetch model metadata from HuggingFace API."""
    url = f"https://huggingface.co/api/models/{model_name}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return {
                    "huggingface_url": f"https://huggingface.co/{model_name}",
                    "description": data.get("description", "")[:500] if data.get("description") else None,
                    "downloads": data.get("downloads", 0),
                    "license": data.get("license"),
                    "last_updated": datetime.fromisoformat(data["lastModified"].replace("Z", "+00:00")) if data.get("lastModified") else None,
                    # Estimate size from safetensors/bin files
                    "size_gb": estimate_model_size(data.get("siblings", []))
                }
            return {}
    except Exception as e:
        print(f"Failed to fetch HuggingFace info for {model_name}: {e}")
        return {}


def estimate_model_size(siblings: list) -> Optional[float]:
    """Estimate model size from file list."""
    total_bytes = 0
    for file in siblings:
        if file.get("rfilename", "").endswith((".bin", ".safetensors", ".pt", ".pth")):
            # Size is in bytes
            size = file.get("size", 0)
            if size:
                total_bytes += size
    if total_bytes > 0:
        return round(total_bytes / (1024 ** 3), 2)  # Convert to GB
    return None


@router.post("", response_model=ModelResponse)
async def register_model(
    model: ModelCreate,
    db: Session = Depends(get_db)
):
    """Register a new model with HuggingFace metadata."""
    # Check if model already exists
    existing = db.query(Model).filter(Model.name == model.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already registered")

    # Fetch HuggingFace info
    hf_info = await fetch_huggingface_info(model.name)

    model_id = str(uuid.uuid4())
    db_model = Model(
        id=model_id,
        name=model.name,
        type=model.type,
        huggingface_url=hf_info.get("huggingface_url"),
        size_gb=hf_info.get("size_gb"),
        description=hf_info.get("description"),
        downloads=hf_info.get("downloads"),
        license=hf_info.get("license"),
        last_updated=hf_info.get("last_updated"),
        approved=False
    )

    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return ModelResponse(
        **{**db_model.__dict__, "usage_count": 0}
    )


@router.get("", response_model=List[ModelResponse])
async def list_models(
    type: Optional[str] = None,
    approved_only: bool = False,
    db: Session = Depends(get_db)
):
    """List all registered models."""
    query = db.query(Model)

    if type:
        query = query.filter(Model.type == type)
    if approved_only:
        query = query.filter(Model.approved == True)

    models = query.order_by(Model.created_at.desc()).all()

    # Get usage counts
    result = []
    for model in models:
        usage_count = db.query(ModelUsage).filter(ModelUsage.model_id == model.id).count()
        result.append(ModelResponse(
            **{**model.__dict__, "usage_count": usage_count}
        ))

    return result


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get model details."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    usage_count = db.query(ModelUsage).filter(ModelUsage.model_id == model_id).count()

    return ModelResponse(
        **{**model.__dict__, "usage_count": usage_count}
    )


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    update: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Update model (approval status, etc.)."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if update.approved is not None:
        model.approved = update.approved
        if update.approved:
            model.approved_at = datetime.utcnow()
            model.approved_by = update.approved_by

    db.commit()
    db.refresh(model)

    usage_count = db.query(ModelUsage).filter(ModelUsage.model_id == model_id).count()

    return ModelResponse(
        **{**model.__dict__, "usage_count": usage_count}
    )


@router.delete("/{model_id}")
async def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Delete a model from the registry."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model)
    db.commit()

    return {"message": "Model deleted", "id": model_id}


@router.get("/{model_id}/usage", response_model=ModelUsageStats)
async def get_model_usage(model_id: str, db: Session = Depends(get_db)):
    """Get usage statistics for a model."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    usages = db.query(ModelUsage).filter(ModelUsage.model_id == model_id).all()

    if not usages:
        return ModelUsageStats(
            model_id=model_id,
            model_name=model.name,
            total_usages=0,
            total_processing_time=0,
            avg_processing_time=0,
            last_used=None
        )

    total_time = sum(u.processing_time_seconds or 0 for u in usages)
    last_usage = max(usages, key=lambda u: u.used_at)

    return ModelUsageStats(
        model_id=model_id,
        model_name=model.name,
        total_usages=len(usages),
        total_processing_time=total_time,
        avg_processing_time=total_time / len(usages) if usages else 0,
        last_used=last_usage.used_at
    )


@router.post("/{model_id}/refresh")
async def refresh_model_info(model_id: str, db: Session = Depends(get_db)):
    """Refresh model info from HuggingFace."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    hf_info = await fetch_huggingface_info(model.name)

    if hf_info:
        model.huggingface_url = hf_info.get("huggingface_url", model.huggingface_url)
        model.size_gb = hf_info.get("size_gb", model.size_gb)
        model.description = hf_info.get("description", model.description)
        model.downloads = hf_info.get("downloads", model.downloads)
        model.license = hf_info.get("license", model.license)
        model.last_updated = hf_info.get("last_updated", model.last_updated)
        db.commit()

    return {"message": "Model info refreshed", "id": model_id}
