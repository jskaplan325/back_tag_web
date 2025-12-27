"""
Annotations Router - CRUD for bounding box annotations on document pages.
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database.db import get_db, Document, Annotation, Tag, AreaOfLaw

router = APIRouter()


# Pydantic schemas
class AnnotationCreate(BaseModel):
    page_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    tag_name: str
    tag_id: Optional[str] = None
    area_of_law: Optional[str] = None
    annotation_type: str = "positive"  # positive, negative, uncertain
    color: str = "green"  # green, yellow, red
    notes: Optional[str] = None
    created_by: Optional[str] = None


class AnnotationUpdate(BaseModel):
    tag_name: Optional[str] = None
    tag_id: Optional[str] = None
    annotation_type: Optional[str] = None
    color: Optional[str] = None
    notes: Optional[str] = None


class AnnotationResponse(BaseModel):
    id: str
    document_id: str
    page_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    tag_name: str
    tag_id: Optional[str]
    area_of_law: Optional[str]
    annotation_type: str
    color: str
    source: str
    created_by: Optional[str]
    created_at: datetime
    notes: Optional[str]


class TagSearchResult(BaseModel):
    id: str
    name: str
    area_of_law: str
    area_of_law_id: str


@router.get("/documents/{document_id}/annotations", response_model=List[AnnotationResponse])
async def get_document_annotations(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get all annotations for a document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    annotations = db.query(Annotation).filter(
        Annotation.document_id == document_id
    ).order_by(Annotation.page_number, Annotation.created_at).all()

    return [
        AnnotationResponse(
            id=ann.id,
            document_id=ann.document_id,
            page_number=ann.page_number,
            x1=ann.x1,
            y1=ann.y1,
            x2=ann.x2,
            y2=ann.y2,
            tag_name=ann.tag_name,
            tag_id=ann.tag_id,
            area_of_law=ann.area_of_law,
            annotation_type=ann.annotation_type,
            color=ann.color,
            source=ann.source,
            created_by=ann.created_by,
            created_at=ann.created_at,
            notes=ann.notes
        )
        for ann in annotations
    ]


@router.get("/documents/{document_id}/pages/{page}/annotations", response_model=List[AnnotationResponse])
async def get_page_annotations(
    document_id: str,
    page: int,
    db: Session = Depends(get_db)
):
    """Get annotations for a specific page."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    annotations = db.query(Annotation).filter(
        Annotation.document_id == document_id,
        Annotation.page_number == page
    ).order_by(Annotation.created_at).all()

    return [
        AnnotationResponse(
            id=ann.id,
            document_id=ann.document_id,
            page_number=ann.page_number,
            x1=ann.x1,
            y1=ann.y1,
            x2=ann.x2,
            y2=ann.y2,
            tag_name=ann.tag_name,
            tag_id=ann.tag_id,
            area_of_law=ann.area_of_law,
            annotation_type=ann.annotation_type,
            color=ann.color,
            source=ann.source,
            created_by=ann.created_by,
            created_at=ann.created_at,
            notes=ann.notes
        )
        for ann in annotations
    ]


@router.post("/documents/{document_id}/annotations", response_model=AnnotationResponse)
async def create_annotation(
    document_id: str,
    data: AnnotationCreate,
    db: Session = Depends(get_db)
):
    """Create a new annotation."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate coordinates are normalized (0-1)
    for coord_name, coord_value in [("x1", data.x1), ("y1", data.y1), ("x2", data.x2), ("y2", data.y2)]:
        if not 0 <= coord_value <= 1:
            raise HTTPException(
                status_code=400,
                detail=f"Coordinate {coord_name} must be normalized (0-1), got {coord_value}"
            )

    # Look up area_of_law from tag if tag_id provided
    area_of_law = data.area_of_law
    if data.tag_id and not area_of_law:
        tag = db.query(Tag).filter(Tag.id == data.tag_id).first()
        if tag:
            aol = db.query(AreaOfLaw).filter(AreaOfLaw.id == tag.area_of_law_id).first()
            if aol:
                area_of_law = aol.name

    annotation = Annotation(
        id=str(uuid.uuid4()),
        document_id=document_id,
        page_number=data.page_number,
        x1=data.x1,
        y1=data.y1,
        x2=data.x2,
        y2=data.y2,
        tag_name=data.tag_name,
        tag_id=data.tag_id,
        area_of_law=area_of_law,
        annotation_type=data.annotation_type,
        color=data.color,
        source="human",
        created_by=data.created_by,
        created_at=datetime.utcnow(),
        notes=data.notes
    )

    db.add(annotation)
    db.commit()
    db.refresh(annotation)

    return AnnotationResponse(
        id=annotation.id,
        document_id=annotation.document_id,
        page_number=annotation.page_number,
        x1=annotation.x1,
        y1=annotation.y1,
        x2=annotation.x2,
        y2=annotation.y2,
        tag_name=annotation.tag_name,
        tag_id=annotation.tag_id,
        area_of_law=annotation.area_of_law,
        annotation_type=annotation.annotation_type,
        color=annotation.color,
        source=annotation.source,
        created_by=annotation.created_by,
        created_at=annotation.created_at,
        notes=annotation.notes
    )


@router.patch("/annotations/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(
    annotation_id: str,
    data: AnnotationUpdate,
    db: Session = Depends(get_db)
):
    """Update an annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    if data.tag_name is not None:
        annotation.tag_name = data.tag_name
    if data.tag_id is not None:
        annotation.tag_id = data.tag_id
    if data.annotation_type is not None:
        annotation.annotation_type = data.annotation_type
    if data.color is not None:
        annotation.color = data.color
    if data.notes is not None:
        annotation.notes = data.notes

    db.commit()
    db.refresh(annotation)

    return AnnotationResponse(
        id=annotation.id,
        document_id=annotation.document_id,
        page_number=annotation.page_number,
        x1=annotation.x1,
        y1=annotation.y1,
        x2=annotation.x2,
        y2=annotation.y2,
        tag_name=annotation.tag_name,
        tag_id=annotation.tag_id,
        area_of_law=annotation.area_of_law,
        annotation_type=annotation.annotation_type,
        color=annotation.color,
        source=annotation.source,
        created_by=annotation.created_by,
        created_at=annotation.created_at,
        notes=annotation.notes
    )


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(
    annotation_id: str,
    db: Session = Depends(get_db)
):
    """Delete an annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(annotation)
    db.commit()

    return {"status": "ok", "message": "Annotation deleted"}


@router.get("/tags/search", response_model=List[TagSearchResult])
async def search_tags(
    q: str = "",
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Search tags for autocomplete."""
    query = db.query(Tag).join(AreaOfLaw)

    if q:
        query = query.filter(Tag.name.ilike(f"%{q}%"))

    tags = query.limit(limit).all()

    results = []
    for tag in tags:
        aol = db.query(AreaOfLaw).filter(AreaOfLaw.id == tag.area_of_law_id).first()
        results.append(TagSearchResult(
            id=tag.id,
            name=tag.name,
            area_of_law=aol.name if aol else "Unknown",
            area_of_law_id=tag.area_of_law_id
        ))

    return results


@router.get("/annotations/stats")
async def get_annotation_stats(
    db: Session = Depends(get_db)
):
    """Get annotation statistics for training insights."""
    all_annotations = db.query(Annotation).all()

    # Group by tag
    stats_by_tag = {}
    for ann in all_annotations:
        key = ann.tag_name
        if key not in stats_by_tag:
            stats_by_tag[key] = {
                "positive": 0,
                "negative": 0,
                "uncertain": 0,
                "total": 0
            }

        stats_by_tag[key][ann.annotation_type] = stats_by_tag[key].get(ann.annotation_type, 0) + 1
        stats_by_tag[key]["total"] += 1

    # Group by area of law
    stats_by_aol = {}
    for ann in all_annotations:
        aol = ann.area_of_law or "Unknown"
        if aol not in stats_by_aol:
            stats_by_aol[aol] = 0
        stats_by_aol[aol] += 1

    return {
        "total_annotations": len(all_annotations),
        "by_tag": stats_by_tag,
        "by_area_of_law": stats_by_aol
    }


@router.get("/training/summary")
async def get_training_summary(
    db: Session = Depends(get_db)
):
    """Get comprehensive training data summary."""
    from sqlalchemy import func, distinct

    # Total annotations
    total_annotations = db.query(func.count(Annotation.id)).scalar() or 0

    # Documents with annotations
    annotated_doc_count = db.query(func.count(distinct(Annotation.document_id))).scalar() or 0

    # Total documents for comparison
    total_docs = db.query(func.count(Document.id)).scalar() or 0

    # Breakdown by annotation type
    type_counts = db.query(
        Annotation.annotation_type,
        func.count(Annotation.id)
    ).group_by(Annotation.annotation_type).all()

    by_type = {t: c for t, c in type_counts}

    # Breakdown by color
    color_counts = db.query(
        Annotation.color,
        func.count(Annotation.id)
    ).group_by(Annotation.color).all()

    by_color = {c: cnt for c, cnt in color_counts}

    # Top tags by annotation count
    tag_counts = db.query(
        Annotation.tag_name,
        Annotation.area_of_law,
        func.count(Annotation.id).label('count')
    ).group_by(Annotation.tag_name, Annotation.area_of_law).order_by(
        func.count(Annotation.id).desc()
    ).limit(20).all()

    top_tags = [
        {"tag": t, "area_of_law": aol or "Unknown", "count": c}
        for t, aol, c in tag_counts
    ]

    # Ignore regions count
    ignore_count = db.query(func.count(Annotation.id)).filter(
        Annotation.tag_name == "__IGNORE__"
    ).scalar() or 0

    return {
        "total_annotations": total_annotations,
        "annotated_documents": annotated_doc_count,
        "total_documents": total_docs,
        "coverage_percent": round((annotated_doc_count / total_docs * 100) if total_docs > 0 else 0, 1),
        "by_type": {
            "positive": by_type.get("positive", 0),
            "negative": by_type.get("negative", 0),
            "uncertain": by_type.get("uncertain", 0)
        },
        "by_color": {
            "green": by_color.get("green", 0),
            "yellow": by_color.get("yellow", 0),
            "red": by_color.get("red", 0)
        },
        "ignore_regions": ignore_count,
        "top_tags": top_tags
    }


@router.get("/training/documents")
async def get_annotated_documents(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get list of documents with annotations for training review."""
    from sqlalchemy import func, case, distinct

    # Subquery for annotation counts per document with color breakdown
    subq = db.query(
        Annotation.document_id,
        func.count(Annotation.id).label('annotation_count'),
        func.sum(case((Annotation.color == 'green', 1), else_=0)).label('green_count'),
        func.sum(case((Annotation.color == 'yellow', 1), else_=0)).label('yellow_count'),
        func.sum(case((Annotation.color == 'red', 1), else_=0)).label('red_count'),
        func.max(Annotation.created_at).label('last_annotated')
    ).group_by(Annotation.document_id).subquery()

    # Join with documents
    results = db.query(
        Document,
        subq.c.annotation_count,
        subq.c.green_count,
        subq.c.yellow_count,
        subq.c.red_count,
        subq.c.last_annotated
    ).join(
        subq, Document.id == subq.c.document_id
    ).order_by(
        subq.c.last_annotated.desc()
    ).offset(skip).limit(limit).all()

    # Total count for pagination
    total = db.query(func.count(distinct(Annotation.document_id))).scalar() or 0

    return {
        "total": total,
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "matter_id": doc.matter_id,
                "status": doc.status,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "annotation_count": ann_count,
                "green_count": green or 0,
                "yellow_count": yellow or 0,
                "red_count": red or 0,
                "last_annotated": last_ann.isoformat() if last_ann else None
            }
            for doc, ann_count, green, yellow, red, last_ann in results
        ]
    }


@router.get("/training/export")
async def export_training_data(
    format: str = "json",
    include_ignore: bool = False,
    db: Session = Depends(get_db)
):
    """Export all annotations as training data."""
    from fastapi.responses import JSONResponse

    query = db.query(Annotation).join(Document)

    if not include_ignore:
        query = query.filter(Annotation.tag_name != "__IGNORE__")

    annotations = query.all()

    # Build export structure
    export_data = []
    for ann in annotations:
        doc = db.query(Document).filter(Document.id == ann.document_id).first()

        export_data.append({
            "annotation_id": ann.id,
            "document_id": ann.document_id,
            "document_filename": doc.filename if doc else None,
            "page_number": ann.page_number,
            "bounding_box": {
                "x1": ann.x1,
                "y1": ann.y1,
                "x2": ann.x2,
                "y2": ann.y2
            },
            "tag_name": ann.tag_name,
            "tag_id": ann.tag_id,
            "area_of_law": ann.area_of_law,
            "annotation_type": ann.annotation_type,
            "color": ann.color,
            "source": ann.source,
            "created_at": ann.created_at.isoformat() if ann.created_at else None,
            "created_by": ann.created_by
        })

    if format == "json":
        return JSONResponse(
            content={
                "export_date": datetime.utcnow().isoformat(),
                "total_annotations": len(export_data),
                "annotations": export_data
            },
            headers={
                "Content-Disposition": "attachment; filename=training_data.json"
            }
        )

    # CSV format
    if format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse

        output = io.StringIO()
        if export_data:
            writer = csv.DictWriter(output, fieldnames=[
                "annotation_id", "document_id", "document_filename", "page_number",
                "x1", "y1", "x2", "y2", "tag_name", "tag_id", "area_of_law",
                "annotation_type", "color", "source", "created_at", "created_by"
            ])
            writer.writeheader()
            for item in export_data:
                flat = {
                    **{k: v for k, v in item.items() if k != "bounding_box"},
                    "x1": item["bounding_box"]["x1"],
                    "y1": item["bounding_box"]["y1"],
                    "x2": item["bounding_box"]["x2"],
                    "y2": item["bounding_box"]["y2"],
                }
                writer.writerow(flat)

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=training_data.csv"}
        )

    return {"error": "Invalid format. Use 'json' or 'csv'"}
