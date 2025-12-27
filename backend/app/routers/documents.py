"""
Documents Router - Upload, process, and manage documents.
"""
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database.db import get_db, Document, Result, Matter

router = APIRouter()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOADS_DIR = DATA_DIR / "uploads"


# Pydantic schemas
class MatterBrief(BaseModel):
    id: str
    name: str
    matter_type: Optional[str]

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    file_size_bytes: Optional[int]
    page_count: Optional[int]
    word_count: Optional[int]
    status: str
    error_message: Optional[str]
    matter_id: Optional[str] = None
    matter: Optional[MatterBrief] = None
    average_confidence: Optional[float] = None

    class Config:
        from_attributes = True


class ProcessRequest(BaseModel):
    semantic_model: str = "intfloat/e5-large-v2"
    enable_vision: bool = False
    vision_model: str = "microsoft/Florence-2-base"


class ProcessingStatus(BaseModel):
    document_id: str
    status: str
    message: str


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique ID
    doc_id = str(uuid.uuid4())

    # Create document directory
    doc_dir = UPLOADS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = doc_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Get file size
    file_size = file_path.stat().st_size

    # Create database record
    document = Document(
        id=doc_id,
        filename=file.filename,
        filepath=str(file_path),
        file_size_bytes=file_size,
        status="uploaded"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return document


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    matter_id: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all documents with optional search and filtering."""
    query = db.query(Document)
    if status:
        query = query.filter(Document.status == status)
    if matter_id:
        query = query.filter(Document.matter_id == matter_id)

    # Search by filename or matter name
    if search:
        search_term = f"%{search}%"
        query = query.outerjoin(Matter).filter(
            (Document.filename.ilike(search_term)) |
            (Matter.name.ilike(search_term))
        )

    documents = query.order_by(Document.uploaded_at.desc()).offset(skip).limit(limit).all()

    # Build response with matter info and confidence
    response = []
    for doc in documents:
        # Get latest result for confidence
        latest_result = db.query(Result).filter(
            Result.document_id == doc.id
        ).order_by(Result.processed_at.desc()).first()

        doc_dict = {
            "id": doc.id,
            "filename": doc.filename,
            "uploaded_at": doc.uploaded_at,
            "file_size_bytes": doc.file_size_bytes,
            "page_count": doc.page_count,
            "word_count": doc.word_count,
            "status": doc.status,
            "error_message": doc.error_message,
            "matter_id": doc.matter_id,
            "matter": None,
            "average_confidence": latest_result.average_confidence if latest_result else None
        }
        if doc.matter_id:
            matter = db.query(Matter).filter(Matter.id == doc.matter_id).first()
            if matter:
                doc_dict["matter"] = {
                    "id": matter.id,
                    "name": matter.name,
                    "matter_type": matter.matter_type
                }
        response.append(doc_dict)

    return response


# ============ Review Queue Endpoints (must be before /{document_id}) ============

class ReviewQueueItem(BaseModel):
    id: str
    filename: str
    status: str  # ignored, failed, needs_review, low_confidence
    reason: str
    uploaded_at: datetime
    matter_id: Optional[str]
    matter_name: Optional[str]
    matter_type: Optional[str]
    file_size_bytes: Optional[int]
    recommended_pipeline: Optional[str]
    average_confidence: Optional[float]

    class Config:
        from_attributes = True

# Threshold for flagging low confidence documents
LOW_CONFIDENCE_THRESHOLD = 0.5


class RetryRequest(BaseModel):
    pipeline: Optional[str] = None  # fast, ocr, smart - None means auto
    force: bool = False  # Force retry even if already processed
    skip_validation: bool = False  # Skip quality checks (human override)


@router.get("/review-queue", response_model=List[ReviewQueueItem])
async def get_review_queue(
    status_filter: Optional[str] = None,  # ignored, failed, needs_review, low_confidence
    matter_id: Optional[str] = None,
    matter_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get documents that need human review.
    Includes: ignored, failed, needs_review, and low_confidence documents.
    """
    review_statuses = ['ignored', 'failed', 'needs_review']
    result = []

    # Handle low_confidence separately - these are completed docs with low scores
    if status_filter == 'low_confidence' or not status_filter:
        # Find completed documents with low confidence
        low_conf_query = db.query(Document).filter(Document.status == 'completed')
        if matter_id:
            low_conf_query = low_conf_query.filter(Document.matter_id == matter_id)
        if matter_type:
            low_conf_query = low_conf_query.join(Matter).filter(Matter.matter_type == matter_type)

        for doc in low_conf_query.all():
            # Get the latest result for this document
            latest_result = db.query(Result).filter(
                Result.document_id == doc.id
            ).order_by(Result.processed_at.desc()).first()

            if latest_result and latest_result.average_confidence is not None:
                if latest_result.average_confidence < LOW_CONFIDENCE_THRESHOLD:
                    matter = db.query(Matter).filter(Matter.id == doc.matter_id).first() if doc.matter_id else None
                    result.append(ReviewQueueItem(
                        id=doc.id,
                        filename=doc.filename,
                        status='low_confidence',
                        reason=f"Low confidence: {latest_result.average_confidence:.1%}",
                        uploaded_at=doc.uploaded_at,
                        matter_id=doc.matter_id,
                        matter_name=matter.name if matter else None,
                        matter_type=matter.matter_type if matter else None,
                        file_size_bytes=doc.file_size_bytes,
                        recommended_pipeline=doc.recommended_pipeline,
                        average_confidence=latest_result.average_confidence
                    ))

    # Get regular review queue items (ignored, failed, needs_review)
    if status_filter != 'low_confidence':
        query = db.query(Document)

        if status_filter and status_filter in review_statuses:
            query = query.filter(Document.status == status_filter)
        elif not status_filter:
            query = query.filter(Document.status.in_(review_statuses))

        if matter_id:
            query = query.filter(Document.matter_id == matter_id)
        if matter_type:
            query = query.join(Matter).filter(Matter.matter_type == matter_type)

        for doc in query.order_by(Document.uploaded_at.desc()).all():
            matter = db.query(Matter).filter(Matter.id == doc.matter_id).first() if doc.matter_id else None

            # Get confidence if available
            latest_result = db.query(Result).filter(
                Result.document_id == doc.id
            ).order_by(Result.processed_at.desc()).first()

            result.append(ReviewQueueItem(
                id=doc.id,
                filename=doc.filename,
                status=doc.status,
                reason=doc.error_message or "Unknown reason",
                uploaded_at=doc.uploaded_at,
                matter_id=doc.matter_id,
                matter_name=matter.name if matter else None,
                matter_type=matter.matter_type if matter else None,
                file_size_bytes=doc.file_size_bytes,
                recommended_pipeline=doc.recommended_pipeline,
                average_confidence=latest_result.average_confidence if latest_result else None
            ))

    # Apply pagination
    return result[skip:skip + limit]


@router.get("/review-queue/stats")
async def get_review_queue_stats(db: Session = Depends(get_db)):
    """Get summary stats for the review queue."""
    ignored = db.query(Document).filter(Document.status == 'ignored').count()
    failed = db.query(Document).filter(Document.status == 'failed').count()
    needs_review = db.query(Document).filter(Document.status == 'needs_review').count()

    # Count low confidence documents
    low_confidence = 0
    completed_docs = db.query(Document).filter(Document.status == 'completed').all()
    for doc in completed_docs:
        latest_result = db.query(Result).filter(
            Result.document_id == doc.id
        ).order_by(Result.processed_at.desc()).first()
        if latest_result and latest_result.average_confidence is not None:
            if latest_result.average_confidence < LOW_CONFIDENCE_THRESHOLD:
                low_confidence += 1

    return {
        "total": ignored + failed + needs_review + low_confidence,
        "ignored": ignored,
        "failed": failed,
        "needs_review": needs_review,
        "low_confidence": low_confidence
    }


@router.post("/review-queue/retry-all")
async def retry_all_in_queue(
    status_filter: Optional[str] = None,
    matter_id: Optional[str] = None,
    pipeline: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Retry all documents in the review queue."""
    from ..routers.matters import run_fast_pipeline, run_ocr_pipeline, run_smart_pipeline

    review_statuses = ['ignored', 'failed', 'needs_review']

    query = db.query(Document)

    if status_filter and status_filter in review_statuses:
        query = query.filter(Document.status == status_filter)
    else:
        query = query.filter(Document.status.in_(review_statuses))

    if matter_id:
        query = query.filter(Document.matter_id == matter_id)

    documents = query.all()

    if not documents:
        return {"message": "No documents to retry", "count": 0}

    doc_info = []
    for doc in documents:
        doc_pipeline = pipeline
        if not doc_pipeline:
            filepath = Path(doc.filepath)
            ext = filepath.suffix.lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
                doc_pipeline = 'ocr'
            elif doc.recommended_pipeline:
                doc_pipeline = doc.recommended_pipeline
            else:
                doc_pipeline = 'fast'

        doc_info.append((doc.id, doc.filepath, doc_pipeline))
        doc.status = "processing"
        doc.error_message = None

    db.commit()

    for doc_id, filepath, doc_pipeline in doc_info:
        if doc_pipeline == 'ocr':
            background_tasks.add_task(run_ocr_pipeline, doc_id, filepath)
        elif doc_pipeline == 'smart':
            background_tasks.add_task(run_smart_pipeline, doc_id, filepath)
        else:
            background_tasks.add_task(run_fast_pipeline, doc_id, filepath)

    return {
        "message": f"Queued {len(doc_info)} documents for retry",
        "count": len(doc_info)
    }


# ============ Individual Document Endpoints ============

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get a specific document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its files."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete files
    doc_dir = UPLOADS_DIR / document_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)

    # Delete from database (cascades to results)
    db.delete(document)
    db.commit()

    return {"message": "Document deleted", "id": document_id}


@router.post("/{document_id}/process", response_model=ProcessingStatus)
async def process_document(
    document_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process a document with the tagging pipeline."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if document.status == "processing":
        raise HTTPException(status_code=400, detail="Document is already being processed")

    # Update status
    document.status = "processing"
    db.commit()

    # Queue background processing
    background_tasks.add_task(
        run_pipeline,
        document_id,
        document.filepath,
        request.semantic_model,
        request.enable_vision,
        request.vision_model
    )

    return ProcessingStatus(
        document_id=document_id,
        status="processing",
        message="Document processing started"
    )


@router.get("/{document_id}/text")
async def get_document_text(document_id: str, db: Session = Depends(get_db)):
    """Get the extracted text content of a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    filepath = Path(document.filepath)
    ext = filepath.suffix.lower()

    try:
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif ext == '.pdf':
            # Try pdfplumber first
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    text = '\n\n'.join([p.extract_text() or '' for p in pdf.pages])
            except ImportError:
                from pypdf import PdfReader
                reader = PdfReader(str(filepath))
                text = '\n\n'.join([p.extract_text() or '' for p in reader.pages])
        else:
            # Try as text
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

        return {"document_id": document_id, "text": text, "filename": document.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


@router.get("/{document_id}/pages/{page_num}")
async def get_page_image(document_id: str, page_num: int, db: Session = Depends(get_db)):
    """Get a page image from a processed document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if this is a text file - no page images available
    filepath = Path(document.filepath)
    if filepath.suffix.lower() in ['.txt', '.text']:
        raise HTTPException(status_code=400, detail="Text files do not have page images")

    # Check for cached page image
    page_image_path = UPLOADS_DIR / document_id / f"page_{page_num}.png"
    if page_image_path.exists():
        return FileResponse(page_image_path)

    # Generate page image on demand
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            document.filepath,
            dpi=150,
            first_page=page_num,
            last_page=page_num
        )
        if images:
            images[0].save(page_image_path, "PNG")
            return FileResponse(page_image_path)
        else:
            raise HTTPException(status_code=404, detail="Page not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate page image: {str(e)}")


@router.get("/{document_id}/result")
async def get_document_result(document_id: str, db: Session = Depends(get_db)):
    """Get the processing result for a document."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    result = db.query(Result).filter(Result.document_id == document_id).order_by(Result.processed_at.desc()).first()
    if not result:
        raise HTTPException(status_code=404, detail="No processing results found")

    return {
        "id": result.id,
        "document_id": result.document_id,
        "processed_at": result.processed_at,
        "processing_time_seconds": result.processing_time_seconds,
        "semantic_model": result.semantic_model,
        "vision_model": result.vision_model,
        "vision_enabled": result.vision_enabled,
        "tag_count": result.tag_count,
        "average_confidence": result.average_confidence,
        "result_json": result.result_json
    }


@router.get("/{document_id}/visual-pages")
async def get_visual_pages(document_id: str, db: Session = Depends(get_db)):
    """Get the visual pages detected for a document."""
    result = db.query(Result).filter(Result.document_id == document_id).order_by(Result.processed_at.desc()).first()
    if not result:
        raise HTTPException(status_code=404, detail="No processing results found")

    return {
        "document_id": document_id,
        "visual_pages": result.visual_pages or {},
        "vision_enabled": result.vision_enabled
    }


@router.post("/{document_id}/dismiss")
async def dismiss_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Dismiss a document from the review queue.
    Sets status to 'dismissed' so it won't appear in the queue.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    document.status = "dismissed"
    document.error_message = "Dismissed by user"
    db.commit()

    return {"message": "Document dismissed", "document_id": document_id}


@router.post("/{document_id}/ignore")
async def ignore_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Mark a document as ignored (e.g., non-processable file types, code files).
    Removes it from failed/pending counts without deleting.
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    document.status = "ignored"
    document.error_message = "Ignored by user"
    db.commit()

    return {"message": "Document ignored", "document_id": document_id}


@router.post("/{document_id}/retry")
async def retry_document(
    document_id: str,
    request: RetryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Retry processing a skipped/failed document.
    Optionally specify a pipeline override (fast, ocr, smart).
    Also handles low_confidence documents (completed but need re-review).
    """
    import logging
    logger = logging.getLogger(__name__)

    from ..routers.matters import run_fast_pipeline, run_ocr_pipeline, run_smart_pipeline

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Allow retrying: ignored, failed, needs_review, uploaded, AND completed (for low confidence)
    allowed_statuses = ['ignored', 'failed', 'needs_review', 'uploaded', 'completed']
    if document.status not in allowed_statuses and not request.force:
        raise HTTPException(
            status_code=400,
            detail=f"Document is {document.status}. Use force=true to override."
        )

    logger.info(f"Retrying document {document_id} (was {document.status}) with pipeline {request.pipeline}")

    # Determine which pipeline to use
    pipeline = request.pipeline
    if not pipeline:
        # Auto-detect based on file type
        filepath = Path(document.filepath)
        ext = filepath.suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
            pipeline = 'ocr'
        elif document.recommended_pipeline:
            pipeline = document.recommended_pipeline
        else:
            pipeline = 'fast'

    # Update status to processing
    document.status = "processing"
    document.error_message = None
    db.commit()

    # Queue the appropriate pipeline
    if pipeline == 'ocr':
        background_tasks.add_task(run_ocr_pipeline, document_id, document.filepath)
    elif pipeline == 'smart':
        background_tasks.add_task(run_smart_pipeline, document_id, document.filepath)
    else:
        background_tasks.add_task(run_fast_pipeline, document_id, document.filepath, request.skip_validation)

    return {
        "message": f"Queued document for retry with {pipeline} pipeline",
        "document_id": document_id,
        "pipeline": pipeline
    }


def run_pipeline(
    document_id: str,
    filepath: str,
    semantic_model: str,
    enable_vision: bool,
    vision_model: str
):
    """Run the document tagging pipeline (background task)."""
    from ..database.db import SessionLocal, Document, Result, Model, ModelUsage
    from ..services.fast_tagger import process_document_fast
    import time

    db = SessionLocal()
    try:
        # Use fast_tagger for consistent output format with highlights
        result_data = process_document_fast(
            filepath,
            db,
            model_name=semantic_model,
            threshold=0.65
        )

        # Get page count
        file_ext = Path(filepath).suffix.lower()
        if file_ext == '.pdf':
            try:
                from pdf2image import pdfinfo_from_path
                pdf_info = pdfinfo_from_path(filepath)
                page_count = pdf_info.get('Pages', 0)
            except Exception:
                page_count = result_data.get('page_count', 0)
        else:
            page_count = 1  # Text files are single page

        # Create result record
        processing_time = result_data.get('processing_time_seconds', 0)
        result_id = str(uuid.uuid4())
        result = Result(
            id=result_id,
            document_id=document_id,
            processing_time_seconds=processing_time,
            semantic_model=semantic_model,
            vision_model=vision_model if enable_vision else None,
            vision_enabled=enable_vision,
            tag_count=result_data.get('tag_count', 0),
            average_confidence=result_data.get('average_confidence', 0),
            result_json=result_data,
            visual_pages=result_data.get('visual_pages')
        )
        db.add(result)

        # Get or create semantic model in registry
        model = db.query(Model).filter(Model.name == semantic_model).first()
        if not model:
            model = Model(
                id=str(uuid.uuid4()),
                name=semantic_model,
                type='semantic',
                huggingface_url=f'https://huggingface.co/{semantic_model}',
                approved=True
            )
            db.add(model)
            db.flush()

        # Create model usage record
        model_usage = ModelUsage(
            id=str(uuid.uuid4()),
            model_id=model.id,
            result_id=result_id,
            processing_time_seconds=processing_time
        )
        db.add(model_usage)

        # If vision model was used, track that too
        if enable_vision and vision_model:
            vision_model_record = db.query(Model).filter(Model.name == vision_model).first()
            if not vision_model_record:
                vision_model_record = Model(
                    id=str(uuid.uuid4()),
                    name=vision_model,
                    type='vision',
                    huggingface_url=f'https://huggingface.co/{vision_model}',
                    approved=True
                )
                db.add(vision_model_record)
                db.flush()
            vision_usage = ModelUsage(
                id=str(uuid.uuid4()),
                model_id=vision_model_record.id,
                result_id=result_id,
                processing_time_seconds=processing_time * 0.3  # Estimate vision portion
            )
            db.add(vision_usage)

        # Update document based on processing status
        doc_status = result_data.get('status', 'processed')
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            if doc_status == 'processed':
                document.status = "completed"
                document.word_count = result_data.get('word_count')
                document.page_count = page_count
                document.error_message = None
            elif doc_status == 'ignored':
                document.status = "ignored"
                document.error_message = result_data.get('status_reason', 'Unsupported file type')
            else:  # failed
                document.status = "failed"
                document.error_message = result_data.get('status_reason', 'Processing failed')

        db.commit()

    except Exception as e:
        # Mark as failed
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "failed"
            document.error_message = str(e)
            db.commit()

    finally:
        db.close()
