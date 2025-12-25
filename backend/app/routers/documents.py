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
    semantic_model: str = "pile-of-law/legalbert-large-1.7M-2"
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


def run_pipeline(
    document_id: str,
    filepath: str,
    semantic_model: str,
    enable_vision: bool,
    vision_model: str
):
    """Run the document tagging pipeline (background task)."""
    from ..database.db import SessionLocal, Document, Result, Model, ModelUsage
    import time

    db = SessionLocal()
    try:
        # Try to import back_tag - it should be pip installed
        try:
            from back_tag import DocumentTagger
        except ImportError:
            # Fallback: add to path if not installed
            import sys
            back_tag_path = Path(__file__).parent.parent.parent.parent.parent / "back_tag"
            if str(back_tag_path) not in sys.path:
                sys.path.insert(0, str(back_tag_path))
            from back_tag import DocumentTagger

        # Check file type
        file_ext = Path(filepath).suffix.lower()

        if file_ext == '.txt':
            # Handle text files directly
            start_time = time.time()

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()

            word_count = len(text_content.split())

            # Initialize tagger for text processing only
            tagger = DocumentTagger(
                semantic_model=semantic_model,
                use_gpu=True,
                enable_vision=False,  # No vision for text files
                vision_model=None
            )

            # Process text directly using the tagger's text processing
            result_data = tagger.process_text(text_content)
            result_data['processing_time_seconds'] = time.time() - start_time
            result_data['word_count'] = word_count
            page_count = 1  # Text files are single page

        else:
            # Initialize tagger for PDF processing
            tagger = DocumentTagger(
                semantic_model=semantic_model,
                use_gpu=True,
                enable_vision=enable_vision,
                vision_model=vision_model
            )

            # Process document (PDF)
            result_data = tagger.process_document(filepath)

            # Get page count from PDF
            try:
                from pdf2image import pdfinfo_from_path
                pdf_info = pdfinfo_from_path(filepath)
                page_count = pdf_info.get('Pages', 0)
            except Exception:
                page_count = result_data.get('page_count', 0)

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

        # Update document
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "completed"
            document.word_count = result_data.get('word_count')
            document.page_count = page_count

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
