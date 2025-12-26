"""
Matters Router - Manage matters and bulk import documents.
"""
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..database.db import get_db, Matter, Document

router = APIRouter()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
UPLOADS_DIR = DATA_DIR / "uploads"

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx'}


# Pydantic schemas
class MatterResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    matter_type: Optional[str]
    source_path: Optional[str]
    created_at: datetime
    document_count: int
    pending_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    average_confidence: Optional[float] = None

    class Config:
        from_attributes = True


class MatterCreate(BaseModel):
    name: str
    description: Optional[str] = None
    matter_type: Optional[str] = None


class DocumentInMatter(BaseModel):
    id: str
    filename: str
    status: str
    uploaded_at: datetime
    file_size_bytes: Optional[int]
    recommended_pipeline: Optional[str] = None

    class Config:
        from_attributes = True


class MatterDetailResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    matter_type: Optional[str]
    source_path: Optional[str]
    created_at: datetime
    document_count: int
    documents: List[DocumentInMatter]

    class Config:
        from_attributes = True


class BulkImportRequest(BaseModel):
    folder_path: str
    selected_folders: Optional[List[str]] = None  # If None, import all non-empty folders
    type_overrides: Optional[dict] = None  # Map of folder_path -> matter_type override
    name_overrides: Optional[dict] = None  # Map of folder_path -> matter_name (for nested folders)


class BulkImportResult(BaseModel):
    matters_created: int
    documents_imported: int
    matters: List[MatterResponse]
    errors: List[str]


class FolderScanResult(BaseModel):
    folder_path: str
    subfolders: List[dict]
    total_documents: int


@router.get("", response_model=List[MatterResponse])
async def list_matters(
    skip: int = 0,
    limit: int = 100,
    matter_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all matters with document status counts and average confidence."""
    from ..database.db import Result

    query = db.query(Matter)
    if matter_type:
        query = query.filter(Matter.matter_type == matter_type)
    matters = query.order_by(Matter.created_at.desc()).offset(skip).limit(limit).all()

    # Compute status counts and confidence for each matter
    result = []
    for matter in matters:
        docs = db.query(Document).filter(Document.matter_id == matter.id).all()
        pending = sum(1 for d in docs if d.status == 'uploaded')
        completed = sum(1 for d in docs if d.status == 'completed')
        failed = sum(1 for d in docs if d.status == 'failed')

        # Calculate average confidence from completed documents
        total_confidence = 0
        confidence_count = 0
        for doc in docs:
            if doc.status == 'completed':
                doc_result = db.query(Result).filter(Result.document_id == doc.id).order_by(Result.processed_at.desc()).first()
                if doc_result and doc_result.average_confidence:
                    total_confidence += doc_result.average_confidence
                    confidence_count += 1

        avg_confidence = round(total_confidence / confidence_count, 3) if confidence_count > 0 else None

        result.append(MatterResponse(
            id=matter.id,
            name=matter.name,
            description=matter.description,
            matter_type=matter.matter_type,
            source_path=matter.source_path,
            created_at=matter.created_at,
            document_count=matter.document_count,
            pending_count=pending,
            completed_count=completed,
            failed_count=failed,
            average_confidence=avg_confidence
        ))

    return result


@router.post("", response_model=MatterResponse)
async def create_matter(
    matter: MatterCreate,
    db: Session = Depends(get_db)
):
    """Create a new matter."""
    matter_id = str(uuid.uuid4())
    db_matter = Matter(
        id=matter_id,
        name=matter.name,
        description=matter.description,
        matter_type=matter.matter_type,
        document_count=0
    )
    db.add(db_matter)
    db.commit()
    db.refresh(db_matter)
    return db_matter


@router.get("/{matter_id}", response_model=MatterDetailResponse)
async def get_matter(matter_id: str, db: Session = Depends(get_db)):
    """Get a specific matter with its documents."""
    matter = db.query(Matter).filter(Matter.id == matter_id).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    documents = db.query(Document).filter(Document.matter_id == matter_id).all()

    return {
        "id": matter.id,
        "name": matter.name,
        "description": matter.description,
        "matter_type": matter.matter_type,
        "source_path": matter.source_path,
        "created_at": matter.created_at,
        "document_count": len(documents),
        "documents": documents
    }


@router.get("/{matter_id}/failed-documents")
async def get_failed_documents(matter_id: str, db: Session = Depends(get_db)):
    """Get failed documents for a matter with error messages."""
    matter = db.query(Matter).filter(Matter.id == matter_id).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    failed_docs = db.query(Document).filter(
        Document.matter_id == matter_id,
        Document.status == "failed"
    ).all()

    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "error_message": doc.error_message
        }
        for doc in failed_docs
    ]


@router.post("/{matter_id}/retry-failed")
async def retry_failed_documents(
    matter_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Reset failed documents to uploaded and queue for reprocessing."""
    matter = db.query(Matter).filter(Matter.id == matter_id).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    # Get failed documents
    failed_docs = db.query(Document).filter(
        Document.matter_id == matter_id,
        Document.status == "failed"
    ).all()

    if not failed_docs:
        return {"message": "No failed documents to retry", "count": 0}

    # Collect info and reset status
    doc_info = [(doc.id, doc.filepath, doc.recommended_pipeline or 'fast') for doc in failed_docs]

    for doc in failed_docs:
        doc.status = "processing"
        doc.error_message = None
    db.commit()

    # Queue for reprocessing using auto pipeline
    for doc_id, filepath, recommended in doc_info:
        background_tasks.add_task(run_auto_pipeline, doc_id, filepath, recommended)

    return {
        "message": f"Queued {len(doc_info)} documents for retry",
        "count": len(doc_info)
    }


@router.get("/{matter_id}/tags")
async def get_matter_tags(matter_id: str, limit: int = 5, db: Session = Depends(get_db)):
    """Get aggregate top tags for a matter based on document processing results."""
    from ..database.db import Result

    matter = db.query(Matter).filter(Matter.id == matter_id).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    # Get all documents for this matter
    documents = db.query(Document).filter(Document.matter_id == matter_id).all()
    doc_ids = [d.id for d in documents]

    # Get all results for these documents
    results = db.query(Result).filter(Result.document_id.in_(doc_ids)).all()

    # Aggregate tags from result_json
    tag_scores: dict = {}
    for result in results:
        if result.result_json and 'tags' in result.result_json:
            for tag in result.result_json['tags']:
                if isinstance(tag, dict):
                    name = tag.get('name', tag.get('tag_name', tag.get('tag', '')))
                    score = tag.get('confidence', tag.get('score', tag.get('semantic_score', 0)))
                else:
                    name = str(tag)
                    score = 0.5

                if name:
                    if name not in tag_scores:
                        tag_scores[name] = {'total_score': 0, 'count': 0}
                    tag_scores[name]['total_score'] += score
                    tag_scores[name]['count'] += 1

    # Calculate average scores and sort
    aggregated = []
    for name, data in tag_scores.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        aggregated.append({
            'tag': name,
            'average_confidence': avg_score,
            'document_count': data['count']
        })

    # Sort by count first, then by confidence
    aggregated.sort(key=lambda x: (x['document_count'], x['average_confidence']), reverse=True)

    return {
        'matter_id': matter_id,
        'total_documents': len(documents),
        'processed_documents': len(results),
        'top_tags': aggregated[:limit]
    }


@router.delete("/{matter_id}")
async def delete_matter(matter_id: str, db: Session = Depends(get_db)):
    """Delete a matter and all its documents."""
    matter = db.query(Matter).filter(Matter.id == matter_id).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    # Delete all document files
    documents = db.query(Document).filter(Document.matter_id == matter_id).all()
    for doc in documents:
        doc_dir = UPLOADS_DIR / doc.id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

    # Delete matter (cascades to documents)
    db.delete(matter)
    db.commit()

    return {"message": "Matter deleted", "id": matter_id}


class BrowseFolderRequest(BaseModel):
    path: Optional[str] = None  # None = list root/home


class BrowseFolderResult(BaseModel):
    current_path: str
    parent_path: Optional[str]
    folders: List[dict]  # name, path


@router.post("/browse-folder", response_model=BrowseFolderResult)
async def browse_folder(request: BrowseFolderRequest):
    """Browse folders on the server filesystem."""
    import platform

    # Determine starting path
    if not request.path:
        # Start at user's home directory
        current = Path.home()
    else:
        current = Path(request.path)

    if not current.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.path}")

    if not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.path}")

    # Get parent (None if at root)
    parent = str(current.parent) if current.parent != current else None

    # List subdirectories
    folders = []
    try:
        for item in sorted(current.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                folders.append({
                    "name": item.name,
                    "path": str(item)
                })
    except PermissionError:
        pass  # Skip folders we can't access

    return BrowseFolderResult(
        current_path=str(current),
        parent_path=parent,
        folders=folders
    )


@router.post("/scan-folder", response_model=FolderScanResult)
async def scan_folder(request: BulkImportRequest, db: Session = Depends(get_db)):
    """Scan a folder to preview what would be imported."""
    folder_path = Path(request.folder_path)

    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.folder_path}")

    # Get already imported paths
    existing_matters = db.query(Matter.source_path).all()
    imported_paths = {m[0] for m in existing_matters if m[0]}

    subfolders = []
    total_documents = 0

    for item in sorted(folder_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this folder has direct files or only subfolders
            direct_files = [f for f in item.iterdir()
                          if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
            child_dirs = [d for d in item.iterdir()
                         if d.is_dir() and not d.name.startswith('.')]

            if direct_files:
                # Folder has direct files - treat as single matter
                doc_count = len(direct_files)
                # Also count files in subfolders
                for file in item.rglob('*'):
                    if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS and file.parent != item:
                        doc_count += 1

                matter_type = infer_matter_type(item.name)
                item_path = str(item)
                already_imported = item_path in imported_paths
                selected = doc_count > 0 and not already_imported

                subfolders.append({
                    "name": item.name,
                    "path": item_path,
                    "document_count": doc_count,
                    "matter_type": matter_type,
                    "selected": selected,
                    "already_imported": already_imported
                })

                if selected:
                    total_documents += doc_count
            elif child_dirs:
                # Folder only has subfolders - each subfolder becomes a matter
                for child in sorted(child_dirs):
                    doc_count = 0
                    for file in child.rglob('*'):
                        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                            doc_count += 1

                    # Name as parent_child
                    matter_name = f"{item.name}_{child.name}"
                    matter_type = infer_matter_type(matter_name)
                    child_path = str(child)
                    already_imported = child_path in imported_paths
                    selected = doc_count > 0 and not already_imported

                    subfolders.append({
                        "name": matter_name,
                        "path": child_path,
                        "document_count": doc_count,
                        "matter_type": matter_type,
                        "selected": selected,
                        "already_imported": already_imported
                    })

                    if selected:
                        total_documents += doc_count

    return FolderScanResult(
        folder_path=str(folder_path),
        subfolders=subfolders,
        total_documents=total_documents
    )


@router.post("/bulk-import", response_model=BulkImportResult)
async def bulk_import(
    request: BulkImportRequest,
    db: Session = Depends(get_db)
):
    """Bulk import documents from a folder. Each subfolder becomes a matter."""
    folder_path = Path(request.folder_path)

    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.folder_path}")

    # Convert selected folders to a set for fast lookup
    selected_set = set(request.selected_folders) if request.selected_folders else None

    matters_created = []
    total_documents = 0
    errors = []

    # Get overrides
    type_overrides = request.type_overrides or {}
    name_overrides = request.name_overrides or {}

    # Import each selected folder
    if selected_set:
        for folder_path_str in selected_set:
            item = Path(folder_path_str)
            if not item.exists() or not item.is_dir():
                errors.append(f"Folder not found: {folder_path_str}")
                continue

            try:
                override_type = type_overrides.get(folder_path_str)
                override_name = name_overrides.get(folder_path_str)
                matter, doc_count = import_matter_folder(db, item, override_type, override_name)
                if matter:
                    matters_created.append(matter)
                    total_documents += doc_count
            except Exception as e:
                errors.append(f"Error importing {item.name}: {str(e)}")
    else:
        # No selection - import all top-level folders
        for item in sorted(folder_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    override_type = type_overrides.get(str(item))
                    override_name = name_overrides.get(str(item))
                    matter, doc_count = import_matter_folder(db, item, override_type, override_name)
                    if matter:
                        matters_created.append(matter)
                        total_documents += doc_count
                except Exception as e:
                    errors.append(f"Error importing {item.name}: {str(e)}")

    return BulkImportResult(
        matters_created=len(matters_created),
        documents_imported=total_documents,
        matters=matters_created,
        errors=errors
    )


def infer_matter_type(folder_name: str) -> str:
    """
    Infer matter type from folder name.

    TODO: This is a placeholder. In the future, this will be populated from
    an internal MatterDB that maps matter IDs to their Area of Law.

    For now, returns 'TBD' since matter names are typically numeric IDs
    like "Matter3323242" that don't indicate the practice area.
    """
    # Future: Query MatterDB to get the actual Area of Law for this matter ID
    # matter_info = matter_db.lookup(folder_name)
    # if matter_info:
    #     return matter_info.area_of_law

    return 'TBD'


def import_matter_folder(db: Session, folder_path: Path, type_override: Optional[str] = None, name_override: Optional[str] = None) -> tuple[Optional[Matter], int]:
    """Import a single folder as a matter with its documents. Overwrites if already exists."""
    # Get list of supported files (recursive)
    files = [f for f in folder_path.rglob('*')
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        return None, 0

    # Use name override if provided, otherwise use folder name
    matter_name = name_override if name_override else folder_path.name

    # Use type override if provided, otherwise infer from matter name
    matter_type = type_override if type_override else infer_matter_type(matter_name)

    # Check if matter already exists (by source_path)
    existing_matter = db.query(Matter).filter(Matter.source_path == str(folder_path)).first()

    if existing_matter:
        # Overwrite: Delete old documents and results, update matter
        from ..database.db import Result

        # Delete old documents (cascades to results via relationship)
        old_docs = db.query(Document).filter(Document.matter_id == existing_matter.id).all()
        for doc in old_docs:
            # Clean up files
            doc_dir = UPLOADS_DIR / doc.id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
            db.delete(doc)

        # Update matter
        existing_matter.matter_type = matter_type
        existing_matter.document_count = len(files)
        matter = existing_matter
        matter_id = existing_matter.id
    else:
        # Create new matter
        matter_id = str(uuid.uuid4())
        matter = Matter(
            id=matter_id,
            name=matter_name,
            matter_type=matter_type,
            source_path=str(folder_path),
            document_count=len(files)
        )
        db.add(matter)

    # Import each document
    for file_path in files:
        doc_id = str(uuid.uuid4())

        # Create document directory
        doc_dir = UPLOADS_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to uploads directory
        dest_path = doc_dir / file_path.name
        shutil.copy2(file_path, dest_path)

        # Get file size
        file_size = dest_path.stat().st_size

        # Count words for text files
        word_count = None
        if file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    word_count = len(content.split())
            except Exception:
                pass

        # Analyze document for pipeline recommendation
        from ..services.document_analyzer import analyze_document, analysis_to_dict
        try:
            analysis = analyze_document(str(dest_path))
            recommended_pipeline = analysis.recommended_pipeline
            analysis_metadata = analysis_to_dict(analysis)
            # Update word count from analysis if available
            if analysis.word_count > 0:
                word_count = analysis.word_count
        except Exception as e:
            # If analysis fails, default to fast pipeline
            recommended_pipeline = "fast"
            analysis_metadata = {"error": str(e)}

        # Create document record
        document = Document(
            id=doc_id,
            matter_id=matter_id,
            filename=file_path.name,
            filepath=str(dest_path),
            file_size_bytes=file_size,
            word_count=word_count,
            status="uploaded",
            recommended_pipeline=recommended_pipeline,
            analysis_metadata=analysis_metadata
        )
        db.add(document)

    db.commit()
    db.refresh(matter)

    return matter, len(files)


@router.get("/types/list")
async def list_matter_types(db: Session = Depends(get_db)):
    """Get list of distinct matter types."""
    results = db.query(Matter.matter_type).distinct().all()
    types = [r[0] for r in results if r[0]]
    return {"types": sorted(types)}


@router.get("/stats/by-type")
async def get_stats_by_type(db: Session = Depends(get_db)):
    """Get matter and document stats grouped by matter type."""
    from sqlalchemy import func
    from ..database.db import Result

    # Get all matters with document counts
    matters = db.query(Matter).all()

    # Group by type
    stats_by_type: dict = {}
    for matter in matters:
        matter_type = matter.matter_type or 'General'
        if matter_type not in stats_by_type:
            stats_by_type[matter_type] = {
                "matter_type": matter_type,
                "matter_count": 0,
                "document_count": 0,
                "pending_count": 0,
                "processing_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "total_confidence": 0,
                "confidence_count": 0
            }

        stats_by_type[matter_type]["matter_count"] += 1

        # Count documents by status for this matter
        docs = db.query(Document).filter(Document.matter_id == matter.id).all()
        for doc in docs:
            stats_by_type[matter_type]["document_count"] += 1
            if doc.status == "uploaded":
                stats_by_type[matter_type]["pending_count"] += 1
            elif doc.status == "processing":
                stats_by_type[matter_type]["processing_count"] += 1
            elif doc.status == "completed":
                stats_by_type[matter_type]["completed_count"] += 1
                # Get confidence for completed documents
                result = db.query(Result).filter(Result.document_id == doc.id).order_by(Result.processed_at.desc()).first()
                if result and result.average_confidence:
                    stats_by_type[matter_type]["total_confidence"] += result.average_confidence
                    stats_by_type[matter_type]["confidence_count"] += 1
            elif doc.status == "failed":
                stats_by_type[matter_type]["failed_count"] += 1

    # Calculate average confidence and clean up
    result_list = []
    for stats in stats_by_type.values():
        avg_confidence = (stats["total_confidence"] / stats["confidence_count"]) if stats["confidence_count"] > 0 else None
        result_list.append({
            "matter_type": stats["matter_type"],
            "matter_count": stats["matter_count"],
            "document_count": stats["document_count"],
            "pending_count": stats["pending_count"],
            "processing_count": stats["processing_count"],
            "completed_count": stats["completed_count"],
            "failed_count": stats["failed_count"],
            "average_confidence": round(avg_confidence, 3) if avg_confidence else None
        })

    return result_list


class BatchProcessRequest(BaseModel):
    matter_type: Optional[str] = None  # If None, process all types
    only_pending: bool = True  # If False, reprocess all including completed
    fast_mode: bool = True  # Use fast text-only tagger (recommended)
    smart_mode: bool = False  # Use LLM-based tagger (requires GOOGLE_API_KEY)
    auto_mode: bool = False  # Use recommended pipeline from document analysis


def run_fast_pipeline(document_id: str, filepath: str):
    """Run fast text-only tagging pipeline."""
    from ..database.db import SessionLocal, Document, Result, Model, ModelUsage
    from ..services.fast_tagger import process_document_fast
    import uuid

    db = SessionLocal()
    try:
        result_data = process_document_fast(filepath, db)

        # Get model name from result or default
        model_name = result_data.get('model', 'pile-of-law/legalbert-large-1.7M-2')
        processing_time = result_data.get('processing_time_seconds', 0)

        # Create result record
        result_id = str(uuid.uuid4())
        result = Result(
            id=result_id,
            document_id=document_id,
            processing_time_seconds=processing_time,
            semantic_model=model_name,
            vision_model=None,
            vision_enabled=False,
            tag_count=result_data.get('tag_count', 0),
            average_confidence=result_data.get('average_confidence', 0),
            result_json=result_data,
            visual_pages=None
        )
        db.add(result)

        # Get or create model in registry
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            model = Model(
                id=str(uuid.uuid4()),
                name=model_name,
                type='semantic',
                huggingface_url=f'https://huggingface.co/{model_name}',
                approved=True  # Auto-approve models used in processing
            )
            db.add(model)
            db.flush()  # Get model.id

        # Create model usage record
        model_usage = ModelUsage(
            id=str(uuid.uuid4()),
            model_id=model.id,
            result_id=result_id,
            processing_time_seconds=processing_time
        )
        db.add(model_usage)

        # Update document
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "completed"
            document.word_count = result_data.get('word_count')
            document.page_count = 1  # Text-based

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


def run_auto_pipeline(document_id: str, filepath: str, recommended_pipeline: str):
    """Run the recommended pipeline based on document analysis."""
    # Route to appropriate pipeline based on recommendation
    if recommended_pipeline == 'fast':
        run_fast_pipeline(document_id, filepath)
    elif recommended_pipeline == 'smart' or recommended_pipeline == 'zone':
        # Zone detection uses LLM, so route to smart pipeline
        run_smart_pipeline(document_id, filepath)
    elif recommended_pipeline == 'vision':
        # Vision pipeline - for now use smart as fallback
        # TODO: Implement dedicated vision pipeline
        run_smart_pipeline(document_id, filepath)
    elif recommended_pipeline == 'ocr':
        # OCR pipeline - for now use fast with OCR extraction
        # TODO: Implement dedicated OCR pipeline
        run_fast_pipeline(document_id, filepath)
    else:
        # Default to fast
        run_fast_pipeline(document_id, filepath)


def run_smart_pipeline(document_id: str, filepath: str):
    """Run LLM-based tagging pipeline (Smart Mode)."""
    from ..database.db import SessionLocal, Document, Result, Model, ModelUsage
    from ..services.llm_tagger import process_document_smart
    import uuid

    db = SessionLocal()
    try:
        result_data = process_document_smart(filepath, db)

        # Check for errors
        if result_data.get('error'):
            raise Exception(result_data['error'])

        # Get model name
        model_name = result_data.get('model', 'gemini-2.0-flash')
        processing_time = result_data.get('processing_time_seconds', 0)

        # Create result record
        result_id = str(uuid.uuid4())
        result = Result(
            id=result_id,
            document_id=document_id,
            processing_time_seconds=processing_time,
            semantic_model=model_name,
            vision_model=None,
            vision_enabled=False,
            tag_count=result_data.get('tag_count', 0),
            average_confidence=result_data.get('average_confidence', 0),
            result_json=result_data,
            visual_pages=None
        )
        db.add(result)

        # Get or create model in registry
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            model = Model(
                id=str(uuid.uuid4()),
                name=model_name,
                type='llm',
                huggingface_url=None,
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

        # Update document
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = "completed"
            document.word_count = result_data.get('word_count')
            document.page_count = 1

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


@router.post("/process-batch")
async def process_batch(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Queue batch processing of documents by matter type."""
    from ..routers.documents import run_pipeline

    # Build query for documents to process
    query = db.query(Document).join(Matter)

    if request.matter_type:
        query = query.filter(Matter.matter_type == request.matter_type)

    if request.only_pending:
        query = query.filter(Document.status == "uploaded")
    else:
        # Process all except currently processing
        query = query.filter(Document.status != "processing")

    documents = query.all()

    if not documents:
        return {
            "queued": 0,
            "message": "No documents to process"
        }

    # Collect document info before marking as processing (include recommended_pipeline for auto mode)
    doc_info = [(doc.id, doc.filepath, doc.recommended_pipeline or 'fast') for doc in documents]

    # Mark all as processing
    for doc in documents:
        doc.status = "processing"
    db.commit()

    # Choose pipeline based on mode
    if request.auto_mode:
        # Use recommended pipeline for each document
        pipeline_counts = {'fast': 0, 'zone': 0, 'vision': 0, 'ocr': 0, 'smart': 0}
        for doc_id, filepath, recommended in doc_info:
            background_tasks.add_task(run_auto_pipeline, doc_id, filepath, recommended)
            pipeline_counts[recommended] = pipeline_counts.get(recommended, 0) + 1
        mode_msg = f"auto ({', '.join(f'{k}:{v}' for k,v in pipeline_counts.items() if v > 0)})"
    elif request.smart_mode:
        # LLM-based pipeline (requires GOOGLE_API_KEY)
        for doc_id, filepath, _ in doc_info:
            background_tasks.add_task(run_smart_pipeline, doc_id, filepath)
        mode_msg = "smart LLM-based (Gemini)"
    elif request.fast_mode:
        # Fast text-only pipeline
        for doc_id, filepath, _ in doc_info:
            background_tasks.add_task(run_fast_pipeline, doc_id, filepath)
        mode_msg = "fast text-only"
    else:
        # Full pipeline with PDF conversion
        semantic_model = "pile-of-law/legalbert-large-1.7M-2"
        enable_vision = False
        vision_model = "microsoft/Florence-2-base"

        for doc_id, filepath, _ in doc_info:
            background_tasks.add_task(
                run_pipeline,
                doc_id,
                filepath,
                semantic_model,
                enable_vision,
                vision_model
            )
        mode_msg = "full pipeline"

    return {
        "queued": len(doc_info),
        "document_ids": [d[0] for d in doc_info],
        "mode": mode_msg,
        "message": f"Queued {len(doc_info)} documents for {mode_msg} processing"
    }
