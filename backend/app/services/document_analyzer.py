"""
Document Analyzer Service

Analyzes documents on import to recommend the best processing pipeline.
Detects file type, page count, text density, visual content, and structure.

Pipeline recommendations:
- fast: Text-heavy simple documents (agreements, contracts)
- zone: Long structured documents (SEC filings, 50+ pages)
- vision: Documents with charts, tables, images
- ocr: Scanned documents with low/no extractable text
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# PDF analysis
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    import fitz  # PyMuPDF for image detection
    IMAGE_ANALYSIS_AVAILABLE = True
except ImportError:
    IMAGE_ANALYSIS_AVAILABLE = False


@dataclass
class DocumentAnalysis:
    """Results of document analysis."""
    file_type: str  # pdf, txt, docx, image
    file_size_bytes: int
    page_count: int
    word_count: int
    char_count: int

    # Text analysis
    text_density: float  # chars per page (0 = no text)
    has_extractable_text: bool

    # Structure analysis
    has_sections: bool  # Detected section headers
    section_count: int
    is_sec_filing: bool  # Detected SEC filing patterns

    # Visual analysis
    has_images: bool
    image_count: int
    has_tables: bool
    table_count: int
    sparse_pages: int  # Pages with low text (potential charts/images)

    # Recommendation
    recommended_pipeline: str  # fast, zone, vision, ocr
    recommendation_reason: str
    confidence: float  # 0-1


def analyze_text_content(text: str) -> Dict[str, Any]:
    """Analyze text content for structure and patterns."""
    results = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'has_sections': False,
        'section_count': 0,
        'is_sec_filing': False,
    }

    # Detect section headers (common patterns)
    section_patterns = [
        r'^#+\s+.+',  # Markdown headers
        r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS headers
        r'^ITEM\s+\d+[A-Z]?\.?\s+',  # SEC filing items
        r'^ARTICLE\s+[IVX\d]+',  # Legal articles
        r'^SECTION\s+\d+',  # Section numbers
        r'^\d+\.\s+[A-Z]',  # Numbered sections
    ]

    lines = text.split('\n')
    section_matches = 0
    for line in lines:
        line = line.strip()
        for pattern in section_patterns:
            if re.match(pattern, line, re.MULTILINE):
                section_matches += 1
                break

    results['section_count'] = section_matches
    results['has_sections'] = section_matches >= 3

    # Detect SEC filing patterns
    sec_patterns = [
        r'ITEM\s+1A\.?\s*RISK\s+FACTORS',
        r'ITEM\s+7\.?\s*MANAGEMENT',
        r'FORM\s+10-[KQ]',
        r'SECURITIES AND EXCHANGE COMMISSION',
        r'ANNUAL REPORT',
        r'QUARTERLY REPORT',
    ]

    for pattern in sec_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            results['is_sec_filing'] = True
            break

    return results


def analyze_pdf(filepath: str) -> DocumentAnalysis:
    """Analyze a PDF document."""
    file_size = os.path.getsize(filepath)

    page_count = 0
    total_text = ""
    image_count = 0
    table_count = 0
    sparse_pages = 0
    pages_text_lengths = []

    # Try pdfplumber first (better for tables)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(filepath) as pdf:
                page_count = len(pdf.pages)

                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    total_text += page_text + "\n"
                    pages_text_lengths.append(len(page_text))

                    # Detect tables
                    tables = page.extract_tables()
                    if tables:
                        table_count += len(tables)

                    # Sparse page detection (low text, might have images/charts)
                    if len(page_text) < 200:
                        sparse_pages += 1
        except Exception as e:
            print(f"pdfplumber failed: {e}")

    # Fallback to pypdf
    if page_count == 0 and PYPDF_AVAILABLE:
        try:
            reader = PdfReader(filepath)
            page_count = len(reader.pages)

            for page in reader.pages:
                page_text = page.extract_text() or ""
                total_text += page_text + "\n"
                pages_text_lengths.append(len(page_text))

                if len(page_text) < 200:
                    sparse_pages += 1
        except Exception as e:
            print(f"pypdf failed: {e}")

    # Try to detect images using PyMuPDF
    if IMAGE_ANALYSIS_AVAILABLE:
        try:
            doc = fitz.open(filepath)
            for page in doc:
                images = page.get_images()
                image_count += len(images)
            doc.close()
        except Exception:
            pass

    # Calculate metrics
    word_count = len(total_text.split())
    char_count = len(total_text)
    text_density = char_count / max(page_count, 1)
    has_extractable_text = char_count > 100

    # Analyze text structure
    text_analysis = analyze_text_content(total_text)

    # Determine recommendation
    pipeline, reason, confidence = recommend_pipeline(
        file_type='pdf',
        page_count=page_count,
        word_count=word_count,
        text_density=text_density,
        has_extractable_text=has_extractable_text,
        has_sections=text_analysis['has_sections'],
        is_sec_filing=text_analysis['is_sec_filing'],
        image_count=image_count,
        table_count=table_count,
        sparse_pages=sparse_pages
    )

    return DocumentAnalysis(
        file_type='pdf',
        file_size_bytes=file_size,
        page_count=page_count,
        word_count=word_count,
        char_count=char_count,
        text_density=text_density,
        has_extractable_text=has_extractable_text,
        has_sections=text_analysis['has_sections'],
        section_count=text_analysis['section_count'],
        is_sec_filing=text_analysis['is_sec_filing'],
        has_images=image_count > 0,
        image_count=image_count,
        has_tables=table_count > 0,
        table_count=table_count,
        sparse_pages=sparse_pages,
        recommended_pipeline=pipeline,
        recommendation_reason=reason,
        confidence=confidence
    )


def analyze_text_file(filepath: str) -> DocumentAnalysis:
    """Analyze a text file."""
    file_size = os.path.getsize(filepath)

    with open(filepath, 'r', errors='ignore') as f:
        text = f.read()

    word_count = len(text.split())
    char_count = len(text)

    # Estimate pages (assuming ~3000 chars per page)
    estimated_pages = max(1, char_count // 3000)

    # Analyze structure
    text_analysis = analyze_text_content(text)

    # Text files are always fast pipeline unless very long with sections
    if estimated_pages > 50 and text_analysis['has_sections']:
        pipeline = 'zone'
        reason = 'Long structured text document'
        confidence = 0.7
    else:
        pipeline = 'fast'
        reason = 'Text file with extractable content'
        confidence = 0.95

    return DocumentAnalysis(
        file_type='txt',
        file_size_bytes=file_size,
        page_count=estimated_pages,
        word_count=word_count,
        char_count=char_count,
        text_density=char_count / estimated_pages,
        has_extractable_text=True,
        has_sections=text_analysis['has_sections'],
        section_count=text_analysis['section_count'],
        is_sec_filing=text_analysis['is_sec_filing'],
        has_images=False,
        image_count=0,
        has_tables=False,
        table_count=0,
        sparse_pages=0,
        recommended_pipeline=pipeline,
        recommendation_reason=reason,
        confidence=confidence
    )


def recommend_pipeline(
    file_type: str,
    page_count: int,
    word_count: int,
    text_density: float,
    has_extractable_text: bool,
    has_sections: bool,
    is_sec_filing: bool,
    image_count: int,
    table_count: int,
    sparse_pages: int
) -> Tuple[str, str, float]:
    """
    Recommend the best pipeline based on document characteristics.

    Returns: (pipeline, reason, confidence)
    """

    # Priority 1: No extractable text = needs OCR
    if not has_extractable_text or text_density < 50:
        return ('ocr', 'Scanned document or image-based PDF requiring OCR', 0.9)

    # Priority 2: Heavy visual content
    visual_ratio = (image_count + sparse_pages) / max(page_count, 1)
    if visual_ratio > 0.3 or image_count > 10:
        return ('vision', f'Document has significant visual content ({image_count} images, {sparse_pages} sparse pages)', 0.85)

    # Priority 3: SEC filings or long structured documents
    if is_sec_filing:
        return ('zone', 'SEC filing detected - zone detection recommended for section extraction', 0.9)

    if page_count > 50 and has_sections:
        return ('zone', f'Long document ({page_count} pages) with {has_sections} sections - zone detection recommended', 0.8)

    # Priority 4: Documents with tables might benefit from vision
    if table_count > 5:
        return ('vision', f'Document has {table_count} tables - vision analysis recommended', 0.75)

    # Default: Fast pipeline for text-heavy documents
    if word_count > 500:
        return ('fast', 'Text-heavy document suitable for fast semantic tagging', 0.9)

    return ('fast', 'Standard document - fast pipeline recommended', 0.85)


def analyze_document(filepath: str) -> DocumentAnalysis:
    """
    Analyze a document and recommend the best processing pipeline.

    Args:
        filepath: Path to the document

    Returns:
        DocumentAnalysis with recommendation
    """
    filepath = str(filepath)
    ext = Path(filepath).suffix.lower()

    if ext == '.pdf':
        return analyze_pdf(filepath)
    elif ext in ['.txt', '.text']:
        return analyze_text_file(filepath)
    elif ext in ['.doc', '.docx']:
        # For now, treat as text-like
        # TODO: Add python-docx support
        return DocumentAnalysis(
            file_type='docx',
            file_size_bytes=os.path.getsize(filepath),
            page_count=1,
            word_count=0,
            char_count=0,
            text_density=0,
            has_extractable_text=True,
            has_sections=False,
            section_count=0,
            is_sec_filing=False,
            has_images=False,
            image_count=0,
            has_tables=False,
            table_count=0,
            sparse_pages=0,
            recommended_pipeline='fast',
            recommendation_reason='Word document - defaulting to fast pipeline',
            confidence=0.7
        )
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return DocumentAnalysis(
            file_type='image',
            file_size_bytes=os.path.getsize(filepath),
            page_count=1,
            word_count=0,
            char_count=0,
            text_density=0,
            has_extractable_text=False,
            has_sections=False,
            section_count=0,
            is_sec_filing=False,
            has_images=True,
            image_count=1,
            has_tables=False,
            table_count=0,
            sparse_pages=1,
            recommended_pipeline='ocr',
            recommendation_reason='Image file requires OCR for text extraction',
            confidence=0.95
        )
    else:
        # Unknown type
        return DocumentAnalysis(
            file_type=ext.lstrip('.'),
            file_size_bytes=os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            page_count=1,
            word_count=0,
            char_count=0,
            text_density=0,
            has_extractable_text=False,
            has_sections=False,
            section_count=0,
            is_sec_filing=False,
            has_images=False,
            image_count=0,
            has_tables=False,
            table_count=0,
            sparse_pages=0,
            recommended_pipeline='fast',
            recommendation_reason='Unknown file type - defaulting to fast pipeline',
            confidence=0.5
        )


def analysis_to_dict(analysis: DocumentAnalysis) -> Dict[str, Any]:
    """Convert DocumentAnalysis to dictionary for JSON storage."""
    return asdict(analysis)
