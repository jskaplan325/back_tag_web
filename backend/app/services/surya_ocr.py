"""
Surya OCR Service

High-accuracy OCR using Surya with GPU support (MPS/CUDA).
Returns text with line-level and word-level bounding boxes.

Surya is significantly more accurate than Tesseract and runs on:
- Mac with Apple Silicon (MPS)
- Windows/Linux with NVIDIA GPUs (CUDA)
- CPU (slower but works)
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy load Surya models
_det_model = None
_det_processor = None
_rec_model = None
_rec_processor = None

def check_surya_available():
    """Check if surya is available (called at runtime, not import time)."""
    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        return True
    except ImportError as e:
        logger.warning(f"Surya not available: {e}. Install with: pip install surya-ocr")
        return False

# Check at import time for backward compat, but also check at runtime
SURYA_AVAILABLE = check_surya_available()

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


@dataclass
class TextLine:
    """A line of text with its bounding box."""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    polygon: List[Tuple[float, float]]  # 4 corners
    page: int
    words: List[Dict[str, Any]]  # Word-level data if available


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    text: str  # Full extracted text
    lines: List[TextLine]  # Line-level results with bboxes
    page_count: int
    word_count: int
    avg_confidence: float
    device: str  # cuda, mps, or cpu


def get_device() -> str:
    """Detect the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _init_surya():
    """Initialize Surya models (lazy loading)."""
    global _det_model, _det_processor, _rec_model, _rec_processor

    if not SURYA_AVAILABLE:
        raise ImportError("Surya OCR not installed. Run: pip install surya-ocr")

    if _det_model is None:
        logger.info("Initializing Surya OCR models...")
        device = get_device()
        logger.info(f"Using device: {device}")

        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        _det_model = load_det_model()
        _det_processor = load_det_processor()
        _rec_model = load_rec_model()
        _rec_processor = load_rec_processor()

        logger.info("Surya OCR models loaded")

    return _det_model, _det_processor, _rec_model, _rec_processor


def extract_text_from_image(
    image: Image.Image,
    page_num: int = 0,
    langs: List[str] = None
) -> Tuple[str, List[TextLine]]:
    """
    Extract text from a PIL Image with bounding boxes.

    Args:
        image: PIL Image object
        page_num: Page number for multi-page documents
        langs: Languages to use for OCR (default: ["en"])

    Returns:
        Tuple of (full_text, list of TextLine objects)
    """
    if langs is None:
        langs = ["en"]

    det_model, det_processor, rec_model, rec_processor = _init_surya()

    from surya.ocr import run_ocr

    # Run OCR
    predictions = run_ocr(
        [image],
        [langs],
        det_model,
        det_processor,
        rec_model,
        rec_processor
    )

    if not predictions or len(predictions) == 0:
        return "", []

    result = predictions[0]
    lines = []
    all_text = []

    # Process each text line
    for line_data in result.text_lines:
        text = line_data.text
        confidence = getattr(line_data, 'confidence', 1.0)
        bbox = getattr(line_data, 'bbox', [0, 0, 0, 0])
        polygon = getattr(line_data, 'polygon', [])

        # Extract words if available
        words = []
        if hasattr(line_data, 'words') and line_data.words:
            for word in line_data.words:
                words.append({
                    'text': getattr(word, 'text', ''),
                    'confidence': getattr(word, 'confidence', 1.0),
                    'bbox': getattr(word, 'bbox', [0, 0, 0, 0]),
                })

        line = TextLine(
            text=text,
            confidence=confidence,
            bbox=tuple(bbox) if bbox else (0, 0, 0, 0),
            polygon=[(p[0], p[1]) for p in polygon] if polygon else [],
            page=page_num,
            words=words
        )
        lines.append(line)
        all_text.append(text)

    full_text = "\n".join(all_text)
    return full_text, lines


def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 200,
    langs: List[str] = None
) -> OCRResult:
    """
    Extract text from a PDF document with bounding boxes.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for PDF to image conversion
        langs: Languages to use for OCR (default: ["en"])

    Returns:
        OCRResult with text and line-level bounding boxes
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image not available. Install with: pip install pdf2image")

    if langs is None:
        langs = ["en"]

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Processing PDF with Surya OCR: {pdf_path.name}")

    # Convert PDF to images
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info(f"Converted {len(images)} pages to images (DPI: {dpi})")

    all_lines = []
    all_text = []
    total_confidence = 0
    confidence_count = 0

    # Process each page
    for page_num, image in enumerate(images):
        logger.debug(f"Processing page {page_num + 1}/{len(images)}")
        page_text, page_lines = extract_text_from_image(image, page_num, langs)

        all_text.append(page_text)
        all_lines.extend(page_lines)

        # Accumulate confidence
        for line in page_lines:
            total_confidence += line.confidence
            confidence_count += 1

    full_text = "\n\n".join(all_text)
    word_count = len(full_text.split())
    avg_confidence = total_confidence / max(confidence_count, 1)

    logger.info(f"Extracted {word_count} words with {avg_confidence:.2%} avg confidence")

    return OCRResult(
        text=full_text,
        lines=all_lines,
        page_count=len(images),
        word_count=word_count,
        avg_confidence=avg_confidence,
        device=get_device()
    )


def extract_text_from_image_file(
    image_path: str,
    langs: List[str] = None
) -> OCRResult:
    """
    Extract text from an image file with bounding boxes.

    Args:
        image_path: Path to image file (PNG, JPG, etc.)
        langs: Languages to use for OCR (default: ["en"])

    Returns:
        OCRResult with text and line-level bounding boxes
    """
    if langs is None:
        langs = ["en"]

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"Processing image with Surya OCR: {image_path.name}")

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    full_text, lines = extract_text_from_image(image, page_num=0, langs=langs)
    word_count = len(full_text.split())

    total_confidence = sum(line.confidence for line in lines)
    avg_confidence = total_confidence / max(len(lines), 1)

    return OCRResult(
        text=full_text,
        lines=lines,
        page_count=1,
        word_count=word_count,
        avg_confidence=avg_confidence,
        device=get_device()
    )


def ocr_result_to_dict(result: OCRResult) -> Dict[str, Any]:
    """Convert OCRResult to dictionary for JSON serialization."""
    return {
        'text': result.text,
        'lines': [asdict(line) for line in result.lines],
        'page_count': result.page_count,
        'word_count': result.word_count,
        'avg_confidence': result.avg_confidence,
        'device': result.device,
    }


# Compatibility layer with old Tesseract API
class SuryaOCRHandler:
    """
    Drop-in replacement for OCRHandler that uses Surya.

    Provides the same interface as the old Tesseract-based OCRHandler
    but with better accuracy and GPU acceleration.
    """

    def __init__(self, dpi: int = 200, langs: List[str] = None):
        """
        Initialize Surya OCR handler.

        Args:
            dpi: DPI for PDF to image conversion
            langs: Languages for OCR (default: ["en"])
        """
        if not SURYA_AVAILABLE:
            raise ImportError(
                "Surya OCR not available.\n"
                "Install with: pip install 'surya-ocr==0.4.15'\n"
                "Requires PyTorch"
            )
        self.dpi = dpi
        self.langs = langs or ["en"]
        self._last_result: Optional[OCRResult] = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF (compatible with old API).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as string
        """
        result = extract_text_from_pdf(pdf_path, dpi=self.dpi, langs=self.langs)
        self._last_result = result
        return result.text

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image file (compatible with old API).

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        result = extract_text_from_image_file(image_path, langs=self.langs)
        self._last_result = result
        return result.text

    def get_last_result(self) -> Optional[OCRResult]:
        """Get the full OCRResult from the last extraction."""
        return self._last_result

    def get_bounding_boxes(self) -> List[Dict[str, Any]]:
        """
        Get bounding boxes from the last extraction.

        Returns:
            List of dicts with text, bbox, confidence, page
        """
        if not self._last_result:
            return []

        return [
            {
                'text': line.text,
                'bbox': line.bbox,
                'confidence': line.confidence,
                'page': line.page,
                'words': line.words,
            }
            for line in self._last_result.lines
        ]


def test_surya():
    """Quick test of Surya OCR."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python surya_ocr.py <pdf_or_image_path>")
        print("\nThis will run Surya OCR and show results with bounding boxes.")
        return

    logging.basicConfig(level=logging.INFO)

    file_path = sys.argv[1]
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        result = extract_text_from_pdf(file_path)
    else:
        result = extract_text_from_image_file(file_path)

    print(f"\n{'='*60}")
    print(f"SURYA OCR RESULTS")
    print(f"{'='*60}")
    print(f"Device: {result.device}")
    print(f"Pages: {result.page_count}")
    print(f"Words: {result.word_count}")
    print(f"Avg Confidence: {result.avg_confidence:.2%}")
    print(f"Lines detected: {len(result.lines)}")
    print(f"\n{'='*60}")
    print("SAMPLE TEXT (first 1000 chars):")
    print(f"{'='*60}")
    print(result.text[:1000])

    if result.lines:
        print(f"\n{'='*60}")
        print("SAMPLE BOUNDING BOXES (first 5 lines):")
        print(f"{'='*60}")
        for line in result.lines[:5]:
            print(f"  [{line.confidence:.0%}] {line.text[:50]}...")
            print(f"       bbox: {line.bbox}")


if __name__ == "__main__":
    test_surya()
