"""
Fast Text-Only Document Tagger

Lightweight tagger that extracts text and runs hybrid pattern + semantic matching,
bypassing PDF-to-image conversion and vision models.

Performance: ~2-3 seconds per document vs 10+ seconds with full pipeline.
"""
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer


# Supported document types for legal document processing
SUPPORTED_EXTENSIONS = {
    '.pdf', '.txt', '.doc', '.docx', '.rtf',
    '.htm', '.html',  # May contain embedded documents
}

# Explicitly unsupported - skip without error
SKIP_EXTENSIONS = {
    '.css', '.js', '.json', '.xml', '.yaml', '.yml',
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg',
    '.zip', '.tar', '.gz', '.rar',
    '.exe', '.dll', '.so', '.bin',
    '.mp3', '.mp4', '.wav', '.avi',
    '.xls', '.xlsx', '.ppt', '.pptx',  # Could add support later
}


class DocumentQualityResult:
    """Result of document quality validation."""
    def __init__(self, is_valid: bool, status: str, reason: str = ""):
        self.is_valid = is_valid
        self.status = status  # 'processed', 'skipped', 'failed'
        self.reason = reason


def check_file_type(filepath: str) -> DocumentQualityResult:
    """Check if file type is supported for processing."""
    ext = Path(filepath).suffix.lower()

    if ext in SUPPORTED_EXTENSIONS:
        return DocumentQualityResult(True, 'processed')
    elif ext in SKIP_EXTENSIONS:
        return DocumentQualityResult(False, 'skipped', f'Unsupported file type: {ext}')
    else:
        # Unknown extension - try to process but may fail
        return DocumentQualityResult(True, 'processed', f'Unknown file type: {ext}')


def validate_text_quality(text: str, min_words: int = 30) -> DocumentQualityResult:
    """
    Validate extracted text quality to detect:
    - Binary/encoded content (base64, MIME)
    - Scrambled/corrupted text
    - Insufficient text content
    - Non-prose content (code, CSS, etc.)
    """
    if not text or not text.strip():
        return DocumentQualityResult(False, 'failed', 'No text content extracted')

    # Check word count
    words = text.split()
    word_count = len(words)

    if word_count < min_words:
        return DocumentQualityResult(False, 'failed', f'Insufficient text: {word_count} words (min: {min_words})')

    # Sample text for quality checks (first 2000 chars)
    sample = text[:2000]

    # Check for binary/encoded content
    # High ratio of non-printable or special characters indicates binary
    special_chars = sum(1 for c in sample if not c.isprintable() or c in '\\@#$%^&*+=|~`')
    special_ratio = special_chars / len(sample) if sample else 0

    if special_ratio > 0.15:
        return DocumentQualityResult(False, 'failed', f'Binary/encoded content detected ({special_ratio:.0%} special chars)')

    # Check for base64/MIME patterns
    base64_pattern = r'[A-Za-z0-9+/=]{50,}'
    if len(re.findall(base64_pattern, sample)) > 3:
        return DocumentQualityResult(False, 'failed', 'Base64/MIME encoded content detected')

    # Check for scrambled text (repeating escape sequences, hex patterns)
    scramble_patterns = [
        r'\\[A-Z][0-9]{2,}',  # \M4$ style
        r'[A-Z0-9]{2,}\\',    # Backslash heavy
        r'[\x00-\x1f]{3,}',   # Control characters
    ]
    scramble_matches = sum(len(re.findall(p, sample)) for p in scramble_patterns)
    if scramble_matches > 10:
        return DocumentQualityResult(False, 'failed', 'Scrambled/corrupted text detected')

    # Check for code/CSS content
    code_indicators = [
        r'\{\s*[a-z-]+\s*:\s*[^}]+\}',  # CSS rules
        r'function\s*\([^)]*\)\s*\{',    # JavaScript
        r'import\s+[\w.]+',              # Python/Java imports
        r'<\?(?:php|xml)',               # PHP/XML
    ]
    code_matches = sum(len(re.findall(p, sample, re.IGNORECASE)) for p in code_indicators)
    if code_matches > 5:
        return DocumentQualityResult(False, 'skipped', 'Code/markup content detected (not a legal document)')

    # Check for reasonable prose (sentences, punctuation)
    sentence_endings = len(re.findall(r'[.!?]\s', sample))
    if word_count > 100 and sentence_endings < 3:
        # Lots of words but no sentences - might be garbage
        return DocumentQualityResult(False, 'failed', 'No sentence structure detected')

    return DocumentQualityResult(True, 'processed')

# Singleton model instance
_model: Optional[SentenceTransformer] = None
_tag_embeddings: Dict[str, np.ndarray] = {}
_tag_metadata: Dict[str, Dict] = {}


def get_model(model_name: str = "pile-of-law/legalbert-large-1.7M-2", force_reload: bool = False) -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None or force_reload:
        _model = SentenceTransformer(model_name)
    return _model


def reset_model():
    """Reset model to force reload on next use."""
    global _model
    _model = None


def extract_text(filepath: str) -> str:
    """Extract text from file using lightweight methods."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif ext == '.pdf':
        # Try pdfplumber first (better for tables)
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return '\n\n'.join([p.extract_text() or '' for p in pdf.pages])
        except ImportError:
            pass

        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            return '\n\n'.join([p.extract_text() or '' for p in reader.pages])
        except ImportError:
            pass

        raise ImportError("Need pdfplumber or pypdf for PDF extraction")

    elif ext in ['.doc', '.docx']:
        try:
            from docx import Document
            doc = Document(filepath)
            return '\n'.join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ImportError("Need python-docx for DOCX extraction")

    else:
        # Try reading as text
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def load_taxonomy_tags(db_session) -> Dict[str, Dict]:
    """Load tag definitions from database taxonomy."""
    from ..database.db import AreaOfLaw, Tag

    tags = {}
    areas = db_session.query(AreaOfLaw).all()

    for area in areas:
        for tag in area.tags:
            if tag.semantic_examples:
                tags[tag.name] = {
                    'area': area.name,
                    'examples': tag.semantic_examples,
                    'patterns': tag.patterns or [],
                    'threshold': tag.threshold or 0.45
                }

    return tags


def compute_tag_embeddings(tags: Dict[str, Dict], model: SentenceTransformer, _retry: bool = False) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all tag semantic examples."""
    embeddings = {}

    try:
        for tag_name, tag_def in tags.items():
            examples = tag_def.get('examples', [])
            if examples:
                emb = model.encode(examples, convert_to_numpy=True)
                embeddings[tag_name] = np.mean(emb, axis=0)  # Average embedding
    except RuntimeError as e:
        if "meta tensor" in str(e) and not _retry:
            # Model has corrupted state - reload and retry once
            reset_model()
            new_model = get_model(force_reload=True)
            return compute_tag_embeddings(tags, new_model, _retry=True)
        raise

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_weighted_confidence(
    tags: List[Dict],
    min_threshold: float = 0.5
) -> float:
    """
    Compute weighted average confidence, filtering out low-confidence tags.

    - Filters out tags below min_threshold (these indicate taxonomy issues or wrong matches)
    - Applies weighted average: sum(c²) / sum(c) - higher scores count more

    Example: [0.9, 0.85, 0.6, 0.3] with threshold 0.5
      → filters to [0.9, 0.85, 0.6]
      → weighted: (0.81 + 0.72 + 0.36) / (0.9 + 0.85 + 0.6) = 0.804
    """
    if not tags:
        return 0.0

    # Filter to tags above threshold
    valid_scores = [t['confidence'] for t in tags if t['confidence'] >= min_threshold]

    if not valid_scores:
        # If nothing passes threshold, return simple average of all (indicates problem)
        return sum(t['confidence'] for t in tags) / len(tags)

    # Weighted average: sum(c²) / sum(c)
    sum_squared = sum(c * c for c in valid_scores)
    sum_scores = sum(valid_scores)

    return round(sum_squared / sum_scores, 3)


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count how many pattern matches are found in text."""
    if not patterns:
        return 0

    text_lower = text.lower()
    total_matches = 0

    for pattern in patterns:
        try:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            total_matches += len(matches)
        except re.error:
            # Invalid regex, try as literal
            total_matches += text_lower.count(pattern.lower())

    return total_matches


def find_pattern_matches(text: str, patterns: List[str], max_matches: int = 50) -> List[Dict[str, Any]]:
    """
    Find pattern matches with their positions in text.
    Returns list of {start, end, text, pattern} dicts for highlighting.
    """
    if not patterns:
        return []

    matches = []
    for pattern in patterns:
        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'pattern': pattern
                })
                if len(matches) >= max_matches:
                    break
        except re.error:
            # Invalid regex, try as literal
            pattern_lower = pattern.lower()
            text_lower = text.lower()
            start = 0
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break
                matches.append({
                    'start': pos,
                    'end': pos + len(pattern),
                    'text': text[pos:pos + len(pattern)],
                    'pattern': pattern
                })
                start = pos + 1
                if len(matches) >= max_matches:
                    break
        if len(matches) >= max_matches:
            break

    # Sort by position and remove overlaps
    matches.sort(key=lambda x: x['start'])
    return matches[:max_matches]


def compute_hybrid_confidence(
    semantic_score: float,
    pattern_matches: int,
    semantic_weight: float = 0.6,
    pattern_boost: float = 0.15,
    max_pattern_boost: float = 0.35
) -> float:
    """
    Compute hybrid confidence from semantic similarity and pattern matches.

    - Base: semantic_score * semantic_weight
    - Pattern boost: min(pattern_matches * pattern_boost, max_pattern_boost)
    - Final: normalized to 0-1 range

    This produces scores in the 0.7-0.95 range for good matches.
    """
    # Base semantic contribution (scaled up from typical 0.4-0.7 range)
    # Map 0.4-0.7 semantic to 0.5-0.8 base
    scaled_semantic = 0.5 + (semantic_score - 0.4) * (0.3 / 0.3)
    scaled_semantic = max(0.3, min(0.85, scaled_semantic))

    # Pattern boost (capped)
    pattern_contribution = min(pattern_matches * pattern_boost, max_pattern_boost)

    # Combined score
    final_score = scaled_semantic + pattern_contribution

    # Ensure in valid range
    return min(0.98, max(0.0, final_score))


def tag_text(
    text: str,
    model: SentenceTransformer,
    tag_embeddings: Dict[str, np.ndarray],
    tag_metadata: Dict[str, Dict],
    threshold: float = 0.45,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    _retry: bool = False
) -> List[Dict[str, Any]]:
    """Tag text using hybrid pattern + semantic similarity matching."""

    # Chunk text
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return []

    # Encode chunks with retry on meta tensor error
    try:
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True)
    except RuntimeError as e:
        if "meta tensor" in str(e) and not _retry:
            # Model has corrupted state - reload and retry once
            reset_model()
            new_model = get_model(force_reload=True)
            return tag_text(
                text, new_model, tag_embeddings, tag_metadata,
                threshold, chunk_size, chunk_overlap, _retry=True
            )
        raise

    # Match against tags
    matches = []
    for tag_name, tag_emb in tag_embeddings.items():
        # Find max similarity across chunks
        similarities = [cosine_similarity(chunk_emb, tag_emb) for chunk_emb in chunk_embeddings]
        max_sim = max(similarities)
        best_chunk_idx = similarities.index(max_sim)

        # Get pattern matches
        patterns = tag_metadata.get(tag_name, {}).get('patterns', [])
        pattern_matches = count_pattern_matches(text, patterns)

        # Compute hybrid confidence
        hybrid_confidence = compute_hybrid_confidence(max_sim, pattern_matches)

        # Use tag-specific threshold or default
        tag_threshold = tag_metadata.get(tag_name, {}).get('threshold', threshold)

        # Accept if semantic passes threshold OR we have strong pattern matches
        if max_sim >= tag_threshold or pattern_matches >= 3:
            # Find actual match positions for highlighting
            highlights = find_pattern_matches(text, patterns, max_matches=20)

            matches.append({
                'name': tag_name,
                'tag_name': tag_name,  # Alias for compatibility
                'tag': tag_name,  # Another alias
                'area': tag_metadata.get(tag_name, {}).get('area', 'Unknown'),
                'confidence': round(hybrid_confidence, 3),
                'score': round(hybrid_confidence, 3),  # Alias
                'semantic_similarity': round(max_sim, 3),
                'pattern_matches': pattern_matches,
                'evidence_chunk': chunks[best_chunk_idx][:300],
                'highlights': highlights  # Match positions for UI highlighting
            })

    return sorted(matches, key=lambda x: x['confidence'], reverse=True)


def process_document_fast(
    filepath: str,
    db_session,
    model_name: str = "pile-of-law/legalbert-large-1.7M-2",
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Process a document using fast text-only tagging.

    Returns a result dict with status field:
    - 'processed': Successfully tagged (confidence is meaningful)
    - 'skipped': Unsupported file type (not included in metrics)
    - 'failed': Extraction/quality issue (shown as error)
    """
    global _tag_embeddings, _tag_metadata

    start_time = time.time()

    # Step 1: Check file type
    file_check = check_file_type(filepath)
    if not file_check.is_valid:
        return {
            'tags': [],
            'tag_count': 0,
            'word_count': 0,
            'processing_time_seconds': time.time() - start_time,
            'average_confidence': 0,
            'method': 'fast_text_only',
            'model': model_name,
            'status': file_check.status,
            'status_reason': file_check.reason
        }

    # Step 2: Extract text
    try:
        text = extract_text(filepath)
    except Exception as e:
        return {
            'tags': [],
            'tag_count': 0,
            'word_count': 0,
            'processing_time_seconds': time.time() - start_time,
            'average_confidence': 0,
            'method': 'fast_text_only',
            'model': model_name,
            'status': 'failed',
            'status_reason': f'Text extraction failed: {str(e)[:100]}'
        }

    # Step 3: Validate text quality
    quality_check = validate_text_quality(text)
    if not quality_check.is_valid:
        return {
            'tags': [],
            'tag_count': 0,
            'word_count': len(text.split()) if text else 0,
            'processing_time_seconds': time.time() - start_time,
            'average_confidence': 0,
            'method': 'fast_text_only',
            'model': model_name,
            'status': quality_check.status,
            'status_reason': quality_check.reason
        }

    word_count = len(text.split())

    # Step 4: Get model and taxonomy
    model = get_model(model_name)

    if not _tag_metadata:
        _tag_metadata = load_taxonomy_tags(db_session)
        _tag_embeddings = compute_tag_embeddings(_tag_metadata, model)

    # Step 5: Tag text
    tags = tag_text(
        text,
        model,
        _tag_embeddings,
        _tag_metadata,
        threshold=threshold
    )

    processing_time = time.time() - start_time

    return {
        'tags': tags,
        'tag_count': len(tags),
        'word_count': word_count,
        'processing_time_seconds': processing_time,
        'average_confidence': compute_weighted_confidence(tags),
        'method': 'fast_text_only',
        'model': model_name,
        'status': 'processed',
        'status_reason': ''
    }


def get_taxonomy_embeddings(db_session, model_name: str = "pile-of-law/legalbert-large-1.7M-2"):
    """
    Get taxonomy tag embeddings and metadata.
    Used by llm_tagger for smart mode processing.
    """
    global _tag_embeddings, _tag_metadata

    model = get_model(model_name)

    if not _tag_metadata:
        _tag_metadata = load_taxonomy_tags(db_session)
        _tag_embeddings = compute_tag_embeddings(_tag_metadata, model)

    return _tag_embeddings, _tag_metadata


def clear_cache():
    """Clear cached model and embeddings (useful for testing or reloading taxonomy)."""
    global _model, _tag_embeddings, _tag_metadata
    _model = None
    _tag_embeddings = {}
    _tag_metadata = {}
