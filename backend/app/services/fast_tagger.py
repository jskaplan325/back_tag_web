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
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer

# Singleton model instance
_model: Optional[SentenceTransformer] = None
_tag_embeddings: Dict[str, np.ndarray] = {}
_tag_metadata: Dict[str, Dict] = {}


def get_model(model_name: str = "pile-of-law/legalbert-large-1.7M-2") -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


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


def compute_tag_embeddings(tags: Dict[str, Dict], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all tag semantic examples."""
    embeddings = {}

    for tag_name, tag_def in tags.items():
        examples = tag_def.get('examples', [])
        if examples:
            emb = model.encode(examples, convert_to_numpy=True)
            embeddings[tag_name] = np.mean(emb, axis=0)  # Average embedding

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
    chunk_overlap: int = 50
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

    # Encode chunks
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

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
            matches.append({
                'name': tag_name,
                'tag_name': tag_name,  # Alias for compatibility
                'tag': tag_name,  # Another alias
                'area': tag_metadata.get(tag_name, {}).get('area', 'Unknown'),
                'confidence': round(hybrid_confidence, 3),
                'score': round(hybrid_confidence, 3),  # Alias
                'semantic_similarity': round(max_sim, 3),
                'pattern_matches': pattern_matches,
                'evidence_chunk': chunks[best_chunk_idx][:300]
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

    Returns a result dict compatible with the existing pipeline format.
    """
    global _tag_embeddings, _tag_metadata

    start_time = time.time()

    # Get model
    model = get_model(model_name)

    # Load taxonomy if not cached
    if not _tag_metadata:
        _tag_metadata = load_taxonomy_tags(db_session)
        _tag_embeddings = compute_tag_embeddings(_tag_metadata, model)

    # Extract text
    text = extract_text(filepath)
    word_count = len(text.split())

    # Tag text
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
        'average_confidence': sum(t['confidence'] for t in tags) / len(tags) if tags else 0,
        'method': 'fast_text_only',
        'model': model_name
    }


def clear_cache():
    """Clear cached model and embeddings (useful for testing or reloading taxonomy)."""
    global _model, _tag_embeddings, _tag_metadata
    _model = None
    _tag_embeddings = {}
    _tag_metadata = {}
