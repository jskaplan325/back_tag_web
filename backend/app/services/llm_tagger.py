"""
LLM-Based Document Tagger (Smart Mode)

Uses Gemini API to intelligently tag documents based on taxonomy.
More accurate than pattern matching but requires API calls.

Supports:
- Gemini (Google): Set GOOGLE_API_KEY env var
- Fallback to fast_tagger if API unavailable
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check for Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .fast_tagger import extract_text, get_model, get_taxonomy_embeddings


# Singleton client
_gemini_client = None


def get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client

    if _gemini_client is None:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        _gemini_client = genai.GenerativeModel('gemini-2.0-flash')

    return _gemini_client


def build_tagging_prompt(text: str, taxonomy: Dict[str, List[str]], max_text_chars: int = 15000) -> str:
    """Build prompt for LLM tagging."""

    # Truncate text if needed
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + "\n\n[... document truncated ...]"

    # Build taxonomy description
    taxonomy_desc = "Available tags by Area of Law:\n"
    for area, tags in taxonomy.items():
        taxonomy_desc += f"\n{area}:\n"
        for tag in tags:
            taxonomy_desc += f"  - {tag}\n"

    prompt = f"""Analyze this legal document and identify which tags apply.

{taxonomy_desc}

For each tag that applies to this document, provide:
1. The tag name (must match exactly from the list above)
2. A confidence score (0.0 to 1.0)
3. Brief evidence from the document

Return your response as a JSON object:
{{
    "document_type": "Brief description of document type",
    "tags": [
        {{"tag": "Tag Name", "confidence": 0.95, "evidence": "Quote or description from document"}},
        {{"tag": "Another Tag", "confidence": 0.85, "evidence": "Supporting evidence"}}
    ]
}}

Only include tags with confidence >= 0.5. Return ONLY the JSON, no other text.

Document:
{text}
"""
    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM JSON response."""
    import re

    # Extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        raise ValueError("No JSON found in response")

    return json.loads(json_match.group())


def process_document_smart(
    filepath: str,
    db=None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process document using LLM-based tagging.

    Args:
        filepath: Path to document
        db: Database session (for taxonomy lookup)
        api_key: Optional API key override

    Returns:
        Tagging results with tags, confidence, and metadata
    """
    start_time = time.time()

    # Set API key if provided
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key

    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")

    if not os.getenv('GOOGLE_API_KEY'):
        raise ValueError("GOOGLE_API_KEY not set")

    # Extract text
    text = extract_text(filepath)
    if not text or len(text.strip()) < 50:
        return {
            'tags': [],
            'tag_count': 0,
            'average_confidence': 0,
            'word_count': 0,
            'processing_time_seconds': time.time() - start_time,
            'model': 'gemini-2.0-flash',
            'mode': 'smart',
            'error': 'Insufficient text content'
        }

    word_count = len(text.split())

    # Get taxonomy from database
    taxonomy = {}
    if db:
        from ..database.db import AreaOfLaw, Tag
        areas = db.query(AreaOfLaw).all()
        for area in areas:
            tags = db.query(Tag).filter(Tag.area_of_law_id == area.id).all()
            taxonomy[area.name] = [t.name for t in tags]

    # Fallback taxonomy if DB not available
    if not taxonomy:
        taxonomy = {
            'Funds': ['Limited Partnership Agreement', 'Subscription Agreement',
                     'Investment Advisory Agreement', 'Investment Management Agreement'],
            'M&A': ['Merger Agreement', 'Purchase Agreement', 'Due Diligence'],
            'General': ['Contract Terms', 'Financial Statements', 'Risk Factors']
        }

    # Build prompt and call LLM
    prompt = build_tagging_prompt(text, taxonomy)

    try:
        client = get_gemini_client()
        response = client.generate_content(prompt)
        result = parse_llm_response(response.text)

        # Extract tags
        tags = []
        for tag_info in result.get('tags', []):
            tags.append({
                'tag': tag_info.get('tag', ''),
                'name': tag_info.get('tag', ''),
                'confidence': float(tag_info.get('confidence', 0.5)),
                'evidence': tag_info.get('evidence', ''),
                'pattern_matches': 0,  # LLM doesn't use patterns
                'semantic_similarity': float(tag_info.get('confidence', 0.5))
            })

        # Sort by confidence
        tags.sort(key=lambda x: x['confidence'], reverse=True)

        # Calculate average confidence
        avg_confidence = sum(t['confidence'] for t in tags) / len(tags) if tags else 0

        processing_time = time.time() - start_time

        return {
            'tags': tags,
            'tag_count': len(tags),
            'average_confidence': round(avg_confidence, 3),
            'word_count': word_count,
            'processing_time_seconds': round(processing_time, 2),
            'model': 'gemini-2.0-flash',
            'mode': 'smart',
            'document_type': result.get('document_type', 'Unknown'),
            'api_tokens_estimated': len(prompt) // 4 + len(response.text) // 4
        }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'tags': [],
            'tag_count': 0,
            'average_confidence': 0,
            'word_count': word_count,
            'processing_time_seconds': round(processing_time, 2),
            'model': 'gemini-2.0-flash',
            'mode': 'smart',
            'error': str(e)
        }


def is_available() -> bool:
    """Check if Smart Mode is available."""
    return GEMINI_AVAILABLE and bool(os.getenv('GOOGLE_API_KEY'))
