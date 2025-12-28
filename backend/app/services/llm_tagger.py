"""
LLM-Based Document Tagger (Smart Mode)

Uses LLM to intelligently tag documents based on taxonomy.
More accurate than pattern matching but requires LLM inference.

Supports:
- Ollama (Local): Set OLLAMA_MODEL env var (default: qwen2.5:7b)
- Gemini (Google): Set GOOGLE_API_KEY env var
- Auto-selects Ollama if available, falls back to Gemini
"""
import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check for Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .fast_tagger import extract_text, get_model, get_taxonomy_embeddings


# Singleton clients
_gemini_client = None
_ollama_available = None

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')


def check_ollama_available() -> bool:
    """Check if Ollama is running and has a model available."""
    global _ollama_available

    if _ollama_available is not None:
        return _ollama_available

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            # Check if our preferred model or any model is available
            _ollama_available = any(OLLAMA_MODEL.split(':')[0] in name for name in model_names) or len(models) > 0
            return _ollama_available
    except Exception:
        pass

    _ollama_available = False
    return False


def call_ollama(prompt: str, model: str = None) -> str:
    """Call Ollama API for text generation."""
    model = model or OLLAMA_MODEL

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for consistent JSON
                "num_predict": 2000
            }
        },
        timeout=120  # 2 min timeout for longer docs
    )

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")

    return response.json().get('response', '')


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


def build_tagging_prompt(text: str, taxonomy: Dict[str, List[str]], max_text_chars: int = 8000) -> str:
    """Build prompt for LLM tagging."""

    # Truncate text if needed (smaller for local models)
    if len(text) > max_text_chars:
        # Take beginning and end for context
        half = max_text_chars // 2
        text = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]

    # Build taxonomy as compact list
    all_tags = []
    for area, tags in taxonomy.items():
        for tag in tags:
            all_tags.append(f"{tag} ({area})")

    taxonomy_str = ", ".join(all_tags)

    prompt = f"""You are a legal document classifier. Analyze the document and identify matching tags.

AVAILABLE TAGS: {taxonomy_str}

INSTRUCTIONS:
1. Read the document carefully
2. Select tags that match the document content
3. Return ONLY valid JSON, no other text

OUTPUT FORMAT (return this exact structure):
{{"document_type": "type here", "tags": [{{"tag": "exact tag name", "confidence": 0.9, "evidence": "brief quote"}}]}}

DOCUMENT:
{text}

JSON RESPONSE:"""
    return prompt


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM JSON response with robust error handling."""
    import re

    # Clean up common issues
    text = response_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'^```json?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Try to find JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        # Try to construct a minimal valid response
        return {"document_type": "Unknown", "tags": []}

    json_str = json_match.group()

    # Fix common JSON issues
    # Replace single quotes with double quotes (but not in strings)
    json_str = re.sub(r"(?<![\"\\])'([^']*)'(?![\"\\])", r'"\1"', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix truncated JSON
        if json_str.count('{') > json_str.count('}'):
            json_str += '}' * (json_str.count('{') - json_str.count('}'))
        if json_str.count('[') > json_str.count(']'):
            json_str += ']' * (json_str.count('[') - json_str.count(']'))

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"document_type": "Unknown", "tags": []}


def process_document_smart(
    filepath: str,
    db=None,
    api_key: Optional[str] = None,
    force_ollama: bool = False,
    force_gemini: bool = False
) -> Dict[str, Any]:
    """
    Process document using LLM-based tagging.

    Args:
        filepath: Path to document
        db: Database session (for taxonomy lookup)
        api_key: Optional API key override for Gemini
        force_ollama: Force use of Ollama (local)
        force_gemini: Force use of Gemini (cloud)

    Returns:
        Tagging results with tags, confidence, and metadata
    """
    start_time = time.time()

    # Determine which backend to use
    use_ollama = False
    model_name = 'unknown'

    if force_ollama:
        if not check_ollama_available():
            raise ValueError("Ollama not available - ensure it's running with a model loaded")
        use_ollama = True
        model_name = OLLAMA_MODEL
    elif force_gemini:
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
        if not GEMINI_AVAILABLE or not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("Gemini not available - install google-generativeai and set GOOGLE_API_KEY")
        use_ollama = False
        model_name = 'gemini-2.0-flash'
    else:
        # Auto-select: prefer Ollama (local) if available
        if check_ollama_available():
            use_ollama = True
            model_name = OLLAMA_MODEL
        elif GEMINI_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
            use_ollama = False
            model_name = 'gemini-2.0-flash'
        else:
            raise ValueError("No LLM backend available. Start Ollama or set GOOGLE_API_KEY")

    # Extract text
    text = extract_text(filepath)
    if not text or len(text.strip()) < 50:
        return {
            'tags': [],
            'tag_count': 0,
            'average_confidence': 0,
            'word_count': 0,
            'processing_time_seconds': time.time() - start_time,
            'model': model_name,
            'mode': 'smart',
            'backend': 'ollama' if use_ollama else 'gemini',
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
        if use_ollama:
            response_text = call_ollama(prompt)
        else:
            client = get_gemini_client()
            response = client.generate_content(prompt)
            response_text = response.text

        result = parse_llm_response(response_text)

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
            'model': model_name,
            'mode': 'smart',
            'backend': 'ollama' if use_ollama else 'gemini',
            'document_type': result.get('document_type', 'Unknown'),
            'tokens_estimated': len(prompt) // 4 + len(response_text) // 4
        }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'tags': [],
            'tag_count': 0,
            'average_confidence': 0,
            'word_count': word_count,
            'processing_time_seconds': round(processing_time, 2),
            'model': model_name,
            'mode': 'smart',
            'backend': 'ollama' if use_ollama else 'gemini',
            'error': str(e)
        }


def is_available() -> bool:
    """Check if Smart Mode is available (either Ollama or Gemini)."""
    return check_ollama_available() or (GEMINI_AVAILABLE and bool(os.getenv('GOOGLE_API_KEY')))


def get_available_backend() -> Optional[str]:
    """Get which backend is available."""
    if check_ollama_available():
        return f'ollama ({OLLAMA_MODEL})'
    elif GEMINI_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
        return 'gemini (gemini-2.0-flash)'
    return None


# =============================================================================
# STAGE 2: LLM REFINEMENT FOR BORDERLINE TAGS
# =============================================================================

# Confidence thresholds for LLM trigger
LLM_TRIGGER_MIN = 0.70  # Minimum confidence to trigger LLM review
LLM_TRIGGER_MAX = 0.75  # Maximum confidence (above this, skip LLM)


def should_trigger_llm_refinement(avg_confidence: float, tags: List[Dict]) -> bool:
    """
    Determine if Stage 2 LLM refinement should be triggered.

    Criteria:
    - Average confidence is between 0.70 and 0.75 (borderline)
    """
    return LLM_TRIGGER_MIN <= avg_confidence < LLM_TRIGGER_MAX


def build_refinement_prompt(text_snippet: str, tags: List[Dict]) -> str:
    """
    Build the prompt for LLM tag refinement (Stage 2).

    Args:
        text_snippet: First ~3000 chars of document text
        tags: List of tags from E5 with confidence scores
    """
    # Format tags for the prompt
    tag_list = []
    for t in tags:
        name = t.get('name') or t.get('tag_name') or t.get('tag', '')
        conf = t.get('confidence', 0)
        area = t.get('area', 'Unknown')
        tag_list.append(f"- {name} (confidence: {conf:.0%}, area: {area})")

    tags_str = "\n".join(tag_list)

    prompt = f"""You are a legal document classifier. Review the suggested tags for this document and determine which are correct.

DOCUMENT EXCERPT:
{text_snippet[:3000]}

SUGGESTED TAGS (from embedding similarity):
{tags_str}

TASK:
For each tag, respond with:
1. CONFIRM if the tag is appropriate for this document
2. REJECT if the tag does not apply
3. Brief reasoning (1 sentence)

Focus especially on tags with confidence between 70-75% as these are borderline.

Respond in JSON format ONLY:
{{"refinements": [{{"tag": "Tag Name", "action": "CONFIRM", "reason": "Brief explanation"}}], "summary": "Document type summary"}}"""

    return prompt


def parse_refinement_response(response_text: str, original_tags: List[Dict]) -> List[Dict]:
    """
    Parse LLM refinement response and apply to tags.

    Args:
        response_text: Raw LLM response
        original_tags: Original tags from E5

    Returns:
        Refined tags with LLM feedback
    """
    import re

    refined_tags = []

    try:
        # Clean up response
        text = response_text.strip()
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # Find JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            refinements = {r["tag"]: r for r in data.get("refinements", [])}

            for tag in original_tags:
                tag_copy = tag.copy()
                tag_name = tag.get("name") or tag.get("tag_name") or tag.get("tag", "")

                if tag_name in refinements:
                    ref = refinements[tag_name]
                    action = ref.get("action", "").upper()
                    reason = ref.get("reason", "")

                    tag_copy["llm_action"] = action
                    tag_copy["llm_reason"] = reason

                    if action == "REJECT":
                        tag_copy["llm_rejected"] = True
                        # Lower confidence for rejected tags
                        tag_copy["confidence"] = min(tag_copy.get("confidence", 0.5), 0.5)
                    elif action == "CONFIRM":
                        tag_copy["llm_confirmed"] = True
                        # Boost confidence for confirmed tags
                        tag_copy["confidence"] = min(tag_copy.get("confidence", 0.7) + 0.1, 0.99)

                refined_tags.append(tag_copy)
        else:
            # No JSON found, return originals
            refined_tags = [t.copy() for t in original_tags]

    except (json.JSONDecodeError, KeyError) as e:
        # Return originals with error note
        for tag in original_tags:
            tag_copy = tag.copy()
            tag_copy["llm_error"] = f"Parse error: {str(e)}"
            refined_tags.append(tag_copy)

    return refined_tags


def refine_tags_with_llm(
    text: str,
    tags: List[Dict],
    model: str = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Stage 2: Refine E5 tags using LLM for borderline confidence cases.

    Args:
        text: Full document text
        tags: Tags from E5 Stage 1
        model: Ollama model to use (default: qwen2.5:7b)
        force: Force LLM even if confidence is high

    Returns:
        Dict with refined tags and metadata
    """
    model = model or OLLAMA_MODEL
    start_time = time.time()

    # Calculate average confidence
    confidences = [t.get("confidence", 0) for t in tags]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    result = {
        "tags": tags,
        "llm_used": False,
        "llm_model": None,
        "llm_processing_time": 0,
        "llm_error": None,
        "avg_confidence_before": round(avg_confidence, 3),
        "trigger_reason": None
    }

    # Check if we should trigger LLM
    if not force and not should_trigger_llm_refinement(avg_confidence, tags):
        if avg_confidence >= LLM_TRIGGER_MAX:
            result["trigger_reason"] = f"Skipped: confidence {avg_confidence:.0%} >= {LLM_TRIGGER_MAX:.0%}"
        else:
            result["trigger_reason"] = f"Skipped: confidence {avg_confidence:.0%} < {LLM_TRIGGER_MIN:.0%}"
        return result

    result["trigger_reason"] = f"Triggered: confidence {avg_confidence:.0%} in range {LLM_TRIGGER_MIN:.0%}-{LLM_TRIGGER_MAX:.0%}"

    # Check if Ollama is available
    if not check_ollama_available():
        result["llm_error"] = "Ollama not available"
        return result

    # Build prompt and call LLM
    prompt = build_refinement_prompt(text, tags)

    try:
        response_text = call_ollama(prompt, model)

        result["tags"] = parse_refinement_response(response_text, tags)
        result["llm_used"] = True
        result["llm_model"] = model
        result["llm_raw_response"] = response_text

        # Calculate new average confidence
        new_confidences = [t.get("confidence", 0) for t in result["tags"]]
        result["avg_confidence_after"] = round(sum(new_confidences) / len(new_confidences), 3) if new_confidences else 0

    except Exception as e:
        result["llm_error"] = str(e)

    result["llm_processing_time"] = round(time.time() - start_time, 2)

    return result


def get_llm_status() -> Dict[str, Any]:
    """Get status of LLM service for health checks and UI display."""
    status = {
        "available": False,
        "model": OLLAMA_MODEL,
        "url": OLLAMA_BASE_URL,
        "trigger_range": f"{LLM_TRIGGER_MIN:.0%} - {LLM_TRIGGER_MAX:.0%}",
        "stage2_enabled": True
    }

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            status["available"] = True
            status["installed_models"] = [m.get("name") for m in models]
            status["qwen_ready"] = check_ollama_available()
    except Exception as e:
        status["error"] = str(e)

    return status
