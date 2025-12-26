#!/usr/bin/env python3
"""
QA Test Script for Back Tag Web

Tests all major API endpoints and validates the system is working correctly.
Run with: python3.11 qa_test.py

Exit codes:
  0 = All tests passed
  1 = Some tests failed
"""
import sys
import json
import time
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000/api"
PASSED = 0
FAILED = 0
WARNINGS = 0

def test(name, condition, details=""):
    """Record test result."""
    global PASSED, FAILED
    if condition:
        print(f"  [PASS] {name}")
        PASSED += 1
        return True
    else:
        print(f"  [FAIL] {name}")
        if details:
            print(f"         {details}")
        FAILED += 1
        return False

def warn(name, details=""):
    """Record warning."""
    global WARNINGS
    print(f"  [WARN] {name}")
    if details:
        print(f"         {details}")
    WARNINGS += 1

def get(endpoint):
    """GET request helper."""
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
        return resp.status_code, resp.json() if resp.text else None
    except Exception as e:
        return 0, str(e)

def post(endpoint, data=None):
    """POST request helper."""
    try:
        resp = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        return resp.status_code, resp.json() if resp.text else None
    except Exception as e:
        return 0, str(e)

def delete(endpoint):
    """DELETE request helper."""
    try:
        resp = requests.delete(f"{BASE_URL}{endpoint}", timeout=30)
        return resp.status_code, resp.json() if resp.text else None
    except Exception as e:
        return 0, str(e)


def test_health():
    """Test basic API health."""
    print("\n=== API Health ===")

    # Test root endpoint
    try:
        resp = requests.get("http://localhost:8000/", timeout=5)
        test("API is running", resp.status_code == 200)
    except:
        test("API is running", False, "Cannot connect to localhost:8000")
        return False

    return True


def test_models():
    """Test models endpoints."""
    print("\n=== Models ===")

    status, data = get("/models")
    test("GET /models returns 200", status == 200)

    if status != 200:
        return

    test("Models list is not empty", len(data) > 0, f"Got {len(data)} models")

    # Check required models exist
    model_names = [m['name'] for m in data]
    test("LegalBERT registered", "pile-of-law/legalbert-large-1.7M-2" in model_names)
    test("Surya OCR registered", "surya-ocr" in model_names)
    test("Gemini registered", "gemini-2.0-flash" in model_names)

    # Check no Claude
    claude_models = [m for m in model_names if 'claude' in m.lower()]
    test("No Claude models", len(claude_models) == 0, f"Found: {claude_models}")

    # Check approved models
    approved = [m for m in data if m['approved']]
    test("At least 3 approved models", len(approved) >= 3, f"Got {len(approved)}")

    # Test model usage endpoint
    legalbert = next((m for m in data if 'legalbert' in m['name'].lower()), None)
    if legalbert:
        status, usage = get(f"/models/{legalbert['id']}/usage")
        test("GET /models/{id}/usage returns 200", status == 200)
        # Note: usage_count may be 0 if model was re-registered after processing
        if status == 200 and usage.get('total_usages', 0) > 0:
            test("LegalBERT has usage count", True, f"total_usages={usage.get('total_usages', 0)}")
        elif legalbert.get('usage_count', 0) > 0:
            test("LegalBERT has usage count", True, f"usage_count={legalbert.get('usage_count', 0)}")
        else:
            warn("LegalBERT usage count is 0", "May need to reprocess documents")


def test_taxonomy():
    """Test taxonomy endpoints."""
    print("\n=== Taxonomy ===")

    status, data = get("/taxonomy")
    test("GET /taxonomy returns 200", status == 200)

    if status != 200:
        return

    test("Taxonomy has areas", len(data) > 0, f"Got {len(data)} areas")

    # Check for expected areas
    area_names = [a['name'] for a in data]
    expected_areas = ['M&A / Corporate', 'Securities / Capital Markets', 'Investment Funds', 'Litigation']
    for area in expected_areas:
        if area in area_names:
            test(f"Area '{area}' exists", True)
        else:
            warn(f"Area '{area}' missing")

    # Count total tags
    total_tags = sum(len(a.get('tags', [])) for a in data)
    test("Taxonomy has tags", total_tags > 0, f"Got {total_tags} tags")
    test("At least 20 tags defined", total_tags >= 20, f"Got {total_tags}")

    # Test scoreboard
    status, scoreboard = get("/taxonomy/scoreboard")
    test("GET /taxonomy/scoreboard returns 200", status == 200)


def test_matters():
    """Test matters endpoints."""
    print("\n=== Matters ===")

    status, data = get("/matters")
    test("GET /matters returns 200", status == 200)

    if status != 200:
        return

    test("Matters list returned", isinstance(data, list))

    if len(data) > 0:
        matter = data[0]
        test("Matter has id", 'id' in matter)
        test("Matter has name", 'name' in matter)
        test("Matter has document_count", 'document_count' in matter)
        test("Matter has status counts", 'completed_count' in matter and 'failed_count' in matter)

        # Test single matter endpoint
        status, detail = get(f"/matters/{matter['id']}")
        test("GET /matters/{id} returns 200", status == 200)

        # Test matter tags
        status, tags = get(f"/matters/{matter['id']}/tags")
        test("GET /matters/{id}/tags returns 200", status == 200)
    else:
        warn("No matters in database to test")

    # Test stats by type
    status, stats = get("/matters/stats/by-type")
    test("GET /matters/stats/by-type returns 200", status == 200)


def test_documents():
    """Test documents endpoints."""
    print("\n=== Documents ===")

    status, data = get("/documents?limit=5")
    test("GET /documents returns 200", status == 200)

    if status != 200:
        return

    if len(data) > 0:
        doc = data[0]
        test("Document has id", 'id' in doc)
        test("Document has filename", 'filename' in doc)
        test("Document has status", 'status' in doc)

        # Test single document endpoint
        status, detail = get(f"/documents/{doc['id']}")
        test("GET /documents/{id} returns 200", status == 200)

        # Test document text endpoint
        status, text_data = get(f"/documents/{doc['id']}/text")
        test("GET /documents/{id}/text returns 200", status == 200)
    else:
        warn("No documents in database to test")


def test_metrics():
    """Test metrics endpoints."""
    print("\n=== Metrics ===")

    # Summary
    status, data = get("/metrics/summary")
    test("GET /metrics/summary returns 200", status == 200)

    if status == 200:
        test("Summary has total_documents", 'total_documents' in data)
        test("Summary has avg_confidence", 'avg_confidence' in data)
        test("Summary has models_registered", 'models_registered' in data)

        if data.get('total_processed', 0) > 0:
            test("Avg confidence is reasonable", 0.5 <= data.get('avg_confidence', 0) <= 1.0,
                 f"avg_confidence={data.get('avg_confidence')}")

    # Processing trends
    status, trends = get("/metrics/processing?days=14")
    test("GET /metrics/processing returns 200", status == 200)

    # Model usage
    status, models = get("/metrics/models")
    test("GET /metrics/models returns 200", status == 200)

    if status == 200 and len(models) > 0:
        # Check LegalBERT usage
        legalbert = next((m for m in models if 'legalbert' in m['model_name'].lower()), None)
        if legalbert:
            test("LegalBERT usage tracked in metrics", legalbert['usage_count'] > 0,
                 f"usage_count={legalbert['usage_count']}")

    # Matter types
    status, types = get("/metrics/matter-types")
    test("GET /metrics/matter-types returns 200", status == 200)


def test_surya_ocr():
    """Test Surya OCR availability."""
    print("\n=== Surya OCR ===")

    try:
        from app.services.surya_ocr import SURYA_AVAILABLE, get_device
        test("Surya OCR module imports", True)
        test("Surya is available", SURYA_AVAILABLE)

        if SURYA_AVAILABLE:
            device = get_device()
            test("Device detection works", device in ['mps', 'cuda', 'cpu'], f"device={device}")
            if device == 'mps':
                test("Using MPS (Apple Silicon)", True)
            elif device == 'cuda':
                test("Using CUDA (NVIDIA GPU)", True)
            else:
                warn("Using CPU (slower)", "Consider using GPU for better performance")
    except ImportError as e:
        test("Surya OCR module imports", False, str(e))


def test_fast_tagger():
    """Test fast tagger availability."""
    print("\n=== Fast Tagger ===")

    try:
        from app.services.fast_tagger import get_model, extract_text
        test("Fast tagger module imports", True)

        # Test model loading (may take a moment)
        print("    Loading LegalBERT model...")
        model = get_model()
        test("LegalBERT model loads", model is not None)
    except ImportError as e:
        test("Fast tagger module imports", False, str(e))
    except Exception as e:
        test("LegalBERT model loads", False, str(e))


def test_database_integrity():
    """Test database integrity."""
    print("\n=== Database Integrity ===")

    # Check documents have valid statuses
    status, docs = get("/documents?limit=100")
    if status == 200 and len(docs) > 0:
        valid_statuses = {'uploaded', 'processing', 'completed', 'failed'}
        invalid = [d for d in docs if d.get('status') not in valid_statuses]
        test("All documents have valid status", len(invalid) == 0,
             f"{len(invalid)} docs with invalid status")

        # Check for stuck processing
        processing = [d for d in docs if d.get('status') == 'processing']
        if len(processing) > 5:
            warn(f"{len(processing)} documents stuck in 'processing'",
                 "Consider resetting to 'uploaded'")
        else:
            test("No stuck processing documents", True)

    # Check models have types
    status, models = get("/models")
    if status == 200:
        valid_types = {'semantic', 'vision', 'llm', 'ocr'}
        invalid = [m for m in models if m.get('type') not in valid_types]
        test("All models have valid type", len(invalid) == 0,
             f"Invalid: {[m['name'] for m in invalid]}")


def main():
    """Run all QA tests."""
    print("=" * 60)
    print("BACK TAG WEB - QA TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start = time.time()

    # Run tests
    if not test_health():
        print("\n[FATAL] API not running. Start with: python3.11 -m uvicorn app.main:app --reload")
        sys.exit(1)

    test_models()
    test_taxonomy()
    test_matters()
    test_documents()
    test_metrics()
    test_database_integrity()

    # These tests require being in the backend directory
    import os
    os.chdir("/Users/jaredkaplan/Projects/back_tag_web/backend")
    sys.path.insert(0, ".")

    test_surya_ocr()
    test_fast_tagger()

    elapsed = time.time() - start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed:   {PASSED}")
    print(f"  Failed:   {FAILED}")
    print(f"  Warnings: {WARNINGS}")
    print(f"  Time:     {elapsed:.1f}s")
    print("=" * 60)

    if FAILED > 0:
        print("\n[RESULT] SOME TESTS FAILED")
        sys.exit(1)
    elif WARNINGS > 0:
        print("\n[RESULT] ALL TESTS PASSED (with warnings)")
        sys.exit(0)
    else:
        print("\n[RESULT] ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
