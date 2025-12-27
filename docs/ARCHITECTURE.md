# Back Tag v0.3.0 - Document Intelligence Pipeline

## Overview

A two-stage semantic document tagging system for legal document classification using AI embeddings and pattern matching.

---

## Pipeline Architecture

```
Document → Quick Scan → Process
                  ↓
             ┌─────────────────────────────────────┐
             │ Detect:                             │
             │ • File type (PDF/TXT/DOCX/etc)      │
             │ • Page count                        │
             │ • Text density (words per page)     │
             │ • Scanned detection (< 50 words/pg) │
             └─────────────────────────────────────┘
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
    Text Extractable     Scanned/Image-Only
    (most documents)     (< 50 words/page)
        ↓                   ↓
    Fast Pipeline       Flagged: needs_ocr
    (~2-5s)             (review queue)
        │                   │
        ▼                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    MODEL REGISTRY                       │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ Semantic (Embeddings) - ACTIVE                  │   │
  │  │ • intfloat/e5-large-v2 [HF] ← current default  │   │
  │  │ • pile-of-law/legalbert-large-1.7M-2 [HF]      │   │
  │  │ • BAAI/bge-large-en-v1.5 [HF]                  │   │
  │  └─────────────────────────────────────────────────┘   │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ Text Extraction - ACTIVE                        │   │
  │  │ • pdfplumber [Python] ← fast, reliable         │   │
  │  │ • python-docx [Python]                         │   │
  │  └─────────────────────────────────────────────────┘   │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ OCR (Future/Manual) - AVAILABLE                 │   │
  │  │ • datalab-to/surya [GitHub] (~2 min/page CPU)  │   │
  │  │   └─ GPU: ~5-10s/page (CUDA/MPS)               │   │
  │  └─────────────────────────────────────────────────┘   │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ LLM Backend (Future Smart Mode) - PLANNED       │   │
  │  │ • Ollama/qwen2.5:7b [Local] ← preferred        │   │
  │  │ • Ollama/llama3.1:8b [Local]                   │   │
  │  │ • Ollama/mistral:7b [Local]                    │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  Status: E5 Active • LLM Planned • OCR On-Demand       │
  └─────────────────────────────────────────────────────────┘
```

---

## Two-Stage Processing

### Stage 1: Fast Mode (E5 Embeddings) - ACTIVE

```
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Fast Mode (E5 Embeddings + Cosine Similarity) │
│ ┌─────────┐    ┌─────────────┐    ┌─────────────────┐  │
│ │ Extract │ →  │ E5-large-v2 │ →  │ Pattern Match + │  │
│ │ Text    │    │ Embeddings  │    │ Cosine Similarity│  │
│ │(pdfplumber)  │ (1024-dim)  │    │ (threshold 0.65)│  │
│ └─────────┘    └─────────────┘    └─────────────────┘  │
│                                                         │
│ • Time: ~2-5 sec per document                          │
│ • Local only, no API calls                             │
│ • Returns: tags[], highlights[], confidence scores     │
└─────────────────────────────────────────────────────────┘
```

### Stage 2: Smart Mode (LLM) - PLANNED

```
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Smart Mode (LLM Refinement) - CONDITIONAL     │
│ ┌─────────┐    ┌─────────────┐    ┌─────────────────┐  │
│ │ Text +  │ →  │ Ollama      │ →  │ Confirm/Reject  │  │
│ │ E5 Tags │    │ Qwen 7B     │    │ + Reasoning     │  │
│ └─────────┘    └─────────────┘    └─────────────────┘  │
│                                                         │
│ • Trigger: 0.70 ≤ avg_confidence < 0.75 (borderline)   │
│ • Expected: ~16% of documents                          │
│ • Time: +3-5s per document                             │
│ • Use case: Low-confidence tags, nuanced docs          │
└─────────────────────────────────────────────────────────┘
```

### Processing Flow

```
Document
   ↓
[pdfplumber] Extract text
   ↓
[E5-large-v2] Generate embeddings → cosine similarity
   ↓
Tags with confidence scores
   ↓
┌─────────────────────────────────────────────────┐
│ IF avg_confidence between 0.70-0.75:            │
│    → [Ollama/Qwen] Review & refine (PLANNED)    │
│ ELSE:                                           │
│    → Return E5 results directly                 │
└─────────────────────────────────────────────────┘
   ↓
Final tagged document
```

---

## Document Status Flow

| Status | Description | UI Color |
|--------|-------------|----------|
| `uploaded` | Pending processing | Gray |
| `processing` | Currently being analyzed | Blue |
| `completed` | Successfully tagged | Green |
| `failed` | Error during processing | Red |
| `ignored` | Manually dismissed (non-legal, etc) | Yellow |
| `needs_ocr` | Scanned PDF, requires OCR for text | Orange |

---

## Tag Categories

The system supports 8 primary legal document categories:

| Category | Examples |
|----------|----------|
| **Investment Funds** | Limited Partnership Agreement, Capital Calls, Distribution |
| **M&A / Corporate** | Merger Agreement, Due Diligence, Corporate Governance |
| **Securities / Capital Markets** | SEC Filings, Financial Statements, Prospectus |
| **Real Estate** | Purchase Agreement, Lease Agreement, Mortgage |
| **Intellectual Property** | Patent, Trademark, License Agreement |
| **Employment** | Employment Agreement, Non-Compete / NDA, Benefits |
| **Litigation** | Settlement, Discovery, Complaint |
| **Regulatory / Compliance** | Regulatory Filing, Anti-Corruption, Privacy |

---

## Hybrid Confidence Scoring

Tags are scored using a combination of:

1. **Semantic Similarity** (60% weight)
   - Cosine similarity between document embeddings and tag examples
   - Uses E5-large-v2 (1024-dimensional vectors)

2. **Pattern Matching** (up to 35% boost)
   - Regex patterns defined per tag
   - Word boundaries enforced for short patterns (e.g., `\bSEC\b`)

```
Final Score = (semantic × 0.6) + min(pattern_boost, 0.35)

Where:
  pattern_boost = 0.15 × log2(pattern_matches + 1)
```

### Confidence Thresholds

| Threshold | Action |
|-----------|--------|
| ≥ 0.65 | Tag assigned to document |
| 0.70 - 0.75 | Stage 2 LLM review (planned) |
| ≥ 0.75 | High confidence, skip LLM |

---

## GPU Acceleration

### Current Development Environment

| Hardware | E5 Embeddings | OCR/page | Throughput |
|----------|---------------|----------|------------|
| Apple M3 (MPS) | 2-5s | ~30s | 50-100 docs/min |

### Production Options (Planned)

| Hardware | E5 Embeddings | Qwen 7B LLM | OCR/page | Throughput |
|----------|---------------|-------------|----------|------------|
| RTX 3090 (24GB) | 0.2-0.5s | 2-3s | ~5s | 200-400 docs/min |
| A100 (40/80GB) | 0.1-0.3s | 0.5-1s | ~2-3s | 400-600 docs/min |
| H100 (80GB) | 0.05-0.1s | 0.3-0.5s | ~1s | 500-1000 docs/min |

---

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite
- **ML Runtime**: PyTorch + SentenceTransformers
- **PDF Processing**: pdfplumber
- **OCR**: Surya (optional)

### Frontend
- **Framework**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query
- **Charts**: Recharts
- **Icons**: Lucide React

### Models
- **Embeddings**: intfloat/e5-large-v2 (HuggingFace)
- **LLM (planned)**: Ollama with Qwen2.5:7b

---

## API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get document details |
| POST | `/api/documents/{id}/process` | Process single document |
| POST | `/api/documents/{id}/ignore` | Mark as ignored |

### Matters
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/matters` | List all matters |
| GET | `/api/matters/{id}` | Get matter with documents |
| POST | `/api/matters/{id}/process` | Process all documents |
| POST | `/api/matters/{id}/upload` | Upload documents |

### Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/metrics/summary` | Dashboard statistics |
| GET | `/api/training/summary` | Tag usage statistics |

---

## Future Roadmap

### Planned Features
- [ ] Stage 2 LLM integration (Ollama/Qwen)
- [ ] Individual document reprocess button
- [ ] Bulk reprocess by tag (when patterns change)
- [ ] Model comparison A/B testing
- [ ] GPU cloud deployment (A100/H100)

### Potential Enhancements
- [ ] Multi-language support
- [ ] Custom tag training UI
- [ ] Confidence calibration
- [ ] Active learning from feedback

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.3.0 | 2024-12-27 | Ignored status, needs_ocr detection, word boundary fix |
| 0.2.0 | 2024-12-26 | E5 model, tag highlights, fast pipeline |
| 0.1.0 | 2024-12-25 | Initial release with LegalBERT |
