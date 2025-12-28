# Back Tag v0.3.0 - Document Intelligence Dashboard

Web dashboard for legal document classification using a two-stage AI pipeline.

## Features

- **Two-Stage AI Pipeline**: E5 embeddings + Qwen LLM refinement
- **8 Legal Tag Categories**: Investment Funds, M&A, Securities, Real Estate, IP, Employment, Litigation, Regulatory
- **Smart Pattern Matching**: Word-boundary aware regex patterns
- **Scanned PDF Detection**: Auto-flags documents needing OCR
- **Matter Management**: Organize documents by matter/case

## Architecture

```
back_tag_web/
├── backend/              # FastAPI server
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── routers/             # API endpoints
│   │   ├── services/
│   │   │   ├── fast_tagger.py   # E5 embedding pipeline
│   │   │   └── llm_tagger.py    # Qwen LLM refinement
│   │   └── database/            # SQLAlchemy models
│   └── start.sh                 # Server start script
├── frontend/             # React + TypeScript
│   └── src/pages/               # Dashboard pages
├── docs/
│   ├── ARCHITECTURE.md          # Full technical docs
│   └── ARCHITECTURE.html        # Printable version
└── data/
    ├── uploads/                 # Uploaded documents
    └── dashboard.db             # SQLite database
```

## Two-Stage Processing Pipeline

```
Document → Extract Text → E5 Embeddings → Confidence Check
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                              conf ≥ 0.75          0.70 ≤ conf < 0.75
                              or conf < 0.70           (borderline)
                                    ↓                   ↓
                                  DONE             Stage 2: LLM
                                                        ↓
                                              ┌─────────────────┐
                                              │ Ollama/Qwen 7B  │
                                              │ Confirm/Reject  │
                                              └─────────────────┘
                                                        ↓
                                                      DONE
```

### Stage 1: E5 Embeddings (Always Runs)
- **Model**: intfloat/e5-large-v2
- **Method**: Cosine similarity + pattern matching
- **Speed**: ~2-5s per document
- **Threshold**: 0.65 minimum confidence

### Stage 2: LLM Refinement (Conditional)
- **Model**: Ollama/qwen2.5:7b
- **Trigger**: Average confidence 70-75%
- **Action**: Confirms or rejects borderline tags
- **Speed**: +10-15s when triggered (~16% of docs)

## Model Registry

| Type | Model | Size | Purpose |
|------|-------|------|---------|
| **Embedding** | intfloat/e5-large-v2 | 1.3GB | Stage 1: Fast similarity |
| **Embedding** | pile-of-law/legalbert | 1.3GB | Alternative (legal domain) |
| **LLM** | qwen2.5:7b | 4.7GB | Stage 2: Tag refinement |
| **OCR** | surya | - | Scanned PDF extraction |

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Ollama (for LLM refinement)

### 1. Install Ollama & Qwen

```bash
# macOS
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
./start.sh
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 4. Access

- **Dashboard**: http://localhost:5173/back_tag_web
- **API Docs**: http://localhost:8000/docs

## Document Status Flow

| Status | Description | UI Color |
|--------|-------------|----------|
| `uploaded` | Pending processing | Gray |
| `processing` | Currently analyzing | Blue |
| `completed` | Successfully tagged | Green |
| `failed` | Processing error | Red |
| `ignored` | Manually dismissed | Yellow |
| `needs_ocr` | Scanned PDF detected | Orange |

## API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get document details |
| POST | `/api/documents/{id}/process` | Process document |
| POST | `/api/documents/{id}/ignore` | Mark as ignored |

### Matters
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/matters` | List all matters |
| POST | `/api/matters/{id}/process` | Process all documents |
| POST | `/api/matters/{id}/upload` | Upload documents |

### Metrics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/metrics/summary` | Dashboard statistics |
| GET | `/api/training/summary` | Tag usage stats |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama server |
| `OLLAMA_MODEL` | qwen2.5:7b | LLM model for Stage 2 |

## Future Roadmap

### Planned
- [ ] GPU cloud deployment (A100/H100)
- [ ] Individual document reprocess
- [ ] Model comparison A/B testing

### Integrations
- [ ] MatterMgmt DB - Pull matter metadata
- [ ] DMS API Framework - Auto-pull documents

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.3.0 | 2024-12-27 | Two-stage pipeline, LLM refinement, word boundaries |
| 0.2.0 | 2024-12-26 | E5 model, tag highlights, fast pipeline |
| 0.1.0 | 2024-12-25 | Initial release |

## License

MIT
