# Document Intelligence Dashboard

Web dashboard for the Document Intelligence Pipeline. Provides a visual interface for:
- Document upload and processing
- Visual page review (charts, tables, images)
- Model registry with HuggingFace integration
- Processing metrics and cost tracking

## Architecture

```
back_tag_web/
├── backend/          # FastAPI server
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── routers/          # API endpoints
│   │   ├── models/           # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   └── database/         # SQLAlchemy models
│   └── requirements.txt
├── frontend/         # React app
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
├── data/
│   ├── uploads/      # Uploaded PDFs
│   └── results/      # Processing results
└── docker-compose.yml
```

## Processing Architecture

```
Document → Quick Scan → Select Pipeline
                ↓
           ┌─────────────────────────────────────┐
           │ Detect:                             │
           │ • File type (PDF/TXT/DOCX/image)    │
           │ • Page count                        │
           │ • Has visual content (charts/tables)│
           │ • Text density (sparse = needs OCR) │
           │ • Document structure (has sections) │
           └─────────────────────────────────────┘
                ↓
      ┌─────────┴─────────┬──────────────┐
      ↓                   ↓              ↓
  Text-heavy          Long/Structured   Visual/Scanned
  (agreements)        (SEC filings)     (charts, images)
      ↓                   ↓              ↓
  Fast Pipeline       Zone + Smart     Vision + OCR + Tag
  (~3s)               (~10s)           (~15s)
      │                   │              │
      ▼                   ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Semantic (Embeddings)                           │   │
│  │ • pile-of-law/legalbert-large-1.7M-2 [HF]      │   │
│  │ • Qwen/Qwen2.5-7B-Instruct [HF/Ollama]         │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Vision (Image Analysis)                         │   │
│  │ • microsoft/Florence-2-large [HF]              │   │
│  │ • microsoft/Florence-2-base [HF]               │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ OCR (Text Extraction)                           │   │
│  │ • datalab-to/surya [GitHub]                    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ LLM (Smart Tagging)                             │   │
│  │ • Ollama/qwen2.5:7b [Local] ← preferred        │   │
│  │ • gemini-2.0-flash [Google API]                │   │
│  │ • gpt-4o-mini [OpenAI API]                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Status: [RAI Approved] [Pending Review]               │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Modes

**Fast Mode (Pattern + Semantic)** - ~3 sec
- Extract text → LegalBERT embeddings → Pattern match + cosine similarity
- Local only, good for standard documents

**Smart Mode (LLM-based)** - ~8-15 sec
- Extract text → Ollama/Gemini LLM → Structured JSON tag response
- Local (Ollama) or API, better accuracy for complex documents

**Vision Mode (Visual Page Detection)** - +5-10 sec
- Render pages → Florence-2 vision → Classify charts/tables/diagrams
- Flags visual content for human review

### Pipeline Selection Logic

| Document Type   | Indicators                  | Pipeline         |
|-----------------|-----------------------------|------------------|
| Text agreement  | .txt, PDF high text density | Fast Mode        |
| Complex legal   | Nuanced language, ambiguous | Smart Mode       |
| Long SEC filing | PDF, 50+ pages, ITEM headers| Zone → Smart     |
| Visual report   | PDF with images/charts      | Vision → Tag     |
| Scanned doc     | PDF with low text extraction| Surya OCR → Tag  |

### LLM Backend Priority

1. **Ollama (Local)** - Free, private, ~7GB RAM
2. **Gemini (API)** - Free tier, then $0.075/1M tokens
3. **OpenAI (API)** - $0.15/1M input, $0.60/1M output

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- back_tag package (document intelligence pipeline)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install back_tag from sibling directory
pip install -e ../../back_tag

# Run server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Access

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Endpoints

### Documents
- `POST /api/documents/upload` - Upload PDF
- `GET /api/documents` - List documents
- `POST /api/documents/{id}/process` - Process document
- `GET /api/documents/{id}/pages/{page}` - Get page image

### Results
- `GET /api/results` - List results
- `GET /api/results/{id}` - Get result details
- `GET /api/results/{id}/tags` - Get tag breakdown

### Models
- `GET /api/models` - List registered models
- `POST /api/models` - Register model (fetches HuggingFace info)
- `PATCH /api/models/{id}` - Update approval status

### Metrics
- `GET /api/metrics/summary` - Dashboard stats
- `GET /api/metrics/processing` - Processing trends
- `GET /api/metrics/models` - Model usage
- `GET /api/metrics/costs` - API costs

## Features

### Document Processing
- Upload PDFs via drag-and-drop
- Select semantic model (LegalBERT, all-MiniLM, etc.)
- Optional Florence-2 vision analysis
- Background processing with status updates

### Visual Page Reviewer
- Page-by-page navigation
- Visual content highlighting (charts, tables)
- Side-by-side results view
- Zoom and pan controls

### Model Registry
- Automatic HuggingFace metadata fetching
- Approval workflow
- Usage statistics
- Size and license info

### Metrics Dashboard
- Processing trends over time
- Tag detection rates
- Model performance comparison
- API cost tracking

## Configuration

### Environment Variables

- `DATABASE_URL` - Database connection (default: SQLite)
- `DATA_DIR` - Data directory (default: ./data)
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Ollama model for Smart Mode (default: qwen2.5:7b)
- `GOOGLE_API_KEY` - Google API key for Gemini (optional, fallback if no Ollama)

### Ollama Setup (Recommended for Local LLM)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Pull Qwen 2.5 7B model (~4.7GB)
ollama pull qwen2.5:7b

# Verify
ollama list
```

The system auto-detects Ollama and prefers it over cloud APIs for privacy and cost.

## License

MIT
