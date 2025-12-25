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

Environment variables:
- `DATABASE_URL` - Database connection (default: SQLite)
- `DATA_DIR` - Data directory (default: ./data)

## License

MIT
