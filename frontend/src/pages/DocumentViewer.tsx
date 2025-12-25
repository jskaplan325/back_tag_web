import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  BarChart2,
  Table,
  FileText,
  Loader2,
  AlertCircle,
  Image,
  AlignLeft
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface Tag {
  tag: string
  confidence: number
  pattern_matches: number
  semantic_similarity: number
}

interface VisualPages {
  visual_pages: number[]
  sparse_pages: number[]
  table_pages: number[]
}

interface DocumentDetail {
  id: string
  filename: string
  uploaded_at: string
  file_size_bytes: number
  page_count: number
  word_count: number
  status: string
  error_message: string | null
}

interface ResultDetail {
  id: string
  document_id: string
  processed_at: string
  processing_time_seconds: number
  semantic_model: string
  vision_model: string | null
  vision_enabled: boolean
  tag_count: number
  average_confidence: number
  result_json: {
    tags: Tag[]
    visual_pages?: VisualPages
    vision_analysis?: Array<{
      page: number
      description: string
      detected_objects: string[]
    }>
  }
}

function PageThumbnail({
  page,
  documentId,
  isActive,
  visualType,
  onClick
}: {
  page: number
  documentId: string
  isActive: boolean
  visualType: 'chart' | 'table' | 'sparse' | null
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'relative w-full aspect-[8.5/11] rounded border-2 overflow-hidden transition-all',
        isActive ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300',
        visualType === 'chart' && 'ring-2 ring-yellow-300',
        visualType === 'table' && 'ring-2 ring-blue-300',
        visualType === 'sparse' && 'ring-2 ring-purple-300'
      )}
    >
      <img
        src={`/api/documents/${documentId}/pages/${page}`}
        alt={`Page ${page}`}
        className="w-full h-full object-cover"
        loading="lazy"
      />
      <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs py-1 text-center">
        {page}
      </div>
      {visualType && (
        <div className={clsx(
          'absolute top-1 right-1 p-1 rounded',
          visualType === 'chart' && 'bg-yellow-500',
          visualType === 'table' && 'bg-blue-500',
          visualType === 'sparse' && 'bg-purple-500'
        )}>
          {visualType === 'chart' && <BarChart2 className="h-3 w-3 text-white" />}
          {visualType === 'table' && <Table className="h-3 w-3 text-white" />}
          {visualType === 'sparse' && <FileText className="h-3 w-3 text-white" />}
        </div>
      )}
    </button>
  )
}

interface TextContent {
  document_id: string
  text: string
  filename: string
}

export default function DocumentViewer() {
  const { id } = useParams<{ id: string }>()
  const [currentPage, setCurrentPage] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [viewMode, setViewMode] = useState<'auto' | 'text' | 'image'>('auto')

  const { data: document, isLoading: docLoading } = useQuery<DocumentDetail>({
    queryKey: ['document', id],
    queryFn: () => api.get(`/api/documents/${id}`).then(r => r.data),
  })

  const { data: result, isLoading: resultLoading } = useQuery<ResultDetail>({
    queryKey: ['document', id, 'result'],
    queryFn: () => api.get(`/api/documents/${id}/result`).then(r => r.data),
    enabled: !!document && document.status === 'completed',
  })

  const { data: textContent, isLoading: textLoading } = useQuery<TextContent>({
    queryKey: ['document', id, 'text'],
    queryFn: () => api.get(`/api/documents/${id}/text`).then(r => r.data),
    enabled: !!document,
  })

  const isLoading = docLoading || resultLoading

  // Determine if we should show text view
  const isTextFile = document?.filename?.toLowerCase().endsWith('.txt')
  const showTextView = viewMode === 'text' || (viewMode === 'auto' && isTextFile)

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (!document) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <AlertCircle className="mx-auto h-12 w-12 text-red-400" />
          <p className="mt-4 text-gray-600">Document not found</p>
          <Link to="/documents" className="mt-4 inline-block text-blue-600 hover:underline">
            Back to documents
          </Link>
        </div>
      </div>
    )
  }

  const visualPages = result?.result_json?.visual_pages
  const tags = result?.result_json?.tags || []
  const pageCount = document.page_count || 1

  const getVisualType = (page: number): 'chart' | 'table' | 'sparse' | null => {
    if (!visualPages) return null
    if (visualPages.visual_pages?.includes(page)) return 'chart'
    if (visualPages.table_pages?.includes(page)) return 'table'
    if (visualPages.sparse_pages?.includes(page)) return 'sparse'
    return null
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-white px-4 py-3">
        <div className="flex items-center gap-4">
          <Link to="/documents" className="text-gray-500 hover:text-gray-700">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <h1 className="text-lg font-semibold">{document.filename}</h1>
        </div>
        <div className="flex items-center gap-4">
          {/* View mode toggle */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('text')}
              className={clsx(
                'flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors',
                showTextView ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'
              )}
            >
              <AlignLeft className="h-4 w-4" />
              Text
            </button>
            <button
              onClick={() => setViewMode('image')}
              disabled={isTextFile}
              className={clsx(
                'flex items-center gap-1 px-3 py-1.5 rounded text-sm transition-colors',
                !showTextView ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900',
                isTextFile && 'opacity-50 cursor-not-allowed'
              )}
            >
              <Image className="h-4 w-4" />
              Pages
            </button>
          </div>
          {/* Zoom controls - only for image view */}
          {!showTextView && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <span className="text-sm text-gray-600">{Math.round(zoom * 100)}%</span>
              <button
                onClick={() => setZoom(z => Math.min(3, z + 0.25))}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar - Page Thumbnails (only for image view) */}
        {!showTextView && (
          <div className="w-48 border-r bg-gray-50 overflow-y-auto p-3 space-y-2">
            <div className="mb-3">
              <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider">Pages</h3>
              {visualPages && (
                <div className="mt-2 space-y-1 text-xs">
                  {visualPages.visual_pages?.length > 0 && (
                    <div className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-yellow-500" />
                      <span>{visualPages.visual_pages.length} charts/images</span>
                    </div>
                  )}
                  {visualPages.table_pages?.length > 0 && (
                    <div className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-blue-500" />
                      <span>{visualPages.table_pages.length} tables</span>
                    </div>
                  )}
                  {visualPages.sparse_pages?.length > 0 && (
                    <div className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded bg-purple-500" />
                      <span>{visualPages.sparse_pages.length} sparse</span>
                    </div>
                  )}
                </div>
              )}
            </div>
            {Array.from({ length: pageCount }, (_, i) => i + 1).map(page => (
              <PageThumbnail
                key={page}
                page={page}
                documentId={document.id}
                isActive={page === currentPage}
                visualType={getVisualType(page)}
                onClick={() => setCurrentPage(page)}
              />
            ))}
          </div>
        )}

        {/* Center - Content Viewer */}
        <div className="flex-1 overflow-auto bg-gray-200 p-4">
          {showTextView ? (
            /* Text View */
            <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg p-8">
              {textLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
                </div>
              ) : textContent?.text ? (
                <pre className="whitespace-pre-wrap font-mono text-sm text-gray-800 leading-relaxed">
                  {textContent.text}
                </pre>
              ) : (
                <div className="text-center text-gray-400 py-12">
                  <FileText className="mx-auto h-12 w-12 mb-4" />
                  <p>No text content available</p>
                </div>
              )}
            </div>
          ) : (
            /* Image/Page View */
            <div className="flex items-center justify-center min-h-full">
              <div
                style={{ transform: `scale(${zoom})`, transformOrigin: 'center' }}
                className="transition-transform"
              >
                <img
                  src={`/api/documents/${document.id}/pages/${currentPage}`}
                  alt={`Page ${currentPage}`}
                  className="max-w-none shadow-lg bg-white"
                  onError={(e) => {
                    // If image fails to load, switch to text view
                    const target = e.target as HTMLImageElement
                    target.style.display = 'none'
                  }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar - Tags & Info */}
        <div className="w-80 border-l bg-white overflow-y-auto">
          <div className="p-4 border-b">
            <h3 className="font-medium">Document Info</h3>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">Pages</dt>
                <dd>{document.page_count}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Words</dt>
                <dd>{document.word_count?.toLocaleString()}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Status</dt>
                <dd className={clsx(
                  'px-2 py-0.5 rounded text-xs',
                  document.status === 'completed' && 'bg-green-100 text-green-700',
                  document.status === 'failed' && 'bg-red-100 text-red-700',
                  document.status === 'processing' && 'bg-yellow-100 text-yellow-700'
                )}>
                  {document.status}
                </dd>
              </div>
              {result && (
                <>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Processing Time</dt>
                    <dd>{result.processing_time_seconds.toFixed(1)}s</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Model</dt>
                    <dd className="text-xs truncate max-w-32">{result.semantic_model.split('/').pop()}</dd>
                  </div>
                </>
              )}
            </dl>
          </div>

          {tags.length > 0 && (
            <div className="p-4">
              <h3 className="font-medium mb-3">Detected Tags</h3>
              <div className="space-y-2">
                {tags.map((tag) => (
                  <div
                    key={tag.tag}
                    className="rounded-lg border p-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{tag.tag}</span>
                      <span className={clsx(
                        'text-xs px-2 py-0.5 rounded',
                        tag.confidence >= 0.7 ? 'bg-green-100 text-green-700' :
                        tag.confidence >= 0.5 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-gray-100 text-gray-700'
                      )}>
                        {(tag.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      <div>Patterns: {tag.pattern_matches}</div>
                      <div>Semantic: {(tag.semantic_similarity * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Visual Page Analysis */}
          {result?.result_json?.vision_analysis && result.result_json.vision_analysis.length > 0 && (
            <div className="p-4 border-t">
              <h3 className="font-medium mb-3">Vision Analysis</h3>
              <div className="space-y-2">
                {result.result_json.vision_analysis.map((va) => (
                  <div key={va.page} className="text-sm rounded border p-2">
                    <div className="font-medium">Page {va.page}</div>
                    <p className="text-gray-600 text-xs mt-1">{va.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer Navigation - only for image view */}
      {!showTextView && (
        <div className="flex items-center justify-center gap-4 border-t bg-white py-3">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
          <span className="text-sm">
            Page {currentPage} of {pageCount}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(pageCount, p + 1))}
            disabled={currentPage === pageCount}
            className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        </div>
      )}
    </div>
  )
}
