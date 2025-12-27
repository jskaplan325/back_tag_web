import { useState, useEffect, useCallback } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  ZoomIn,
  ZoomOut,
  BarChart2,
  Table,
  FileText,
  Loader2,
  AlertCircle,
  Image,
  AlignLeft,
  Check,
  X,
  RotateCcw,
  PenTool,
  Trash2,
  EyeOff
} from 'lucide-react'
import { AnnotationCanvas, type Annotation as AnnotationType, type BoundingBox } from '../components/AnnotationCanvas'
import { TagAssignmentModal } from '../components/TagAssignmentModal'
import clsx from 'clsx'
import api from '../api'

interface Highlight {
  start: number
  end: number
  text: string
  pattern: string
}

interface Tag {
  tag: string
  confidence: number
  pattern_matches: number
  semantic_similarity: number
  highlights?: Highlight[]
  area?: string
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

interface TagFeedback {
  id: string
  tag_name: string
  action: 'confirmed' | 'rejected' | 'added'
  original_confidence: number
  reviewed_at: string
  reviewed_by: string | null
}

interface AnnotationData {
  id: string
  document_id: string
  page_number: number
  x1: number
  y1: number
  x2: number
  y2: number
  tag_name: string
  tag_id: string | null
  area_of_law: string | null
  annotation_type: 'positive' | 'negative' | 'uncertain'
  color: 'green' | 'yellow' | 'red'
  source: string
  created_by: string | null
  created_at: string
  notes: string | null
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

// Color palette for different tags
const TAG_COLORS: Record<string, string> = {
  'M&A / Corporate': 'bg-purple-200 border-purple-400',
  'Securities / Capital Markets': 'bg-blue-200 border-blue-400',
  'Investment Funds': 'bg-amber-200 border-amber-400',
  'Litigation': 'bg-red-200 border-red-400',
  'Real Estate': 'bg-green-200 border-green-400',
  'Employment': 'bg-cyan-200 border-cyan-400',
  'Intellectual Property': 'bg-pink-200 border-pink-400',
  'Regulatory / Compliance': 'bg-indigo-200 border-indigo-400',
}

const DEFAULT_HIGHLIGHT_COLOR = 'bg-yellow-200 border-yellow-400'

export default function DocumentViewer() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [currentPage, setCurrentPage] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [viewMode, setViewMode] = useState<'auto' | 'text' | 'image'>('auto')
  const [selectedTag, setSelectedTag] = useState<string | null>(null)
  const [selectedTagOccurrenceIndex, setSelectedTagOccurrenceIndex] = useState(0)

  // Annotation state
  const [annotationMode, setAnnotationMode] = useState(false)
  const [pendingBox, setPendingBox] = useState<BoundingBox | null>(null)
  const [showTagModal, setShowTagModal] = useState(false)
  const [selectedAnnotation, setSelectedAnnotation] = useState<AnnotationType | null>(null)

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

  // Fetch existing feedback for this document
  const { data: feedbackList } = useQuery<TagFeedback[]>({
    queryKey: ['document', id, 'feedback'],
    queryFn: () => api.get(`/api/documents/${id}/feedback`).then(r => r.data),
    enabled: !!id,
  })

  // Create a map for quick lookup
  const feedbackMap = new Map<string, TagFeedback>()
  feedbackList?.forEach(fb => feedbackMap.set(fb.tag_name, fb))

  // Submit feedback mutation
  const submitFeedback = useMutation({
    mutationFn: ({ tagName, action, confidence }: { tagName: string; action: 'confirmed' | 'rejected'; confidence: number }) =>
      api.post(`/api/documents/${id}/feedback`, {
        tag_name: tagName,
        action,
        original_confidence: confidence,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id, 'feedback'] })
    },
  })

  // Undo feedback mutation
  const undoFeedback = useMutation({
    mutationFn: (tagName: string) =>
      api.delete(`/api/documents/${id}/feedback/${encodeURIComponent(tagName)}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id, 'feedback'] })
    },
  })

  // Determine if we should show text view (needed before annotation query)
  const isTextFile = document?.filename?.toLowerCase().endsWith('.txt')
  const showTextView = viewMode === 'text' || (viewMode === 'auto' && isTextFile)

  // Fetch annotations for current page (for canvas display)
  const { data: annotationsData } = useQuery<AnnotationData[]>({
    queryKey: ['document', id, 'annotations', currentPage],
    queryFn: () => api.get(`/api/documents/${id}/pages/${currentPage}/annotations`).then(r => r.data),
    enabled: !!id && !showTextView,
  })

  // Fetch ALL annotations for the document (for sidebar display)
  const { data: allAnnotationsData } = useQuery<AnnotationData[]>({
    queryKey: ['document', id, 'annotations', 'all'],
    queryFn: () => api.get(`/api/documents/${id}/annotations`).then(r => r.data),
    enabled: !!id,
  })

  // Convert API annotations to canvas format
  const annotations: AnnotationType[] = (annotationsData || []).map(ann => ({
    id: ann.id,
    x1: ann.x1,
    y1: ann.y1,
    x2: ann.x2,
    y2: ann.y2,
    tag_name: ann.tag_name,
    color: ann.color,
    annotation_type: ann.annotation_type
  }))

  // Create annotation mutation
  const createAnnotation = useMutation({
    mutationFn: (data: {
      page_number: number
      x1: number
      y1: number
      x2: number
      y2: number
      tag_name: string
      tag_id: string | null
      area_of_law: string | null
      annotation_type: string
      color: string
    }) => api.post(`/api/documents/${id}/annotations`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id, 'annotations'] })
      setShowTagModal(false)
      setPendingBox(null)
    },
  })

  // Delete annotation mutation
  const deleteAnnotation = useMutation({
    mutationFn: (annotationId: string) => api.delete(`/api/annotations/${annotationId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id, 'annotations'] })
      setSelectedAnnotation(null)
    },
  })

  // Retry with skip validation mutation (for "Process Anyway" on failed docs)
  const retryWithSkipValidation = useMutation({
    mutationFn: () => api.post(`/api/documents/${id}/retry`, { skip_validation: true }),
    onSuccess: () => {
      // Invalidate document query to get updated status
      queryClient.invalidateQueries({ queryKey: ['document', id] })
    },
  })

  // Ignore document mutation (mark as ignored, remove from failed/pending counts)
  const ignoreDocument = useMutation({
    mutationFn: () => api.post(`/api/documents/${id}/ignore`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id] })
      navigate(-1) // Go back after ignoring
    },
  })

  // Derived values that don't need document to be loaded
  const tags = result?.result_json?.tags || []

  // Compute occurrences for the selected tag (for navigation)
  // Must be before any early returns to follow React hooks rules
  const getTagOccurrences = useCallback(() => {
    if (!selectedTag) return []

    if (showTextView) {
      // Text view: use highlights from the tag
      const tag = tags.find(t => t.tag === selectedTag)
      return (tag?.highlights || []).map((h, i) => ({
        type: 'highlight' as const,
        index: i,
        start: h.start,
        end: h.end,
        text: h.text
      }))
    } else {
      // Image view: use annotations with this tag name
      const tagAnnotations = (allAnnotationsData || []).filter(
        ann => ann.tag_name === selectedTag
      )
      return tagAnnotations.map((ann, i) => ({
        type: 'annotation' as const,
        index: i,
        page: ann.page_number,
        annotationId: ann.id
      }))
    }
  }, [selectedTag, showTextView, tags, allAnnotationsData])

  const tagOccurrences = getTagOccurrences()
  const totalOccurrences = tagOccurrences.length

  // Reset occurrence index when tag changes
  useEffect(() => {
    setSelectedTagOccurrenceIndex(0)
  }, [selectedTag])

  // Navigate to current occurrence
  useEffect(() => {
    if (!selectedTag || totalOccurrences === 0) return

    const currentOccurrence = tagOccurrences[selectedTagOccurrenceIndex]
    if (!currentOccurrence) return

    if (currentOccurrence.type === 'highlight' && showTextView) {
      // Scroll to highlight in text view
      setTimeout(() => {
        const highlightEl = window.document.querySelector(`[data-highlight-index="${selectedTagOccurrenceIndex}"]`)
        if (highlightEl) {
          highlightEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }, 100)
    } else if (currentOccurrence.type === 'annotation' && !showTextView) {
      // Switch to the page with this annotation
      setCurrentPage(currentOccurrence.page)
    }
  }, [selectedTag, selectedTagOccurrenceIndex, showTextView, totalOccurrences, tagOccurrences])

  // Select annotation when page loads with the annotation (separate effect to avoid loops)
  useEffect(() => {
    if (!selectedTag || totalOccurrences === 0 || showTextView) return

    const currentOccurrence = tagOccurrences[selectedTagOccurrenceIndex]
    if (!currentOccurrence || currentOccurrence.type !== 'annotation') return

    // Find and select the annotation on the current page
    const ann = annotations.find(a => a.id === currentOccurrence.annotationId)
    if (ann) {
      setSelectedAnnotation(ann)
    }
  }, [selectedTag, selectedTagOccurrenceIndex, annotations, showTextView, totalOccurrences, tagOccurrences])

  const goToPrevOccurrence = useCallback(() => {
    if (totalOccurrences === 0) return
    setSelectedTagOccurrenceIndex(i => (i - 1 + totalOccurrences) % totalOccurrences)
  }, [totalOccurrences])

  const goToNextOccurrence = useCallback(() => {
    if (totalOccurrences === 0) return
    setSelectedTagOccurrenceIndex(i => (i + 1) % totalOccurrences)
  }, [totalOccurrences])

  // Handle keyboard navigation for tag occurrences
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!selectedTag || totalOccurrences === 0) return
      if (e.key === 'ArrowUp' || e.key === 'k') {
        e.preventDefault()
        goToPrevOccurrence()
      } else if (e.key === 'ArrowDown' || e.key === 'j') {
        e.preventDefault()
        goToNextOccurrence()
      } else if (e.key === 'Escape') {
        setSelectedTag(null)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedTag, totalOccurrences, goToPrevOccurrence, goToNextOccurrence])

  const isLoading = docLoading || resultLoading

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

  // For failed documents, we'll show the viewer with an error banner (handled below)
  const isFailedDoc = document.status === 'failed'
  const isFailedReviewed = feedbackMap.has('__failed_reviewed__')

  const visualPages = result?.result_json?.visual_pages
  const pageCount = document.page_count || 1

  // Get highlights for selected tag or all tags
  const getHighlights = (): Array<Highlight & { tagName: string; area: string }> => {
    const allHighlights: Array<Highlight & { tagName: string; area: string }> = []
    for (const tag of tags) {
      if (selectedTag && tag.tag !== selectedTag) continue
      for (const h of tag.highlights || []) {
        allHighlights.push({ ...h, tagName: tag.tag, area: tag.area || '' })
      }
    }
    // Sort by position, remove overlaps
    allHighlights.sort((a, b) => a.start - b.start)
    return allHighlights
  }

  // Render text with highlighted matches
  const renderHighlightedText = (text: string) => {
    const highlights = getHighlights()
    if (highlights.length === 0) {
      return <span>{text}</span>
    }

    const parts: JSX.Element[] = []
    let lastEnd = 0

    // Track which highlight index this is for the selected tag
    let selectedTagHighlightIdx = 0

    for (let i = 0; i < highlights.length; i++) {
      const h = highlights[i]
      // Skip overlapping highlights
      if (h.start < lastEnd) continue

      // Add text before highlight
      if (h.start > lastEnd) {
        parts.push(<span key={`text-${i}`}>{text.slice(lastEnd, h.start)}</span>)
      }

      // Check if this is the currently active occurrence
      const isCurrentOccurrence = selectedTag === h.tagName &&
        selectedTagHighlightIdx === selectedTagOccurrenceIndex

      // Add highlighted text
      const colorClass = TAG_COLORS[h.area] || DEFAULT_HIGHLIGHT_COLOR
      parts.push(
        <mark
          key={`hl-${i}`}
          className={clsx(
            'px-0.5 rounded border transition-all',
            colorClass,
            isCurrentOccurrence && 'ring-2 ring-blue-500 ring-offset-1'
          )}
          title={`${h.tagName}: "${h.text}"`}
          data-highlight-index={selectedTag === h.tagName ? selectedTagHighlightIdx : undefined}
        >
          {text.slice(h.start, h.end)}
        </mark>
      )

      // Increment counter for selected tag
      if (selectedTag === h.tagName) {
        selectedTagHighlightIdx++
      }
      lastEnd = h.end
    }

    // Add remaining text
    if (lastEnd < text.length) {
      parts.push(<span key="text-end">{text.slice(lastEnd)}</span>)
    }

    return <>{parts}</>
  }

  const getVisualType = (page: number): 'chart' | 'table' | 'sparse' | null => {
    if (!visualPages) return null
    if (visualPages.visual_pages?.includes(page)) return 'chart'
    if (visualPages.table_pages?.includes(page)) return 'table'
    if (visualPages.sparse_pages?.includes(page)) return 'sparse'
    return null
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Error Banner for Failed Documents */}
      {isFailedDoc && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <div>
                <p className="text-sm font-medium text-red-800">Processing Failed</p>
                <p className="text-xs text-red-600">{document.error_message || 'Unknown error'}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {/* Process Anyway button - skip validation and retry */}
              <button
                onClick={() => retryWithSkipValidation.mutate()}
                disabled={retryWithSkipValidation.isPending}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                title="Skip validation checks and process this document anyway"
              >
                {retryWithSkipValidation.isPending ? (
                  <><Loader2 className="h-3 w-3 animate-spin" /> Processing...</>
                ) : (
                  <><RotateCcw className="h-3 w-3" /> Process Anyway</>
                )}
              </button>
              {/* Ignore button - mark as ignored and go back */}
              <button
                onClick={() => ignoreDocument.mutate()}
                disabled={ignoreDocument.isPending}
                className="flex items-center gap-1 px-3 py-1.5 text-xs bg-yellow-500 text-white rounded hover:bg-yellow-600 transition-colors disabled:opacity-50"
                title="Mark this document as ignored (non-processable)"
              >
                {ignoreDocument.isPending ? (
                  <><Loader2 className="h-3 w-3 animate-spin" /> Ignoring...</>
                ) : (
                  <><EyeOff className="h-3 w-3" /> Ignore</>
                )}
              </button>
              <div className="border-l border-red-300 h-6 mx-1" />
              {isFailedReviewed ? (
                <>
                  <span className={clsx(
                    "text-xs px-2 py-1 rounded flex items-center gap-1",
                    feedbackMap.get('__failed_reviewed__')?.action === 'confirmed'
                      ? "text-green-600 bg-green-100"
                      : "text-gray-500 bg-gray-200"
                  )}>
                    {feedbackMap.get('__failed_reviewed__')?.action === 'confirmed' ? (
                      <><Check className="h-3 w-3" /> Confirmed</>
                    ) : (
                      <><X className="h-3 w-3" /> Dismissed</>
                    )}
                  </span>
                  <button
                    onClick={() => undoFeedback.mutate('__failed_reviewed__')}
                    className="text-xs text-gray-500 hover:text-gray-700 p-1"
                    title="Undo"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => submitFeedback.mutate({
                      tagName: '__failed_reviewed__',
                      action: 'confirmed',
                      confidence: 0
                    })}
                    disabled={submitFeedback.isPending}
                    className="flex items-center gap-1 px-2 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                  >
                    <Check className="h-3 w-3" />
                    Confirm
                  </button>
                  <button
                    onClick={() => submitFeedback.mutate({
                      tagName: '__failed_reviewed__',
                      action: 'rejected',
                      confidence: 0
                    })}
                    disabled={submitFeedback.isPending}
                    className="flex items-center gap-1 px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                  >
                    <X className="h-3 w-3" />
                    Dismiss
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between border-b bg-white px-4 py-3">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="text-gray-500 hover:text-gray-700"
            title="Go back"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
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
          {/* Annotate toggle - only for image view */}
          {!showTextView && (
            <button
              onClick={() => setAnnotationMode(!annotationMode)}
              className={clsx(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                annotationMode
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              )}
            >
              <PenTool className="h-4 w-4" />
              {annotationMode ? 'Drawing...' : 'Annotate'}
            </button>
          )}
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
                <div className="whitespace-pre-wrap font-mono text-sm text-gray-800 leading-relaxed">
                  {renderHighlightedText(textContent.text)}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-12">
                  <FileText className="mx-auto h-12 w-12 mb-4" />
                  <p>No text content available</p>
                </div>
              )}
            </div>
          ) : (
            /* Image/Page View with Annotation Canvas */
            <div className="flex items-center justify-center min-h-full">
              <div
                style={{ transform: `scale(${zoom})`, transformOrigin: 'center' }}
                className="transition-transform relative"
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
                {/* Annotation Canvas Overlay */}
                <AnnotationCanvas
                  annotations={annotations}
                  isDrawingEnabled={annotationMode}
                  onAnnotationCreate={(box) => {
                    setPendingBox(box)
                    setShowTagModal(true)
                  }}
                  onAnnotationSelect={(ann) => setSelectedAnnotation(ann)}
                  onAnnotationDelete={(annId) => deleteAnnotation.mutate(annId)}
                  selectedAnnotationId={selectedAnnotation?.id}
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

          {/* Tags Section - only show for completed documents (not failed) */}
          {document.status === 'completed' && !isFailedDoc && (
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium">Detected Tags</h3>
                {selectedTag && (
                  <button
                    onClick={() => setSelectedTag(null)}
                    className="text-xs text-blue-600 hover:underline"
                  >
                    Clear selection
                  </button>
                )}
              </div>

              {/* Tag Navigation Bar - shown when a tag is selected */}
              {selectedTag && totalOccurrences > 0 && (
                <div className="mb-3 p-2 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-blue-700">
                        {selectedTag}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={goToPrevOccurrence}
                        className="p-1 hover:bg-blue-100 rounded text-blue-600"
                        title="Previous (↑ or k)"
                      >
                        <ChevronUp className="h-4 w-4" />
                      </button>
                      <span className="text-xs font-medium text-blue-700 min-w-[3rem] text-center">
                        {selectedTagOccurrenceIndex + 1} / {totalOccurrences}
                      </span>
                      <button
                        onClick={goToNextOccurrence}
                        className="p-1 hover:bg-blue-100 rounded text-blue-600"
                        title="Next (↓ or j)"
                      >
                        <ChevronDown className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-blue-600 mt-1">
                    Use ↑↓ keys to navigate • Esc to close
                  </p>
                </div>
              )}
              {tags.length === 0 && (!allAnnotationsData || allAnnotationsData.length === 0) ? (
                /* No ML tags AND no user annotations */
                <div className="text-center py-6 bg-amber-50 rounded-lg border border-amber-200">
                  <AlertCircle className="mx-auto h-8 w-8 text-amber-500 mb-2" />
                  <p className="text-sm text-amber-700 font-medium">No tags detected</p>
                  <p className="text-xs text-amber-600 mt-1">This document needs manual review</p>
                  {feedbackMap.has('__no_tags__') ? (
                    <div className="mt-4 flex flex-col items-center gap-2">
                      <span className={clsx(
                        "text-sm px-3 py-1.5 rounded-lg flex items-center gap-1",
                        feedbackMap.get('__no_tags__')?.action === 'confirmed'
                          ? "text-green-600 bg-green-50"
                          : "text-gray-500 bg-gray-100"
                      )}>
                        {feedbackMap.get('__no_tags__')?.action === 'confirmed' ? (
                          <><Check className="h-4 w-4" /> Confirmed OK</>
                        ) : (
                          <><X className="h-4 w-4" /> Dismissed</>
                        )}
                      </span>
                      <button
                        onClick={() => undoFeedback.mutate('__no_tags__')}
                        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 px-2 py-1 hover:bg-gray-100 rounded"
                      >
                        <RotateCcw className="h-3 w-3" />
                        Undo
                      </button>
                    </div>
                  ) : (
                    <div className="mt-4 flex items-center justify-center gap-2">
                      <button
                        onClick={() => submitFeedback.mutate({
                          tagName: '__no_tags__',
                          action: 'confirmed',
                          confidence: 0
                        })}
                        disabled={submitFeedback.isPending}
                        className="flex items-center gap-1 px-3 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                      >
                        <Check className="h-4 w-4" />
                        Confirm OK
                      </button>
                      <button
                        onClick={() => submitFeedback.mutate({
                          tagName: '__no_tags__',
                          action: 'rejected',
                          confidence: 0
                        })}
                        disabled={submitFeedback.isPending}
                        className="flex items-center gap-1 px-3 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                      >
                        <X className="h-4 w-4" />
                        Dismiss
                      </button>
                    </div>
                  )}
                </div>
              ) : tags.length === 0 && allAnnotationsData && allAnnotationsData.length > 0 ? (
                /* No ML tags BUT user has added annotations */
                <div className="text-center py-4 bg-green-50 rounded-lg border border-green-200">
                  <Check className="mx-auto h-6 w-6 text-green-500 mb-2" />
                  <p className="text-sm text-green-700 font-medium">
                    {allAnnotationsData.length} manual annotation{allAnnotationsData.length !== 1 ? 's' : ''} added
                  </p>
                  <p className="text-xs text-green-600 mt-1">No ML tags detected - using human labels</p>
                </div>
              ) : (
                <p className="text-xs text-gray-500 mb-3">Click a tag to highlight matches in text view</p>
              )}
              <div className="space-y-2">
                {tags.map((tag) => {
                  const colorClass = TAG_COLORS[tag.area || ''] || DEFAULT_HIGHLIGHT_COLOR
                  const isSelected = selectedTag === tag.tag
                  // Use highlights array length, or fall back to pattern_matches count from API
                  const highlightCount = tag.highlights?.length || tag.pattern_matches || 0
                  // For image view, count annotations with this tag name
                  const annotationCount = showTextView ? 0 : (allAnnotationsData || []).filter(
                    ann => ann.tag_name === tag.tag
                  ).length
                  // For text view cycling, only use actual highlights (with positions), not just the count
                  const navigableHighlights = tag.highlights?.length || 0
                  const occurrenceCount = showTextView ? navigableHighlights : annotationCount
                  const feedback = feedbackMap.get(tag.tag)
                  const isReviewed = !!feedback
                  const isConfirmed = feedback?.action === 'confirmed'
                  const isRejected = feedback?.action === 'rejected'

                  return (
                    <div
                      key={tag.tag}
                      className={clsx(
                        'rounded-lg border p-3 transition-all',
                        isSelected ? 'ring-2 ring-blue-500 border-blue-500' : 'hover:border-gray-300',
                        isRejected && 'opacity-50'
                      )}
                    >
                      {/* Tag header - clickable for highlight */}
                      <button
                        onClick={() => setSelectedTag(isSelected ? null : tag.tag)}
                        className="w-full text-left"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className={clsx('w-3 h-3 rounded', colorClass.split(' ')[0])} />
                            <span className={clsx('font-medium text-sm', isRejected && 'line-through')}>{tag.tag}</span>
                          </div>
                          {/* Show status or confidence */}
                          {isReviewed ? (
                            <span className={clsx(
                              'text-xs px-2 py-0.5 rounded flex items-center gap-1',
                              isConfirmed ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                            )}>
                              {isConfirmed ? (
                                <><Check className="h-3 w-3" /> Confirmed</>
                              ) : (
                                'Dismissed'
                              )}
                            </span>
                          ) : (
                            <span className={clsx(
                              'text-xs px-2 py-0.5 rounded',
                              tag.confidence >= 0.7 ? 'bg-green-100 text-green-700' :
                              tag.confidence >= 0.5 ? 'bg-yellow-100 text-yellow-700' :
                              'bg-gray-100 text-gray-700'
                            )}>
                              {(tag.confidence * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                        <div className="mt-2 text-xs text-gray-500 flex justify-between">
                          <span className="flex items-center gap-1">
                            {occurrenceCount > 0 ? (
                              <span className="text-blue-600 font-medium">
                                {occurrenceCount} {showTextView ? 'match' : 'annotation'}{occurrenceCount !== 1 ? 'es' : ''}
                                {isSelected && ' ▼'}
                              </span>
                            ) : (
                              <span>Pattern matches: {highlightCount}</span>
                            )}
                          </span>
                          <span>Semantic: {(tag.semantic_similarity * 100).toFixed(0)}%</span>
                        </div>
                      </button>

                      {/* Feedback buttons */}
                      <div className="mt-3 pt-2 border-t flex items-center justify-between">
                        {isReviewed ? (
                          <button
                            onClick={() => undoFeedback.mutate(tag.tag)}
                            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
                            disabled={undoFeedback.isPending}
                          >
                            <RotateCcw className="h-3 w-3" />
                            Undo
                          </button>
                        ) : (
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => submitFeedback.mutate({ tagName: tag.tag, action: 'confirmed', confidence: tag.confidence })}
                              className="flex items-center gap-1 px-2 py-1 text-xs bg-green-50 text-green-700 rounded hover:bg-green-100 transition-colors"
                              disabled={submitFeedback.isPending}
                            >
                              <Check className="h-3 w-3" />
                              Confirm
                            </button>
                            <button
                              onClick={() => submitFeedback.mutate({ tagName: tag.tag, action: 'rejected', confidence: tag.confidence })}
                              className="flex items-center gap-1 px-2 py-1 text-xs bg-red-50 text-red-700 rounded hover:bg-red-100 transition-colors"
                              disabled={submitFeedback.isPending}
                            >
                              <X className="h-3 w-3" />
                              Dismiss
                            </button>
                          </div>
                        )}
                        {!isReviewed && tag.confidence < 0.7 && (
                          <span className="text-xs text-amber-600">Needs review</span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Review Status Summary - show for completed documents (not failed) */}
          {document.status === 'completed' && !isFailedDoc && feedbackList && feedbackList.length > 0 && (
            <div className="p-4 border-t bg-gray-50">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Review Progress</h3>
              <div className="flex items-center gap-4 text-xs mb-3">
                <span className="text-green-600">
                  {feedbackList.filter(f => f.action === 'confirmed').length} confirmed
                </span>
                <span className="text-red-500">
                  {feedbackList.filter(f => f.action === 'rejected').length} dismissed
                </span>
                <span className="text-gray-500">
                  {tags.length - feedbackList.length} pending
                </span>
              </div>
              {/* Show reviewed items with undo */}
              <div className="space-y-1">
                {feedbackList.filter(f => f.tag_name !== '__no_tags__').map(fb => (
                  <div key={fb.id} className="flex items-center justify-between text-xs py-1 px-2 bg-white rounded border">
                    <div className="flex items-center gap-2">
                      {fb.action === 'confirmed' ? (
                        <Check className="h-3 w-3 text-green-600" />
                      ) : (
                        <X className="h-3 w-3 text-red-500" />
                      )}
                      <span className={fb.action === 'rejected' ? 'text-gray-400 line-through' : ''}>
                        {fb.tag_name}
                      </span>
                    </div>
                    <button
                      onClick={() => undoFeedback.mutate(fb.tag_name)}
                      className="text-gray-400 hover:text-gray-600 p-1"
                      title="Undo"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </button>
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

          {/* Annotations Section - only for image view */}
          {!showTextView && (
            <div className="p-4 border-t">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium">Annotations</h3>
                <span className="text-xs text-gray-500">Page {currentPage}</span>
              </div>
              {annotations.length === 0 ? (
                <div className="text-center py-4 bg-gray-50 rounded-lg">
                  <PenTool className="mx-auto h-6 w-6 text-gray-300 mb-2" />
                  <p className="text-xs text-gray-500">No annotations on this page</p>
                  {!annotationMode && (
                    <button
                      onClick={() => setAnnotationMode(true)}
                      className="mt-2 text-xs text-blue-600 hover:underline"
                    >
                      Enable annotation mode
                    </button>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  {annotations.map((ann) => (
                    <div
                      key={ann.id}
                      className={clsx(
                        'rounded-lg border p-2 cursor-pointer transition-all',
                        selectedAnnotation?.id === ann.id
                          ? 'ring-2 ring-blue-500 border-blue-500'
                          : 'hover:border-gray-300'
                      )}
                      onClick={() => setSelectedAnnotation(
                        selectedAnnotation?.id === ann.id ? null : ann
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={clsx(
                            'w-3 h-3 rounded',
                            ann.color === 'green' && 'bg-green-500',
                            ann.color === 'yellow' && 'bg-yellow-500',
                            ann.color === 'red' && 'bg-red-500'
                          )} />
                          <span className="text-sm font-medium">{ann.tag_name}</span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteAnnotation.mutate(ann.id)
                          }}
                          className="p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded"
                          title="Delete annotation"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        {ann.annotation_type === 'positive' && 'Positive example'}
                        {ann.annotation_type === 'negative' && 'Negative example'}
                        {ann.annotation_type === 'uncertain' && 'Uncertain'}
                      </div>
                    </div>
                  ))}
                </div>
              )}
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

      {/* Tag Assignment Modal */}
      <TagAssignmentModal
        isOpen={showTagModal}
        onClose={() => {
          setShowTagModal(false)
          setPendingBox(null)
        }}
        onSubmit={(data) => {
          if (!pendingBox) return
          createAnnotation.mutate({
            page_number: currentPage,
            x1: pendingBox.x1,
            y1: pendingBox.y1,
            x2: pendingBox.x2,
            y2: pendingBox.y2,
            tag_name: data.tag_name,
            tag_id: data.tag_id,
            area_of_law: data.area_of_law,
            annotation_type: data.annotation_type,
            color: data.color
          })
        }}
        boundingBox={pendingBox}
      />
    </div>
  )
}
