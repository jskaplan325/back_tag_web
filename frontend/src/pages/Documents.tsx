import { useState, useCallback, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import {
  FileText,
  Upload,
  X,
  Play,
  Trash2,
  Eye,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  EyeOff
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface MatterBrief {
  id: string
  name: string
  matter_type: string | null
}

interface Document {
  id: string
  filename: string
  uploaded_at: string
  file_size_bytes: number
  page_count: number | null
  word_count: number | null
  status: string
  error_message: string | null
  matter_id: string | null
  matter: MatterBrief | null
  average_confidence: number | null
}

type SortField = 'filename' | 'confidence' | 'uploaded_at'
type SortDirection = 'asc' | 'desc'

function UploadModal({ onClose }: { onClose: () => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const queryClient = useQueryClient()

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      return api.post('/api/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      onClose()
    },
  })

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0])
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Upload Document</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <div
          className={clsx(
            'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300',
            file && 'border-green-500 bg-green-50'
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {file ? (
            <div className="flex items-center justify-center gap-3">
              <FileText className="h-8 w-8 text-green-600" />
              <div>
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={() => setFile(null)}
                className="ml-4 text-gray-400 hover:text-gray-600"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-gray-600">
                Drag and drop a PDF file here, or
              </p>
              <label className="mt-2 inline-block cursor-pointer text-blue-600 hover:underline">
                browse
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleChange}
                  className="hidden"
                />
              </label>
            </>
          )}
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={() => file && uploadMutation.mutate(file)}
            disabled={!file || uploadMutation.isPending}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {uploadMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Upload className="h-4 w-4" />
            )}
            Upload
          </button>
        </div>
      </div>
    </div>
  )
}

function ProcessModal({ document, onClose }: { document: Document; onClose: () => void }) {
  const [model, setModel] = useState('pile-of-law/legalbert-large-1.7M-2')
  const [enableVision, setEnableVision] = useState(false)
  const queryClient = useQueryClient()

  const processMutation = useMutation({
    mutationFn: async () => {
      return api.post(`/api/documents/${document.id}/process`, {
        semantic_model: model,
        enable_vision: enableVision,
        vision_model: 'microsoft/Florence-2-base',
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      onClose()
    },
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Process Document</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <p className="text-gray-600 mb-4">{document.filename}</p>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Semantic Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border p-2"
            >
              <option value="pile-of-law/legalbert-large-1.7M-2">LegalBERT (Recommended)</option>
              <option value="sentence-transformers/all-MiniLM-L12-v2">all-MiniLM (Fast)</option>
              <option value="sentence-transformers/all-mpnet-base-v2">all-mpnet (Balanced)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="enableVision"
              checked={enableVision}
              onChange={(e) => setEnableVision(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="enableVision" className="text-sm">
              Enable Florence-2 vision analysis (charts/tables)
            </label>
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={() => processMutation.mutate()}
            disabled={processMutation.isPending}
            className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-white hover:bg-green-700 disabled:opacity-50"
          >
            {processMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            Process
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Documents() {
  const [searchParams, setSearchParams] = useSearchParams()
  const showUpload = searchParams.get('upload') === 'true'
  const [processDoc, setProcessDoc] = useState<Document | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortField, setSortField] = useState<SortField>('uploaded_at')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  const queryClient = useQueryClient()

  const { data: documents, isLoading } = useQuery<Document[]>({
    queryKey: ['documents', searchQuery],
    queryFn: () => api.get(`/api/documents${searchQuery ? `?search=${encodeURIComponent(searchQuery)}` : ''}`).then(r => r.data),
    refetchInterval: 5000, // Poll for status updates
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.delete(`/api/documents/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const closeUpload = () => {
    searchParams.delete('upload')
    setSearchParams(searchParams)
  }

  // Sort documents
  const sortedDocuments = useMemo(() => {
    if (!documents) return []
    return [...documents].sort((a, b) => {
      let comparison = 0
      switch (sortField) {
        case 'filename':
          comparison = a.filename.localeCompare(b.filename)
          break
        case 'confidence':
          comparison = (a.average_confidence ?? 0) - (b.average_confidence ?? 0)
          break
        case 'uploaded_at':
          comparison = new Date(a.uploaded_at).getTime() - new Date(b.uploaded_at).getTime()
          break
      }
      return sortDirection === 'asc' ? comparison : -comparison
    })
  }, [documents, sortField, sortDirection])

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection(field === 'confidence' ? 'asc' : 'desc') // Low confidence first to review
    }
  }

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ArrowUpDown className="h-3 w-3 opacity-50" />
    return sortDirection === 'asc' ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search documents or matters..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg w-64 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <button
            onClick={() => setSearchParams({ upload: 'true' })}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Upload className="h-4 w-4" />
            Upload
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      ) : sortedDocuments.length > 0 ? (
        <div className="rounded-lg bg-white shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th
                  className="px-6 py-3 text-left text-sm font-medium text-gray-500 cursor-pointer hover:bg-gray-100"
                  onClick={() => toggleSort('filename')}
                >
                  <div className="flex items-center gap-1">
                    Document
                    <SortIcon field="filename" />
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Matter</th>
                <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Status</th>
                <th
                  className="px-6 py-3 text-left text-sm font-medium text-gray-500 cursor-pointer hover:bg-gray-100"
                  onClick={() => toggleSort('confidence')}
                >
                  <div className="flex items-center gap-1">
                    Confidence
                    <SortIcon field="confidence" />
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-sm font-medium text-gray-500 cursor-pointer hover:bg-gray-100"
                  onClick={() => toggleSort('uploaded_at')}
                >
                  <div className="flex items-center gap-1">
                    Uploaded
                    <SortIcon field="uploaded_at" />
                  </div>
                </th>
                <th className="px-6 py-3 text-right text-sm font-medium text-gray-500">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {sortedDocuments.map((doc) => (
                <tr key={doc.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-gray-400" />
                      <span className="font-medium">{doc.filename}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {doc.matter ? (
                      <Link
                        to={`/matters/${doc.matter.id}`}
                        className="text-blue-600 hover:underline text-sm"
                      >
                        {doc.matter.name}
                      </Link>
                    ) : (
                      <span className="text-gray-400 text-sm">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <span className={clsx(
                      'inline-flex items-center gap-1 px-2 py-1 rounded text-sm',
                      doc.status === 'completed' && 'bg-green-100 text-green-700',
                      doc.status === 'failed' && 'bg-red-100 text-red-700',
                      doc.status === 'processing' && 'bg-yellow-100 text-yellow-700',
                      doc.status === 'uploaded' && 'bg-gray-100 text-gray-700',
                      doc.status === 'ignored' && 'bg-yellow-100 text-yellow-700'
                    )}>
                      {doc.status === 'completed' && <CheckCircle className="h-3 w-3" />}
                      {doc.status === 'failed' && <XCircle className="h-3 w-3" />}
                      {doc.status === 'processing' && <Loader2 className="h-3 w-3 animate-spin" />}
                      {doc.status === 'uploaded' && <Clock className="h-3 w-3" />}
                      {doc.status === 'ignored' && <EyeOff className="h-3 w-3" />}
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    {doc.average_confidence != null ? (
                      <span className={clsx(
                        'font-medium',
                        doc.average_confidence >= 0.7 && 'text-green-600',
                        doc.average_confidence >= 0.5 && doc.average_confidence < 0.7 && 'text-yellow-600',
                        doc.average_confidence < 0.5 && 'text-red-600'
                      )}>
                        {(doc.average_confidence * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-gray-500">
                    {new Date(doc.uploaded_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center justify-end gap-2">
                      {doc.status === 'uploaded' && (
                        <button
                          onClick={() => setProcessDoc(doc)}
                          className="p-2 text-green-600 hover:bg-green-50 rounded"
                          title="Process"
                        >
                          <Play className="h-4 w-4" />
                        </button>
                      )}
                      {doc.status === 'completed' && (
                        <Link
                          to={`/documents/${doc.id}`}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded"
                          title="View"
                        >
                          <Eye className="h-4 w-4" />
                        </Link>
                      )}
                      <button
                        onClick={() => {
                          if (confirm('Delete this document?')) {
                            deleteMutation.mutate(doc.id)
                          }
                        }}
                        className="p-2 text-red-600 hover:bg-red-50 rounded"
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : searchQuery ? (
        <div className="rounded-lg bg-white p-12 shadow text-center">
          <Search className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No results found</h3>
          <p className="mt-2 text-gray-500">No documents match "{searchQuery}"</p>
          <button
            onClick={() => setSearchQuery('')}
            className="mt-4 text-blue-600 hover:underline"
          >
            Clear search
          </button>
        </div>
      ) : (
        <div className="rounded-lg bg-white p-12 shadow text-center">
          <FileText className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No documents yet</h3>
          <p className="mt-2 text-gray-500">Upload a PDF to get started</p>
          <button
            onClick={() => setSearchParams({ upload: 'true' })}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Upload className="h-4 w-4" />
            Upload Document
          </button>
        </div>
      )}

      {showUpload && <UploadModal onClose={closeUpload} />}
      {processDoc && <ProcessModal document={processDoc} onClose={() => setProcessDoc(null)} />}
    </div>
  )
}
