import { Link, useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  FolderOpen,
  FileText,
  Loader2,
  AlertCircle,
  Play,
  Trash2,
  CheckCircle,
  Clock,
  XCircle,
  Tag
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface DocumentInMatter {
  id: string
  filename: string
  status: string
  uploaded_at: string
  file_size_bytes: number | null
  word_count?: number | null
}

interface MatterTag {
  tag: string
  average_confidence: number
  document_count: number
}

interface MatterTagsResponse {
  matter_id: string
  total_documents: number
  processed_documents: number
  top_tags: MatterTag[]
}

interface MatterDetail {
  id: string
  name: string
  description: string | null
  matter_type: string | null
  source_path: string | null
  created_at: string
  document_count: number
  documents: DocumentInMatter[]
}

const matterTypeColors: Record<string, string> = {
  'Funds': '#3b82f6',
  'M&A': '#8b5cf6',
  'Leveraged Finance': '#10b981',
  'Real Estate': '#f59e0b',
  'Intellectual Property': '#ec4899',
  'Employment': '#06b6d4',
  'Tax': '#84cc16',
  'Litigation': '#ef4444',
  'General': '#6b7280'
}

const statusConfig: Record<string, { color: string; icon: React.ElementType; label: string }> = {
  uploaded: { color: 'text-gray-500', icon: Clock, label: 'Pending' },
  processing: { color: 'text-blue-500', icon: Loader2, label: 'Processing' },
  completed: { color: 'text-green-500', icon: CheckCircle, label: 'Completed' },
  failed: { color: 'text-red-500', icon: XCircle, label: 'Failed' },
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

function DocumentRow({ doc, matterId }: { doc: DocumentInMatter; matterId: string }) {
  const queryClient = useQueryClient()
  const status = statusConfig[doc.status] || statusConfig.uploaded
  const StatusIcon = status.icon

  const processMutation = useMutation({
    mutationFn: () => api.post(`/api/documents/${doc.id}/process`, {
      semantic_model: 'pile-of-law/legalbert-large-1.7M-2',
      enable_vision: false
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matter', matterId] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => api.delete(`/api/documents/${doc.id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matter', matterId] })
    },
  })

  return (
    <tr className="border-t hover:bg-gray-50">
      <td className="p-3">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-gray-400" />
          <Link
            to={`/documents/${doc.id}`}
            className="text-blue-600 hover:underline"
          >
            {doc.filename}
          </Link>
        </div>
      </td>
      <td className="p-3">
        <div className={clsx('flex items-center gap-1.5', status.color)}>
          <StatusIcon className={clsx('h-4 w-4', doc.status === 'processing' && 'animate-spin')} />
          <span className="text-sm">{status.label}</span>
        </div>
      </td>
      <td className="p-3 text-sm text-gray-500">
        {doc.file_size_bytes ? formatBytes(doc.file_size_bytes) : '-'}
      </td>
      <td className="p-3 text-sm text-gray-500">
        {doc.word_count?.toLocaleString() || '-'}
      </td>
      <td className="p-3">
        <div className="flex items-center gap-1">
          {doc.status === 'uploaded' && (
            <button
              onClick={() => processMutation.mutate()}
              disabled={processMutation.isPending}
              className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded"
              title="Process document"
            >
              {processMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </button>
          )}
          <button
            onClick={() => {
              if (confirm(`Delete "${doc.filename}"?`)) {
                deleteMutation.mutate()
              }
            }}
            className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
            title="Delete document"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  )
}

export default function MatterDetailPage() {
  const { id } = useParams<{ id: string }>()
  const queryClient = useQueryClient()

  const { data: matter, isLoading, error } = useQuery<MatterDetail>({
    queryKey: ['matter', id],
    queryFn: () => api.get(`/api/matters/${id}`).then(r => r.data),
  })

  const { data: tagsData } = useQuery<MatterTagsResponse>({
    queryKey: ['matter-tags', id],
    queryFn: () => api.get(`/api/matters/${id}/tags`).then(r => r.data),
    enabled: !!id,
  })

  const processAllMutation = useMutation({
    mutationFn: async () => {
      if (!matter) return
      const pendingDocs = matter.documents.filter(d => d.status === 'uploaded')
      for (const doc of pendingDocs) {
        await api.post(`/api/documents/${doc.id}/process`, {
          semantic_model: 'pile-of-law/legalbert-large-1.7M-2',
          enable_vision: false
        })
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matter', id] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (error || !matter) {
    return (
      <div className="p-8">
        <Link to="/matters" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Matters
        </Link>
        <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
          <h3 className="mt-2 font-medium text-red-800">Matter not found</h3>
        </div>
      </div>
    )
  }

  const color = matterTypeColors[matter.matter_type || 'General'] || matterTypeColors['General']
  const pendingCount = matter.documents.filter(d => d.status === 'uploaded').length
  const completedCount = matter.documents.filter(d => d.status === 'completed').length

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Link to="/matters" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Matters
        </Link>

        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div
              className="rounded-xl p-4"
              style={{ backgroundColor: `${color}20` }}
            >
              <FolderOpen className="h-8 w-8" style={{ color }} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{matter.name}</h1>
              <div className="flex items-center gap-3 mt-1">
                {matter.matter_type && (
                  <span
                    className="px-2 py-0.5 rounded text-sm"
                    style={{ backgroundColor: `${color}20`, color }}
                  >
                    {matter.matter_type}
                  </span>
                )}
                <span className="text-sm text-gray-400">{matter.document_count} documents</span>
              </div>
              {matter.source_path && (
                <p className="text-xs text-gray-400 font-mono mt-2">{matter.source_path}</p>
              )}
            </div>
          </div>

          {pendingCount > 0 && (
            <button
              onClick={() => processAllMutation.mutate()}
              disabled={processAllMutation.isPending}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {processAllMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Process All ({pendingCount})
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="rounded-lg bg-white p-4 shadow">
          <p className="text-2xl font-bold">{matter.document_count}</p>
          <p className="text-sm text-gray-500">Total Documents</p>
        </div>
        <div className="rounded-lg bg-white p-4 shadow">
          <p className="text-2xl font-bold text-green-600">{completedCount}</p>
          <p className="text-sm text-gray-500">Processed</p>
        </div>
        <div className="rounded-lg bg-white p-4 shadow">
          <p className="text-2xl font-bold text-gray-500">{pendingCount}</p>
          <p className="text-sm text-gray-500">Pending</p>
        </div>
      </div>

      {/* Top Tags */}
      {tagsData && tagsData.top_tags.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Tag className="h-4 w-4 text-gray-500" />
            <h2 className="text-sm font-medium text-gray-700">Top Tags</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {tagsData.top_tags.map((tag) => (
              <div
                key={tag.tag}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-200"
              >
                <span className="text-sm font-medium text-blue-700">{tag.tag}</span>
                <span className="text-xs text-blue-500">
                  {Math.round(tag.average_confidence * 100)}%
                </span>
                <span className="text-xs text-gray-400">
                  ({tag.document_count} docs)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Documents table */}
      {matter.documents.length > 0 ? (
        <div className="rounded-lg bg-white shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left p-3 font-medium text-gray-700">Document</th>
                <th className="text-left p-3 font-medium text-gray-700">Status</th>
                <th className="text-left p-3 font-medium text-gray-700">Size</th>
                <th className="text-left p-3 font-medium text-gray-700">Words</th>
                <th className="text-left p-3 font-medium text-gray-700 w-24">Actions</th>
              </tr>
            </thead>
            <tbody>
              {matter.documents.map((doc) => (
                <DocumentRow key={doc.id} doc={doc} matterId={matter.id} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-lg bg-white border-2 border-dashed p-12 text-center">
          <FileText className="mx-auto h-12 w-12 text-gray-300" />
          <h3 className="mt-4 text-lg font-medium text-gray-600">No documents</h3>
          <p className="mt-2 text-gray-400">This matter has no documents yet</p>
        </div>
      )}
    </div>
  )
}
