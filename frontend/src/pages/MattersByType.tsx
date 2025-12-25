import { Link, useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  FolderOpen,
  Loader2,
  AlertCircle,
  Trash2,
  ChevronRight,
  CheckCircle,
  Clock,
  XCircle,
  TrendingUp
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface Matter {
  id: string
  name: string
  description: string | null
  matter_type: string | null
  source_path: string | null
  created_at: string
  document_count: number
  pending_count: number
  completed_count: number
  failed_count: number
  average_confidence: number | null
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

function MatterRow({ matter, onDelete }: { matter: Matter; onDelete: () => void }) {
  const color = matterTypeColors[matter.matter_type || 'General'] || matterTypeColors['General']
  const isComplete = matter.completed_count === matter.document_count && matter.document_count > 0
  const hasFailed = matter.failed_count > 0
  const isPending = matter.pending_count > 0

  return (
    <tr className="border-t hover:bg-gray-50">
      <td className="p-4">
        <Link
          to={`/matters/${matter.id}`}
          className="flex items-center gap-3"
        >
          <div
            className="rounded-lg p-2"
            style={{ backgroundColor: `${color}20` }}
          >
            <FolderOpen className="h-5 w-5" style={{ color }} />
          </div>
          <div>
            <p className="font-medium text-gray-900">{matter.name}</p>
            {matter.source_path && (
              <p className="text-xs text-gray-400 font-mono truncate max-w-md">{matter.source_path}</p>
            )}
          </div>
        </Link>
      </td>
      <td className="p-4 text-center">
        <span className="text-lg font-medium">{matter.document_count}</span>
      </td>
      <td className="p-4">
        {isComplete ? (
          <div className="flex items-center gap-1.5 text-green-600">
            <CheckCircle className="h-4 w-4" />
            <span className="text-sm">Complete</span>
          </div>
        ) : hasFailed ? (
          <div className="flex items-center gap-1.5 text-red-500">
            <XCircle className="h-4 w-4" />
            <span className="text-sm">{matter.failed_count} failed</span>
          </div>
        ) : isPending ? (
          <div className="flex items-center gap-1.5 text-gray-500">
            <Clock className="h-4 w-4" />
            <span className="text-sm">{matter.pending_count} pending</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 text-gray-400">
            <Clock className="h-4 w-4" />
            <span className="text-sm">-</span>
          </div>
        )}
      </td>
      <td className="p-4 text-center">
        {matter.average_confidence != null ? (
          <span className={clsx(
            "font-medium",
            matter.average_confidence >= 0.7 && "text-green-600",
            matter.average_confidence >= 0.5 && matter.average_confidence < 0.7 && "text-yellow-600",
            matter.average_confidence < 0.5 && "text-red-600"
          )}>
            {(matter.average_confidence * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        )}
      </td>
      <td className="p-4 text-gray-500 text-sm">
        {new Date(matter.created_at).toLocaleDateString()}
      </td>
      <td className="p-4">
        <div className="flex items-center justify-end gap-2">
          <Link
            to={`/matters/${matter.id}`}
            className="p-2 text-blue-600 hover:bg-blue-50 rounded"
            title="View matter"
          >
            <ChevronRight className="h-4 w-4" />
          </Link>
          <button
            onClick={onDelete}
            className="p-2 text-red-600 hover:bg-red-50 rounded"
            title="Delete matter"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </td>
    </tr>
  )
}

export default function MattersByType() {
  const { type } = useParams<{ type: string }>()
  const decodedType = decodeURIComponent(type || '')
  const queryClient = useQueryClient()
  const color = matterTypeColors[decodedType] || matterTypeColors['General']

  const { data: matters, isLoading, error } = useQuery<Matter[]>({
    queryKey: ['matters', 'type', decodedType],
    queryFn: () => api.get(`/api/matters?matter_type=${encodeURIComponent(decodedType)}`).then(r => r.data),
  })

  const deleteMutation = useMutation({
    mutationFn: (matterId: string) => api.delete(`/api/matters/${matterId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matters'] })
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
    },
  })

  const totalDocs = matters?.reduce((sum, m) => sum + m.document_count, 0) || 0
  const completedDocs = matters?.reduce((sum, m) => sum + m.completed_count, 0) || 0

  // Calculate aggregate confidence for the type
  const confidenceData = matters?.reduce((acc, m) => {
    if (m.average_confidence != null && m.completed_count > 0) {
      acc.total += m.average_confidence * m.completed_count
      acc.count += m.completed_count
    }
    return acc
  }, { total: 0, count: 0 }) || { total: 0, count: 0 }
  const avgConfidence = confidenceData.count > 0 ? confidenceData.total / confidenceData.count : null

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <Link to="/matters" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Matters
        </Link>
        <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
          <h3 className="mt-2 font-medium text-red-800">Failed to load matters</h3>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Link to="/matters" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Matters
        </Link>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className="rounded-xl p-4"
              style={{ backgroundColor: `${color}20` }}
            >
              <FolderOpen className="h-8 w-8" style={{ color }} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{decodedType}</h1>
              <p className="text-gray-500">
                {matters?.length || 0} matters, {totalDocs} documents
              </p>
            </div>
          </div>
          {/* Aggregate Confidence */}
          {avgConfidence != null && completedDocs > 0 && (
            <div className="text-right">
              <div className="flex items-center gap-2">
                <TrendingUp className={clsx(
                  "h-5 w-5",
                  avgConfidence >= 0.7 && "text-green-600",
                  avgConfidence >= 0.5 && avgConfidence < 0.7 && "text-yellow-600",
                  avgConfidence < 0.5 && "text-red-600"
                )} />
                <span className={clsx(
                  "text-2xl font-bold",
                  avgConfidence >= 0.7 && "text-green-600",
                  avgConfidence >= 0.5 && avgConfidence < 0.7 && "text-yellow-600",
                  avgConfidence < 0.5 && "text-red-600"
                )}>
                  {(avgConfidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-gray-500">
                Avg. Confidence ({completedDocs} docs)
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Matters table */}
      {matters && matters.length > 0 ? (
        <div className="rounded-lg bg-white shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left p-4 font-medium text-gray-700">Matter</th>
                <th className="text-center p-4 font-medium text-gray-700 w-24">Docs</th>
                <th className="text-left p-4 font-medium text-gray-700 w-32">Status</th>
                <th className="text-center p-4 font-medium text-gray-700 w-28">Confidence</th>
                <th className="text-left p-4 font-medium text-gray-700 w-28">Imported</th>
                <th className="text-right p-4 font-medium text-gray-700 w-24">Actions</th>
              </tr>
            </thead>
            <tbody>
              {matters.map((matter) => (
                <MatterRow
                  key={matter.id}
                  matter={matter}
                  onDelete={() => {
                    if (confirm(`Delete matter "${matter.name}" and all its documents?`)) {
                      deleteMutation.mutate(matter.id)
                    }
                  }}
                />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="rounded-lg bg-white border-2 border-dashed p-12 text-center">
          <FolderOpen className="mx-auto h-12 w-12 text-gray-300" />
          <h3 className="mt-4 text-lg font-medium text-gray-600">No matters of this type</h3>
          <p className="mt-2 text-gray-400">Import documents to create matters</p>
          <Link
            to="/matters"
            className="mt-4 inline-flex items-center gap-2 text-blue-600 hover:underline"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Matters
          </Link>
        </div>
      )}
    </div>
  )
}
