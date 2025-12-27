import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  AlertTriangle,
  FileQuestion,
  TrendingDown,
  Loader2,
  CheckCircle,
  Filter,
  X,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Search
} from 'lucide-react'
import api from '../api'

interface ReviewQueueItem {
  id: string
  type: 'no_tags' | 'low_confidence' | 'failed'
  document_id: string
  document_name: string
  matter_id: string | null
  matter_name: string | null
  tag_name: string | null
  confidence: number | null
  error_message: string | null
  priority: number
}

interface ReviewQueueResponse {
  total_items: number
  items: ReviewQueueItem[]
  by_type: {
    failed: number
    no_tags: number
    low_confidence: number
  }
}

interface Matter {
  id: string
  name: string
  matter_type: string
}

function TypeBadge({ type }: { type: string }) {
  const config: Record<string, { bg: string; icon: React.ReactNode; label: string }> = {
    failed: { bg: 'bg-red-100 text-red-700', icon: <AlertTriangle className="h-3 w-3" />, label: 'Failed' },
    no_tags: { bg: 'bg-yellow-100 text-yellow-700', icon: <FileQuestion className="h-3 w-3" />, label: 'No Tags' },
    low_confidence: { bg: 'bg-orange-100 text-orange-700', icon: <TrendingDown className="h-3 w-3" />, label: 'Low Conf' },
  }
  const { bg, icon, label } = config[type] || { bg: 'bg-gray-100 text-gray-700', icon: null, label: type }
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded flex items-center gap-1 whitespace-nowrap ${bg}`}>
      {icon}
      {label}
    </span>
  )
}

export default function ReviewQueue() {
  const queryClient = useQueryClient()
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [matterFilter, setMatterFilter] = useState<string>('')
  const [matterSearch, setMatterSearch] = useState<string>('')
  const [actionInProgress, setActionInProgress] = useState<Set<string>>(new Set())

  // Fetch matters for filter
  const { data: matters } = useQuery<Matter[]>({
    queryKey: ['matters'],
    queryFn: () => api.get('/api/matters').then(r => r.data),
  })

  // Filter matters by search
  const filteredMatters = matters?.filter(m =>
    m.name.toLowerCase().includes(matterSearch.toLowerCase()) ||
    m.matter_type.toLowerCase().includes(matterSearch.toLowerCase())
  ) || []

  // Fetch review queue
  const { data: reviewData, isLoading } = useQuery<ReviewQueueResponse>({
    queryKey: ['review-queue', matterFilter],
    queryFn: () => {
      const params = new URLSearchParams()
      if (matterFilter) params.append('matter_id', matterFilter)
      const queryString = params.toString()
      return api.get(`/api/review-queue${queryString ? '?' + queryString : ''}`).then(r => r.data)
    },
    refetchInterval: 5000
  })

  // Filter items by type
  const filteredItems = typeFilter
    ? reviewData?.items.filter(item => item.type === typeFilter)
    : reviewData?.items

  // Submit feedback mutation
  const submitFeedback = useMutation({
    mutationFn: ({ documentId, tagName, action, confidence }: {
      documentId: string;
      tagName: string;
      action: string;
      confidence: number;
    }) => api.post(`/api/documents/${documentId}/feedback`, {
      tag_name: tagName,
      action: action,
      original_confidence: confidence
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['review-queue'] })
    }
  })

  const handleAction = async (item: ReviewQueueItem, action: 'confirmed' | 'rejected') => {
    const key = item.id
    setActionInProgress(prev => new Set(prev).add(key))

    // Determine the tag name based on item type
    let tagName = item.tag_name || ''
    if (item.type === 'failed') tagName = '__failed_reviewed__'
    else if (item.type === 'no_tags') tagName = '__no_tags__'

    try {
      await submitFeedback.mutateAsync({
        documentId: item.document_id,
        tagName: tagName,
        action: action,
        confidence: item.confidence || 0
      })
    } finally {
      setActionInProgress(prev => { const n = new Set(prev); n.delete(key); return n })
    }
  }

  const stats = reviewData?.by_type || { failed: 0, no_tags: 0, low_confidence: 0 }
  const total = reviewData?.total_items || 0

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-900">Review Queue</h1>
        <p className="text-sm text-gray-500">Items needing human review - confirm or dismiss to improve ML accuracy</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <button
          onClick={() => setTypeFilter('')}
          className={`p-3 rounded-lg border text-left ${!typeFilter ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
        >
          <div className="flex items-center gap-2">
            <CheckCircle className="h-4 w-4 text-gray-500" />
            <span className="text-xs text-gray-500">Total</span>
          </div>
          <p className="text-xl font-semibold mt-1">{total}</p>
        </button>

        <button
          onClick={() => setTypeFilter('failed')}
          className={`p-3 rounded-lg border text-left ${typeFilter === 'failed' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-500" />
            <span className="text-xs text-gray-500">Failed</span>
          </div>
          <p className="text-xl font-semibold mt-1">{stats.failed}</p>
        </button>

        <button
          onClick={() => setTypeFilter('no_tags')}
          className={`p-3 rounded-lg border text-left ${typeFilter === 'no_tags' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
        >
          <div className="flex items-center gap-2">
            <FileQuestion className="h-4 w-4 text-yellow-500" />
            <span className="text-xs text-gray-500">No Tags</span>
          </div>
          <p className="text-xl font-semibold mt-1">{stats.no_tags}</p>
        </button>

        <button
          onClick={() => setTypeFilter('low_confidence')}
          className={`p-3 rounded-lg border text-left ${typeFilter === 'low_confidence' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white hover:bg-gray-50'}`}
        >
          <div className="flex items-center gap-2">
            <TrendingDown className="h-4 w-4 text-orange-500" />
            <span className="text-xs text-gray-500">Low Confidence</span>
          </div>
          <p className="text-xl font-semibold mt-1">{stats.low_confidence}</p>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="flex items-center gap-3 mb-4">
        <Filter className="h-4 w-4 text-gray-400" />

        {/* Matter search/filter */}
        <div className="relative">
          <div className="flex items-center border rounded-lg bg-white">
            <Search className="h-4 w-4 text-gray-400 ml-2" />
            <input
              type="text"
              placeholder="Search matters..."
              value={matterSearch}
              onChange={(e) => setMatterSearch(e.target.value)}
              className="text-sm px-2 py-1.5 w-48 focus:outline-none"
            />
          </div>
          {matterSearch && filteredMatters.length > 0 && (
            <div className="absolute z-10 mt-1 w-64 bg-white border rounded-lg shadow-lg max-h-48 overflow-auto">
              {filteredMatters.slice(0, 10).map(matter => (
                <button
                  key={matter.id}
                  onClick={() => {
                    setMatterFilter(matter.id)
                    setMatterSearch(matter.name)
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-gray-50 text-sm"
                >
                  <div className="font-medium">{matter.name}</div>
                  <div className="text-xs text-gray-500">{matter.matter_type}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {(typeFilter || matterFilter) && (
          <button
            onClick={() => { setTypeFilter(''); setMatterFilter(''); setMatterSearch('') }}
            className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
          >
            <X className="h-3 w-3" /> Clear filters
          </button>
        )}
      </div>

      {/* Queue Table */}
      <div className="bg-white rounded-lg shadow">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
          </div>
        ) : !filteredItems?.length ? (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400">
            <CheckCircle className="h-12 w-12 mb-3 text-green-400" />
            <p className="text-lg font-medium text-gray-600">All clear!</p>
            <p className="text-sm">No items need review</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Type</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Document</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Tag / Issue</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Confidence</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Matter</th>
                  <th className="px-4 py-3 text-right font-medium text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredItems?.map((item) => (
                  <tr key={item.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">
                      <TypeBadge type={item.type} />
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        to={`/documents/${item.document_id}`}
                        className="font-medium text-blue-600 hover:underline flex items-center gap-1"
                      >
                        {item.document_name.length > 35 ? item.document_name.slice(0, 35) + '...' : item.document_name}
                        <ExternalLink className="h-3 w-3" />
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      {item.type === 'low_confidence' && item.tag_name ? (
                        <span className="px-2 py-0.5 bg-gray-100 rounded text-xs font-medium">
                          {item.tag_name}
                        </span>
                      ) : item.type === 'failed' ? (
                        <span className="text-red-600 text-xs" title={item.error_message || ''}>
                          {item.error_message && item.error_message.length > 40
                            ? item.error_message.slice(0, 40) + '...'
                            : item.error_message || 'Processing failed'}
                        </span>
                      ) : (
                        <span className="text-gray-500 text-xs">No tags detected</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {item.confidence !== null ? (
                        <span className={`font-medium ${
                          item.confidence >= 0.7 ? 'text-green-600' :
                          item.confidence >= 0.5 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(item.confidence * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {item.matter_name ? (
                        <Link to={`/matters/${item.matter_id}`} className="text-gray-600 hover:text-blue-600 text-xs">
                          {item.matter_name.length > 20 ? item.matter_name.slice(0, 20) + '...' : item.matter_name}
                        </Link>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => handleAction(item, 'confirmed')}
                          disabled={actionInProgress.has(item.id)}
                          className="p-1.5 bg-green-100 text-green-700 rounded hover:bg-green-200 disabled:opacity-50"
                          title={item.type === 'low_confidence' ? 'Confirm tag is correct' : 'Confirm OK'}
                        >
                          {actionInProgress.has(item.id) ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <ThumbsUp className="h-3.5 w-3.5" />
                          )}
                        </button>
                        <button
                          onClick={() => handleAction(item, 'rejected')}
                          disabled={actionInProgress.has(item.id)}
                          className="p-1.5 bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50"
                          title={item.type === 'low_confidence' ? 'Reject tag as incorrect' : 'Dismiss'}
                        >
                          {actionInProgress.has(item.id) ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <ThumbsDown className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Help text */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700">
        <strong>How it works:</strong> Review items to improve ML accuracy.
        <span className="inline-flex items-center gap-1 mx-1"><ThumbsUp className="h-3 w-3" /> Confirm</span>
        marks items as correct (counts as 100% confidence).
        <span className="inline-flex items-center gap-1 mx-1"><ThumbsDown className="h-3 w-3" /> Dismiss</span>
        marks items as incorrect (excluded from confidence calculations).
      </div>
    </div>
  )
}
