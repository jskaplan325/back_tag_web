import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Box,
  Plus,
  X,
  Check,
  ExternalLink,
  Loader2,
  Download,
  Clock,
  AlertCircle,
  ChevronRight
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface Model {
  id: string
  name: string
  type: string
  huggingface_url: string
  size_gb: number | null
  description: string | null
  downloads: number | null
  approved: boolean
  approved_by: string | null
  approved_at: string | null
  created_at: string
}

interface ModelUsageStats {
  model_id: string
  usage_count: number
  avg_processing_time: number
}

function AddModelModal({ onClose }: { onClose: () => void }) {
  const [modelName, setModelName] = useState('')
  const [modelType, setModelType] = useState('semantic')
  const [error, setError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const addMutation = useMutation({
    mutationFn: async () => {
      return api.post('/api/models', {
        name: modelName,
        type: modelType,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      onClose()
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelName.trim()) {
      setError('Model name is required')
      return
    }
    addMutation.mutate()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Add Model</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              HuggingFace Model Name
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2"
              className="w-full rounded-lg border p-2"
            />
            <p className="mt-1 text-xs text-gray-500">
              Enter the full model path from HuggingFace
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Model Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-lg border p-2"
            >
              <option value="semantic">Semantic (Embeddings)</option>
              <option value="vision">Vision (Image Analysis)</option>
              <option value="ocr">OCR (Text Extraction)</option>
            </select>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={addMutation.isPending}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {addMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              Add Model
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function ModelCard({ model, onApprove }: { model: Model; onApprove: (id: string) => void }) {
  const { data: usage } = useQuery<ModelUsageStats>({
    queryKey: ['models', model.id, 'usage'],
    queryFn: () => api.get(`/api/models/${model.id}/usage`).then(r => r.data),
  })

  return (
    <div className="rounded-lg bg-white p-6 shadow hover:shadow-md transition-shadow">
      <Link to={`/models/${model.id}`} className="block">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={clsx(
              'rounded-lg p-2',
              model.type === 'semantic' && 'bg-blue-100',
              model.type === 'vision' && 'bg-purple-100',
              model.type === 'ocr' && 'bg-green-100'
            )}>
              <Box className={clsx(
                'h-5 w-5',
                model.type === 'semantic' && 'text-blue-600',
                model.type === 'vision' && 'text-purple-600',
                model.type === 'ocr' && 'text-green-600'
              )} />
            </div>
            <div>
              <h3 className="font-medium">{model.name.split('/').pop()}</h3>
              <p className="text-sm text-gray-500">{model.name.split('/')[0]}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={clsx(
              'px-2 py-1 rounded text-xs',
              model.approved ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
            )}>
              {model.approved ? 'Approved' : 'Pending'}
            </span>
            <ChevronRight className="h-4 w-4 text-gray-400" />
          </div>
        </div>

        {model.description && (
          <p className="mt-3 text-sm text-gray-600 line-clamp-2">{model.description}</p>
        )}
      </Link>

      <div className="mt-4 flex items-center gap-4 text-sm text-gray-500">
        {model.size_gb && (
          <span className="flex items-center gap-1">
            <Download className="h-4 w-4" />
            {model.size_gb.toFixed(1)} GB
          </span>
        )}
        {model.downloads && (
          <span>{model.downloads.toLocaleString()} downloads</span>
        )}
        {usage && usage.usage_count > 0 && (
          <span className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            {usage.usage_count} uses
          </span>
        )}
      </div>

      <div className="mt-4 flex items-center gap-3">
        <a
          href={model.huggingface_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-sm text-blue-600 hover:underline"
        >
          <ExternalLink className="h-4 w-4" />
          HuggingFace
        </a>
        {!model.approved && (
          <button
            onClick={() => onApprove(model.id)}
            className="flex items-center gap-1 text-sm text-green-600 hover:underline"
          >
            <Check className="h-4 w-4" />
            Approve
          </button>
        )}
      </div>
    </div>
  )
}

export default function Models() {
  const [showAddModal, setShowAddModal] = useState(false)
  const [filter, setFilter] = useState<'all' | 'approved' | 'pending'>('all')
  const queryClient = useQueryClient()

  const { data: models, isLoading, error } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: () => api.get('/api/models').then(r => r.data),
  })

  // Debug: log any errors
  if (error) {
    console.error('Failed to fetch models:', error)
  }

  const approveMutation = useMutation({
    mutationFn: (id: string) => api.patch(`/api/models/${id}`, { approved: true }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })

  const filteredModels = models?.filter(m => {
    if (filter === 'approved') return m.approved
    if (filter === 'pending') return !m.approved
    return true
  })

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Model Registry</h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
        >
          <Plus className="h-4 w-4" />
          Add Model
        </button>
      </div>

      {/* Filters */}
      <div className="mb-6 flex gap-2">
        {(['all', 'approved', 'pending'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={clsx(
              'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              filter === f
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            )}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      ) : error ? (
        <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
          <h3 className="mt-2 font-medium text-red-800">Failed to load models</h3>
          <p className="mt-1 text-sm text-red-600">Make sure the backend is running on port 8000</p>
        </div>
      ) : filteredModels && filteredModels.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              onApprove={(id) => approveMutation.mutate(id)}
            />
          ))}
        </div>
      ) : (
        <div className="rounded-lg bg-white p-12 shadow text-center">
          <Box className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No models registered</h3>
          <p className="mt-2 text-gray-500">Add a model from HuggingFace to get started</p>
          <button
            onClick={() => setShowAddModal(true)}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Add Model
          </button>
        </div>
      )}

      {showAddModal && <AddModelModal onClose={() => setShowAddModal(false)} />}
    </div>
  )
}
