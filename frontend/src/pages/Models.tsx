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

function getModelSourceInfo(url: string | null, name: string): { href: string; label: string } {
  if (!url) {
    return { href: `https://huggingface.co/${name}`, label: 'View on HuggingFace' }
  }
  if (url.includes('github.com')) {
    return { href: url, label: 'View on GitHub' }
  }
  if (url.includes('ai.google.dev') || url.includes('google.com')) {
    return { href: url, label: 'View Google AI Docs' }
  }
  if (url.includes('openai.com')) {
    return { href: url, label: 'View OpenAI Docs' }
  }
  return { href: url, label: 'View on HuggingFace' }
}

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

interface UnregisteredModel {
  name: string
  type: string
  usage_count: number
  first_seen: string
}

function detectModelSource(url: string): { source: string; name: string } | null {
  try {
    const urlObj = new URL(url)
    const hostname = urlObj.hostname.toLowerCase()
    const path = urlObj.pathname

    if (hostname.includes('huggingface.co')) {
      // Extract org/model from path like /org/model or /org/model/tree/main
      const match = path.match(/^\/([^/]+\/[^/]+)/)
      return match ? { source: 'HuggingFace', name: match[1] } : null
    }
    if (hostname.includes('github.com')) {
      const match = path.match(/^\/([^/]+\/[^/]+)/)
      return match ? { source: 'GitHub', name: match[1] } : null
    }
    if (hostname.includes('ai.google.dev') || hostname.includes('google.com')) {
      return { source: 'Google AI', name: path.split('/').pop() || 'google-model' }
    }
    if (hostname.includes('openai.com')) {
      return { source: 'OpenAI', name: path.split('/').pop() || 'openai-model' }
    }
    // Generic URL
    return { source: 'Custom', name: hostname }
  } catch {
    return null
  }
}

function AddModelModal({ onClose }: { onClose: () => void }) {
  const [modelUrl, setModelUrl] = useState('')
  const [modelName, setModelName] = useState('')
  const [modelType, setModelType] = useState('semantic')
  const [detectedSource, setDetectedSource] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Auto-detect source and name from URL
  const handleUrlChange = (url: string) => {
    setModelUrl(url)
    setError(null)

    if (url.trim()) {
      const detected = detectModelSource(url)
      if (detected) {
        setDetectedSource(detected.source)
        if (!modelName) {
          setModelName(detected.name)
        }
      } else {
        setDetectedSource(null)
      }
    } else {
      setDetectedSource(null)
    }
  }

  const addMutation = useMutation({
    mutationFn: async () => {
      return api.post('/api/models', {
        name: modelName,
        type: modelType,
        huggingface_url: modelUrl || undefined,
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
      <div className="w-full max-w-lg rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Add Model to Registry</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Model URL
            </label>
            <input
              type="text"
              value={modelUrl}
              onChange={(e) => handleUrlChange(e.target.value)}
              placeholder="https://huggingface.co/org/model or https://github.com/org/repo"
              className="w-full rounded-lg border p-2"
            />
            <p className="mt-1 text-xs text-gray-500">
              Paste URL from HuggingFace, GitHub, Google AI, OpenAI, or any source
            </p>
            {detectedSource && (
              <p className="mt-1 text-xs text-green-600 flex items-center gap-1">
                <Check className="h-3 w-3" />
                Detected: {detectedSource}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Model Name / Identifier
            </label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g., sentence-transformers/all-MiniLM-L6-v2"
              className="w-full rounded-lg border p-2"
            />
            <p className="mt-1 text-xs text-gray-500">
              Unique identifier for this model in your registry
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
              <option value="llm">LLM (Language Model)</option>
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
              disabled={addMutation.isPending || !modelName.trim()}
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
              'px-2 py-1 rounded text-xs font-medium',
              model.approved ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
            )}>
              {model.approved ? 'RAI Approved' : 'Pending Review'}
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
        {(() => {
          const sourceInfo = getModelSourceInfo(model.huggingface_url, model.name)
          return (
            <a
              href={sourceInfo.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-blue-600 hover:underline"
            >
              <ExternalLink className="h-4 w-4" />
              {sourceInfo.label}
            </a>
          )
        })()}
        {!model.approved && (
          <button
            onClick={() => onApprove(model.id)}
            className="flex items-center gap-1 text-sm text-green-600 hover:underline"
          >
            <Check className="h-4 w-4" />
            Approve Model
          </button>
        )}
      </div>
    </div>
  )
}

export default function Models() {
  const [showAddModal, setShowAddModal] = useState(false)
  const [filter, setFilter] = useState<'all' | 'approved' | 'pending'>('all')
  const [addingModel, setAddingModel] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const { data: models, isLoading, error } = useQuery<Model[]>({
    queryKey: ['models'],
    queryFn: () => api.get('/api/models').then(r => r.data),
  })

  const { data: unregistered } = useQuery<UnregisteredModel[]>({
    queryKey: ['models', 'unregistered'],
    queryFn: () => api.get('/api/models/check/unregistered').then(r => r.data),
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

  const addUnregisteredMutation = useMutation({
    mutationFn: ({ name, type }: { name: string; type: string }) =>
      api.post('/api/models', { name, type }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({ queryKey: ['models', 'unregistered'] })
      setAddingModel(null)
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

      {/* Unregistered Models Warning */}
      {unregistered && unregistered.length > 0 && (
        <div className="mb-6 rounded-lg bg-amber-50 border border-amber-200 p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-medium text-amber-800">
                {unregistered.length} Unregistered Model{unregistered.length > 1 ? 's' : ''} Detected
              </h3>
              <p className="text-sm text-amber-700 mt-1">
                The following models are being used in document processing but haven't been formally registered:
              </p>
              <div className="mt-3 space-y-2">
                {unregistered.map((model) => (
                  <div key={model.name} className="flex items-center justify-between bg-white rounded-lg px-3 py-2 border border-amber-200">
                    <div>
                      <span className="font-medium text-gray-900">{model.name}</span>
                      <span className="ml-2 text-xs text-gray-500">
                        ({model.type} • {model.usage_count} uses)
                      </span>
                    </div>
                    <button
                      onClick={() => {
                        setAddingModel(model.name)
                        addUnregisteredMutation.mutate({ name: model.name, type: model.type })
                      }}
                      disabled={addingModel === model.name}
                      className="flex items-center gap-1 text-sm text-amber-700 hover:text-amber-900 disabled:opacity-50"
                    >
                      {addingModel === model.name ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Plus className="h-4 w-4" />
                      )}
                      Add to Registry
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

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
