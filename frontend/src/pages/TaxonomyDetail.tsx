import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  Plus,
  X,
  Edit2,
  Trash2,
  Tag,
  Loader2,
  AlertCircle,
  Save,
  BarChart2
} from 'lucide-react'
import api from '../api'

interface TagResponse {
  id: string
  area_of_law_id: string
  name: string
  description: string | null
  patterns: string[]
  semantic_examples: string[]
  threshold: number
  usage_count: number
  avg_confidence: number
}

interface AreaOfLaw {
  id: string
  name: string
  description: string | null
  color: string
  icon: string
  tag_count: number
  tags: TagResponse[]
}

interface TagFormData {
  name: string
  description: string
  patterns: string
  semantic_examples: string
  threshold: number
}

function TagModal({
  aolId,
  tag,
  onClose
}: {
  aolId: string
  tag?: TagResponse
  onClose: () => void
}) {
  const [formData, setFormData] = useState<TagFormData>({
    name: tag?.name || '',
    description: tag?.description || '',
    patterns: tag?.patterns.join('\n') || '',
    semantic_examples: tag?.semantic_examples.join('\n') || '',
    threshold: tag?.threshold || 0.45,
  })
  const [error, setError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const isEditing = !!tag

  const mutation = useMutation({
    mutationFn: async () => {
      const data = {
        name: formData.name,
        description: formData.description || null,
        patterns: formData.patterns.split('\n').filter(p => p.trim()),
        semantic_examples: formData.semantic_examples.split('\n').filter(s => s.trim()),
        threshold: formData.threshold,
      }
      if (isEditing) {
        return api.patch(`/api/taxonomy/${aolId}/tags/${tag.id}`, data)
      }
      return api.post(`/api/taxonomy/${aolId}/tags`, data)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['taxonomy', aolId] })
      onClose()
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 overflow-auto py-8">
      <div className="w-full max-w-2xl rounded-lg bg-white p-6 shadow-xl mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">
            {isEditing ? 'Edit Tag' : 'Add New Tag'}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={(e) => { e.preventDefault(); mutation.mutate() }} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Tag Name *</label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="e.g., Intellectual Property"
                className="w-full rounded-lg border p-2"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Confidence Threshold</label>
              <input
                type="number"
                value={formData.threshold}
                onChange={(e) => setFormData({ ...formData, threshold: parseFloat(e.target.value) })}
                min="0"
                max="1"
                step="0.05"
                className="w-full rounded-lg border p-2"
              />
              <p className="text-xs text-gray-500 mt-1">0.0 - 1.0 (default: 0.45)</p>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <input
              type="text"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Brief description of what this tag identifies"
              className="w-full rounded-lg border p-2"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Regex Patterns
              <span className="text-gray-400 font-normal ml-2">(one per line)</span>
            </label>
            <textarea
              value={formData.patterns}
              onChange={(e) => setFormData({ ...formData, patterns: e.target.value })}
              placeholder="merger\s+agreement&#10;acquisition\s+agreement&#10;business\s+combination"
              className="w-full rounded-lg border p-2 font-mono text-sm"
              rows={4}
            />
            <p className="text-xs text-gray-500 mt-1">
              Use regex patterns to match specific terms. Case-insensitive.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Semantic Examples
              <span className="text-gray-400 font-normal ml-2">(one per line)</span>
            </label>
            <textarea
              value={formData.semantic_examples}
              onChange={(e) => setFormData({ ...formData, semantic_examples: e.target.value })}
              placeholder="agreement and plan of merger between companies&#10;acquisition agreement for purchase of target&#10;definitive merger agreement terms"
              className="w-full rounded-lg border p-2 text-sm"
              rows={4}
            />
            <p className="text-xs text-gray-500 mt-1">
              Example phrases for semantic similarity matching with LegalBERT.
            </p>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          <div className="flex justify-end gap-3 pt-4">
            <button type="button" onClick={onClose} className="px-4 py-2 text-gray-600">
              Cancel
            </button>
            <button
              type="submit"
              disabled={mutation.isPending || !formData.name}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {mutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : isEditing ? (
                <Save className="h-4 w-4" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              {isEditing ? 'Save Changes' : 'Add Tag'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function TagCard({
  tag,
  color,
  onEdit,
  onDelete
}: {
  tag: TagResponse
  color: string
  onEdit: () => void
  onDelete: () => void
}) {
  return (
    <div className="rounded-lg border bg-white p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className="rounded-lg p-2"
            style={{ backgroundColor: `${color}20` }}
          >
            <Tag className="h-4 w-4" style={{ color }} />
          </div>
          <div>
            <h3 className="font-medium">{tag.name}</h3>
            {tag.description && (
              <p className="text-sm text-gray-500">{tag.description}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={onEdit}
            className="p-1.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded"
          >
            <Edit2 className="h-4 w-4" />
          </button>
          <button
            onClick={onDelete}
            className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-3 flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1 text-gray-500">
          <BarChart2 className="h-3.5 w-3.5" />
          <span>{tag.usage_count} uses</span>
        </div>
        {tag.avg_confidence > 0 && (
          <div className="text-gray-500">
            {(tag.avg_confidence * 100).toFixed(0)}% avg confidence
          </div>
        )}
        <div className="text-gray-400">
          {tag.patterns.length} patterns, {tag.semantic_examples.length} examples
        </div>
      </div>

      {/* Patterns preview */}
      {tag.patterns.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {tag.patterns.slice(0, 3).map((pattern, idx) => (
            <code
              key={idx}
              className="px-1.5 py-0.5 bg-gray-100 text-gray-600 text-xs rounded font-mono"
            >
              {pattern.length > 25 ? pattern.slice(0, 25) + '...' : pattern}
            </code>
          ))}
          {tag.patterns.length > 3 && (
            <span className="px-1.5 py-0.5 text-gray-400 text-xs">
              +{tag.patterns.length - 3} more
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export default function TaxonomyDetail() {
  const { id } = useParams<{ id: string }>()
  const [showTagModal, setShowTagModal] = useState(false)
  const [editingTag, setEditingTag] = useState<TagResponse | null>(null)
  const queryClient = useQueryClient()

  const { data: area, isLoading, error } = useQuery<AreaOfLaw>({
    queryKey: ['taxonomy', id],
    queryFn: () => api.get(`/api/taxonomy/${id}`).then(r => r.data),
  })

  const deleteMutation = useMutation({
    mutationFn: (tagId: string) => api.delete(`/api/taxonomy/${id}/tags/${tagId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['taxonomy', id] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (error || !area) {
    return (
      <div className="p-8">
        <Link to="/taxonomy" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Taxonomy
        </Link>
        <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
          <h3 className="mt-2 font-medium text-red-800">Area of Law not found</h3>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Link to="/taxonomy" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Taxonomy
        </Link>

        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div
              className="rounded-xl p-4"
              style={{ backgroundColor: `${area.color}20` }}
            >
              <Tag className="h-8 w-8" style={{ color: area.color }} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{area.name}</h1>
              {area.description && (
                <p className="text-gray-500 mt-1">{area.description}</p>
              )}
              <p className="text-sm text-gray-400 mt-1">{area.tag_count} tags</p>
            </div>
          </div>
          <button
            onClick={() => { setEditingTag(null); setShowTagModal(true) }}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Add Tag
          </button>
        </div>
      </div>

      {/* Tags List */}
      {area.tags.length > 0 ? (
        <div className="space-y-3">
          {area.tags.map((tag) => (
            <TagCard
              key={tag.id}
              tag={tag}
              color={area.color}
              onEdit={() => { setEditingTag(tag); setShowTagModal(true) }}
              onDelete={() => {
                if (confirm(`Delete tag "${tag.name}"?`)) {
                  deleteMutation.mutate(tag.id)
                }
              }}
            />
          ))}
        </div>
      ) : (
        <div className="rounded-lg bg-white border-2 border-dashed p-12 text-center">
          <Tag className="mx-auto h-12 w-12 text-gray-300" />
          <h3 className="mt-4 text-lg font-medium text-gray-600">No tags yet</h3>
          <p className="mt-2 text-gray-400">Add tags to start classifying documents</p>
          <button
            onClick={() => { setEditingTag(null); setShowTagModal(true) }}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Add First Tag
          </button>
        </div>
      )}

      {/* Tag Modal */}
      {showTagModal && (
        <TagModal
          aolId={area.id}
          tag={editingTag || undefined}
          onClose={() => { setShowTagModal(false); setEditingTag(null) }}
        />
      )}
    </div>
  )
}
