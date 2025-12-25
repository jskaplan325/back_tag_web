import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Scale,
  Plus,
  X,
  ChevronRight,
  Tag,
  FileText,
  Loader2,
  AlertCircle,
  TrendingUp,
  Building2,
  Trophy
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface TagResponse {
  id: string
  area_of_law_id: string
  name: string
  description: string | null
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

interface TagScoreboardEntry {
  tag_id: string
  tag_name: string
  area_of_law_name: string
  area_of_law_color: string
  usage_count: number
  avg_confidence: number
  document_percent: number
}

interface Scoreboard {
  total_documents: number
  total_tags: number
  total_areas: number
  top_tags: TagScoreboardEntry[]
}

const iconMap: Record<string, React.ElementType> = {
  Scale: Scale,
  Building2: Building2,
  TrendingUp: TrendingUp,
  FileText: FileText,
  Tag: Tag,
}

function AddAreaModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [color, setColor] = useState('#3b82f6')
  const [error, setError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const colors = [
    '#3b82f6', // blue
    '#8b5cf6', // purple
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#84cc16', // lime
  ]

  const createMutation = useMutation({
    mutationFn: async () => {
      return api.post('/api/taxonomy', { name, description, color, icon: 'Scale' })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] })
      onClose()
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Add Area of Law</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={(e) => { e.preventDefault(); createMutation.mutate() }} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Intellectual Property"
              className="w-full rounded-lg border p-2"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this area of law"
              className="w-full rounded-lg border p-2"
              rows={2}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Color</label>
            <div className="flex gap-2">
              {colors.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setColor(c)}
                  className={clsx(
                    'w-8 h-8 rounded-full transition-transform',
                    color === c && 'ring-2 ring-offset-2 ring-gray-400 scale-110'
                  )}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
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
              disabled={createMutation.isPending || !name}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {createMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function AreaCard({ area }: { area: AreaOfLaw }) {
  const IconComponent = iconMap[area.icon] || Scale

  return (
    <Link
      to={`/taxonomy/${area.id}`}
      className="block rounded-lg bg-white p-6 shadow hover:shadow-md transition-shadow"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className="rounded-lg p-3"
            style={{ backgroundColor: `${area.color}20` }}
          >
            <IconComponent className="h-6 w-6" style={{ color: area.color }} />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{area.name}</h3>
            <p className="text-sm text-gray-500">{area.tag_count} tags</p>
          </div>
        </div>
        <ChevronRight className="h-5 w-5 text-gray-400" />
      </div>
      {area.description && (
        <p className="mt-3 text-sm text-gray-600">{area.description}</p>
      )}
      {area.tags.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {area.tags.slice(0, 4).map((tag) => (
            <span
              key={tag.id}
              className="px-2 py-1 rounded text-xs"
              style={{ backgroundColor: `${area.color}15`, color: area.color }}
            >
              {tag.name}
            </span>
          ))}
          {area.tags.length > 4 && (
            <span className="px-2 py-1 rounded text-xs bg-gray-100 text-gray-500">
              +{area.tags.length - 4} more
            </span>
          )}
        </div>
      )}
    </Link>
  )
}

function Scoreboard({ data }: { data: Scoreboard }) {
  return (
    <div className="rounded-lg bg-white shadow p-6">
      <div className="flex items-center gap-2 mb-4">
        <Trophy className="h-5 w-5 text-yellow-500" />
        <h2 className="font-semibold text-lg">Tag Scoreboard</h2>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <p className="text-2xl font-bold">{data.total_areas}</p>
          <p className="text-xs text-gray-500">Areas of Law</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold">{data.total_tags}</p>
          <p className="text-xs text-gray-500">Total Tags</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold">{data.total_documents}</p>
          <p className="text-xs text-gray-500">Documents</p>
        </div>
      </div>

      {/* Top Tags */}
      {data.top_tags.length > 0 ? (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-500">Most Used Tags</h3>
          {data.top_tags.slice(0, 10).map((tag, idx) => (
            <div key={tag.tag_id} className="flex items-center gap-3">
              <span className="w-6 text-center text-sm font-medium text-gray-400">
                {idx + 1}
              </span>
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: tag.area_of_law_color }}
              />
              <div className="flex-1">
                <p className="text-sm font-medium">{tag.tag_name}</p>
                <p className="text-xs text-gray-500">{tag.area_of_law_name}</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium">{tag.usage_count}</p>
                <p className="text-xs text-gray-500">{tag.document_percent}% docs</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-400">
          <Tag className="mx-auto h-8 w-8 mb-2" />
          <p className="text-sm">No tag usage data yet</p>
          <p className="text-xs">Process documents to see stats</p>
        </div>
      )}
    </div>
  )
}

export default function Taxonomy() {
  const [showAddModal, setShowAddModal] = useState(false)
  const queryClient = useQueryClient()

  const { data: areas, isLoading, error } = useQuery<AreaOfLaw[]>({
    queryKey: ['taxonomy'],
    queryFn: () => api.get('/api/taxonomy').then(r => r.data),
  })

  const { data: scoreboard } = useQuery<Scoreboard>({
    queryKey: ['taxonomy', 'scoreboard'],
    queryFn: () => api.get('/api/taxonomy/scoreboard').then(r => r.data),
  })

  const seedMutation = useMutation({
    mutationFn: () => api.post('/api/taxonomy/seed-defaults'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] })
    },
  })

  if (error) {
    console.error('Failed to fetch taxonomy:', error)
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Taxonomy</h1>
          <p className="text-gray-500">Manage Areas of Law and their tags</p>
        </div>
        <div className="flex gap-2">
          {areas?.length === 0 && (
            <button
              onClick={() => seedMutation.mutate()}
              disabled={seedMutation.isPending}
              className="flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
            >
              {seedMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Tag className="h-4 w-4" />}
              Load Defaults
            </button>
          )}
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Add Area of Law
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Areas of Law */}
        <div className="lg:col-span-2">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
            </div>
          ) : error ? (
            <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
              <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
              <h3 className="mt-2 font-medium text-red-800">Failed to load taxonomy</h3>
              <p className="mt-1 text-sm text-red-600">Make sure the backend is running</p>
            </div>
          ) : areas && areas.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {areas.map((area) => (
                <AreaCard key={area.id} area={area} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg bg-white p-12 shadow text-center">
              <Scale className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-4 text-lg font-medium">No Areas of Law yet</h3>
              <p className="mt-2 text-gray-500">Create an area or load the default taxonomy</p>
              <div className="mt-4 flex justify-center gap-3">
                <button
                  onClick={() => seedMutation.mutate()}
                  disabled={seedMutation.isPending}
                  className="inline-flex items-center gap-2 rounded-lg border px-4 py-2"
                >
                  Load Defaults
                </button>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white"
                >
                  <Plus className="h-4 w-4" />
                  Create New
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Scoreboard */}
        <div>
          {scoreboard && <Scoreboard data={scoreboard} />}
        </div>
      </div>

      {showAddModal && <AddAreaModal onClose={() => setShowAddModal(false)} />}
    </div>
  )
}
