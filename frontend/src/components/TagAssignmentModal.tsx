import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, Search, Plus, Check, Ban } from 'lucide-react'
import clsx from 'clsx'
import api from '../api'
import type { BoundingBox } from './AnnotationCanvas'

interface TagSearchResult {
  id: string
  name: string
  area_of_law: string
  area_of_law_id: string
}

interface TagAssignmentModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (data: {
    tag_name: string
    tag_id: string | null
    area_of_law: string | null
    annotation_type: 'positive' | 'negative' | 'uncertain'
    color: 'green' | 'yellow' | 'red'
  }) => void
  boundingBox: BoundingBox | null
}

export function TagAssignmentModal({
  isOpen,
  onClose,
  onSubmit,
  boundingBox
}: TagAssignmentModalProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTag, setSelectedTag] = useState<TagSearchResult | null>(null)
  const [customTagName, setCustomTagName] = useState('')
  const [annotationType, setAnnotationType] = useState<'positive' | 'negative' | 'uncertain'>('positive')
  const [color, setColor] = useState<'green' | 'yellow' | 'red'>('green')
  const [showDropdown, setShowDropdown] = useState(false)

  // Search tags
  const { data: searchResults } = useQuery<TagSearchResult[]>({
    queryKey: ['tags-search', searchQuery],
    queryFn: () => api.get(`/api/tags/search?q=${encodeURIComponent(searchQuery)}&limit=10`).then(r => r.data),
    enabled: searchQuery.length > 0 && annotationType !== 'negative'
  })

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setSearchQuery('')
      setSelectedTag(null)
      setCustomTagName('')
      setAnnotationType('positive')
      setColor('green')
      setShowDropdown(false)
    }
  }, [isOpen])

  // Update color based on annotation type
  useEffect(() => {
    if (annotationType === 'positive') setColor('green')
    else if (annotationType === 'negative') setColor('red')
    else setColor('yellow')
  }, [annotationType])

  const handleSelectTag = (tag: TagSearchResult) => {
    setSelectedTag(tag)
    setCustomTagName('')
    setSearchQuery(tag.name)
    setShowDropdown(false)
  }

  const handleCreateCustomTag = () => {
    if (searchQuery.trim()) {
      setCustomTagName(searchQuery.trim())
      setSelectedTag(null)
      setShowDropdown(false)
    }
  }

  const handleSubmit = () => {
    // For negative/ignore zones, use special tag name
    if (annotationType === 'negative') {
      onSubmit({
        tag_name: '__IGNORE__',
        tag_id: null,
        area_of_law: null,
        annotation_type: 'negative',
        color: 'red'
      })
      return
    }

    const tagName = selectedTag?.name || customTagName
    if (!tagName) return

    onSubmit({
      tag_name: tagName,
      tag_id: selectedTag?.id || null,
      area_of_law: selectedTag?.area_of_law || null,
      annotation_type: annotationType,
      color: color
    })
  }

  // Can submit if: negative type (ignore zone) OR has a tag selected
  const canSubmit = annotationType === 'negative' || selectedTag || customTagName

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h3 className="text-lg font-semibold text-gray-900">
            {annotationType === 'negative' ? 'Mark Ignore Region' : 'Assign Tag to Region'}
          </h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Bounding box preview */}
          {boundingBox && (
            <div className="text-xs text-gray-500 bg-gray-50 px-2 py-1 rounded">
              Region: ({(boundingBox.x1 * 100).toFixed(0)}%, {(boundingBox.y1 * 100).toFixed(0)}%) to ({(boundingBox.x2 * 100).toFixed(0)}%, {(boundingBox.y2 * 100).toFixed(0)}%)
            </div>
          )}

          {/* Annotation Type - moved to top */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Region Type
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setAnnotationType('positive')}
                className={clsx(
                  'flex-1 py-2 px-3 rounded-lg border-2 text-sm font-medium transition-colors',
                  annotationType === 'positive'
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-200 hover:border-gray-300'
                )}
              >
                <div className="flex items-center justify-center gap-1">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  Positive
                </div>
                <div className="text-xs text-gray-500 mt-0.5">Tag applies</div>
              </button>

              <button
                onClick={() => setAnnotationType('uncertain')}
                className={clsx(
                  'flex-1 py-2 px-3 rounded-lg border-2 text-sm font-medium transition-colors',
                  annotationType === 'uncertain'
                    ? 'border-yellow-500 bg-yellow-50 text-yellow-700'
                    : 'border-gray-200 hover:border-gray-300'
                )}
              >
                <div className="flex items-center justify-center gap-1">
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  Uncertain
                </div>
                <div className="text-xs text-gray-500 mt-0.5">Needs review</div>
              </button>

              <button
                onClick={() => setAnnotationType('negative')}
                className={clsx(
                  'flex-1 py-2 px-3 rounded-lg border-2 text-sm font-medium transition-colors',
                  annotationType === 'negative'
                    ? 'border-red-500 bg-red-50 text-red-700'
                    : 'border-gray-200 hover:border-gray-300'
                )}
              >
                <div className="flex items-center justify-center gap-1">
                  <Ban className="w-3 h-3 text-red-500" />
                  Ignore
                </div>
                <div className="text-xs text-gray-500 mt-0.5">Skip region</div>
              </button>
            </div>
          </div>

          {/* Ignore zone message */}
          {annotationType === 'negative' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-center">
              <Ban className="mx-auto h-8 w-8 text-red-400 mb-2" />
              <p className="text-sm text-red-700 font-medium">Ignore Region</p>
              <p className="text-xs text-red-600 mt-1">
                This area will be excluded from ML training and tag detection
              </p>
            </div>
          )}

          {/* Tag Search - only show for positive/uncertain */}
          {annotationType !== 'negative' && (
            <>
              <div className="relative">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Tag Name
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value)
                      setShowDropdown(true)
                      setSelectedTag(null)
                      setCustomTagName('')
                    }}
                    onFocus={() => setShowDropdown(true)}
                    placeholder="Search existing tags or create new..."
                    className="w-full pl-9 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Dropdown */}
                {showDropdown && searchQuery && (
                  <div className="absolute z-10 mt-1 w-full bg-white border rounded-lg shadow-lg max-h-48 overflow-auto">
                    {searchResults?.map(tag => (
                      <button
                        key={tag.id}
                        onClick={() => handleSelectTag(tag)}
                        className="w-full text-left px-3 py-2 hover:bg-gray-50 flex items-center justify-between"
                      >
                        <div>
                          <div className="font-medium text-sm">{tag.name}</div>
                          <div className="text-xs text-gray-500">{tag.area_of_law}</div>
                        </div>
                        {selectedTag?.id === tag.id && (
                          <Check className="h-4 w-4 text-green-500" />
                        )}
                      </button>
                    ))}

                    {/* Create new option */}
                    {searchQuery && !searchResults?.some(t => t.name.toLowerCase() === searchQuery.toLowerCase()) && (
                      <button
                        onClick={handleCreateCustomTag}
                        className="w-full text-left px-3 py-2 hover:bg-blue-50 flex items-center gap-2 text-blue-600 border-t"
                      >
                        <Plus className="h-4 w-4" />
                        <span className="text-sm">Create "{searchQuery}"</span>
                      </button>
                    )}

                    {!searchResults?.length && !searchQuery && (
                      <div className="px-3 py-2 text-sm text-gray-500">
                        Type to search tags...
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Selected tag indicator */}
              {(selectedTag || customTagName) && (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600">Selected:</span>
                  <span className={clsx(
                    'px-2 py-0.5 rounded text-sm font-medium',
                    selectedTag ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'
                  )}>
                    {selectedTag?.name || customTagName}
                    {customTagName && ' (new)'}
                  </span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={clsx(
              'px-4 py-2 text-sm font-medium rounded-lg',
              canSubmit
                ? annotationType === 'negative'
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            )}
          >
            {annotationType === 'negative' ? 'Mark as Ignore' : 'Save Annotation'}
          </button>
        </div>
      </div>
    </div>
  )
}
