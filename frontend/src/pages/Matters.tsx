import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FolderOpen,
  Plus,
  X,
  ChevronRight,
  Loader2,
  AlertCircle,
  FolderInput,
  CheckCircle,
  Play,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Activity,
  XCircle
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
}

interface SubfolderInfo {
  name: string
  path: string
  document_count: number
  matter_type: string
  selected: boolean
  already_imported: boolean
}

interface AreaOfLaw {
  id: string
  name: string
  color: string
}

interface ScanResult {
  folder_path: string
  subfolders: SubfolderInfo[]
  total_documents: number
}

interface BulkImportResult {
  matters_created: number
  documents_imported: number
  matters: Matter[]
  errors: string[]
}

interface TypeStats {
  matter_type: string
  matter_count: number
  document_count: number
  pending_count: number
  processing_count: number
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

function ProcessWarningModal({
  matterType,
  documentCount,
  onConfirm,
  onCancel,
  isPending
}: {
  matterType: string | null
  documentCount: number
  onConfirm: () => void
  onCancel: () => void
  isPending: boolean
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl mx-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="rounded-full bg-yellow-100 p-2">
            <AlertTriangle className="h-6 w-6 text-yellow-600" />
          </div>
          <h2 className="text-xl font-semibold">Reprocess All Documents?</h2>
        </div>

        <p className="text-gray-600 mb-4">
          You are about to reprocess <strong>{documentCount}</strong> documents
          {matterType ? ` in ${matterType}` : ' across all matter types'}.
          This includes documents that have already been processed.
        </p>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
          <p className="text-sm text-yellow-800">
            <strong>Warning:</strong> This operation may incur significant processing costs
            and time. Only proceed if you need to reprocess with updated models or taxonomy.
          </p>
        </div>

        <div className="flex justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isPending}
            className="flex items-center gap-2 rounded-lg bg-yellow-600 px-4 py-2 text-white hover:bg-yellow-700 disabled:opacity-50"
          >
            {isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            Reprocess All
          </button>
        </div>
      </div>
    </div>
  )
}

function ProcessingStatusPanel({
  stats,
  isExpanded,
  onToggle
}: {
  stats: TypeStats[] | undefined
  isExpanded: boolean
  onToggle: () => void
}) {
  const totalProcessing = stats?.reduce((sum, s) => sum + s.processing_count, 0) || 0
  const totalCompleted = stats?.reduce((sum, s) => sum + s.completed_count, 0) || 0
  const totalFailed = stats?.reduce((sum, s) => sum + s.failed_count, 0) || 0
  const totalPending = stats?.reduce((sum, s) => sum + s.pending_count, 0) || 0

  const isActive = totalProcessing > 0

  if (!isActive && totalCompleted === 0 && totalFailed === 0) {
    return null
  }

  return (
    <div className={clsx(
      "fixed bottom-4 right-4 z-40 rounded-lg shadow-lg border transition-all",
      isActive ? "bg-blue-50 border-blue-200" : "bg-white border-gray-200"
    )}>
      {/* Header - always visible */}
      <button
        onClick={onToggle}
        className="flex items-center gap-3 px-4 py-3 w-full text-left"
      >
        <div className={clsx(
          "rounded-full p-1.5",
          isActive ? "bg-blue-100" : "bg-gray-100"
        )}>
          {isActive ? (
            <Activity className="h-4 w-4 text-blue-600 animate-pulse" />
          ) : totalFailed > 0 ? (
            <XCircle className="h-4 w-4 text-red-500" />
          ) : (
            <CheckCircle className="h-4 w-4 text-green-500" />
          )}
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium">
            {isActive ? (
              <>Processing {totalProcessing} documents...</>
            ) : totalFailed > 0 ? (
              <>{totalCompleted} completed, {totalFailed} failed</>
            ) : (
              <>All {totalCompleted} documents processed</>
            )}
          </p>
          {isActive && (
            <p className="text-xs text-gray-500">
              {totalCompleted} done, {totalPending} pending
            </p>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronUp className="h-4 w-4 text-gray-400" />
        )}
      </button>

      {/* Expanded details */}
      {isExpanded && stats && (
        <div className="px-4 pb-3 border-t border-gray-200">
          <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
            {stats.map((s) => {
              const progress = s.document_count > 0
                ? Math.round((s.completed_count / s.document_count) * 100)
                : 0
              const color = matterTypeColors[s.matter_type] || matterTypeColors['General']

              return (
                <div key={s.matter_type} className="text-sm">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium" style={{ color }}>{s.matter_type}</span>
                    <span className="text-gray-500 text-xs">
                      {s.completed_count}/{s.document_count}
                      {s.processing_count > 0 && (
                        <span className="text-blue-500 ml-1">
                          ({s.processing_count} running)
                        </span>
                      )}
                      {s.failed_count > 0 && (
                        <span className="text-red-500 ml-1">
                          ({s.failed_count} failed)
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${progress}%`,
                        backgroundColor: color
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
          <p className="text-xs text-gray-400 mt-3 text-center">
            Auto-refreshes every 2 seconds
          </p>
        </div>
      )}
    </div>
  )
}

function BulkImportModal({ onClose }: { onClose: () => void }) {
  const [folderPath, setFolderPath] = useState('/Users/jaredkaplan/Projects/legal_dataset/legal_test_matters')
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [selections, setSelections] = useState<Record<string, boolean>>({})
  const [typeOverrides, setTypeOverrides] = useState<Record<string, string>>({})
  const [editingType, setEditingType] = useState<string | null>(null)
  const [importResult, setImportResult] = useState<BulkImportResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  // Fetch Areas of Law for type dropdown
  const { data: areasOfLaw } = useQuery<AreaOfLaw[]>({
    queryKey: ['taxonomy'],
    queryFn: () => api.get('/api/taxonomy').then(r => r.data),
  })

  const scanMutation = useMutation({
    mutationFn: async () => {
      return api.post('/api/matters/scan-folder', { folder_path: folderPath })
    },
    onSuccess: (response) => {
      setScanResult(response.data)
      const initialSelections: Record<string, boolean> = {}
      response.data.subfolders.forEach((folder: SubfolderInfo) => {
        initialSelections[folder.path] = folder.selected
      })
      setSelections(initialSelections)
      setError(null)
    },
    onError: (err: Error) => {
      setError(err.message)
      setScanResult(null)
    },
  })

  const importMutation = useMutation({
    mutationFn: async () => {
      const selectedFolders = Object.entries(selections)
        .filter(([_, selected]) => selected)
        .map(([path]) => path)
      return api.post('/api/matters/bulk-import', {
        folder_path: folderPath,
        selected_folders: selectedFolders,
        type_overrides: typeOverrides
      })
    },
    onSuccess: (response) => {
      setImportResult(response.data)
      queryClient.invalidateQueries({ queryKey: ['matters'] })
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  const toggleSelection = (path: string) => {
    setSelections(prev => ({ ...prev, [path]: !prev[path] }))
  }

  const selectAll = () => {
    if (!scanResult) return
    const newSelections: Record<string, boolean> = {}
    scanResult.subfolders.forEach(folder => {
      newSelections[folder.path] = folder.document_count > 0 && !folder.already_imported
    })
    setSelections(newSelections)
  }

  const selectNone = () => {
    if (!scanResult) return
    const newSelections: Record<string, boolean> = {}
    scanResult.subfolders.forEach(folder => {
      newSelections[folder.path] = false
    })
    setSelections(newSelections)
  }

  const selectedCount = Object.values(selections).filter(Boolean).length
  const selectedDocCount = scanResult?.subfolders
    .filter(f => selections[f.path])
    .reduce((sum, f) => sum + f.document_count, 0) || 0

  // Get effective type (override or inferred)
  const getEffectiveType = (folder: SubfolderInfo) =>
    typeOverrides[folder.path] || folder.matter_type

  // Available types from taxonomy + fallback
  const availableTypes = areasOfLaw?.map(a => a.name) || Object.keys(matterTypeColors)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 overflow-auto py-8">
      <div className="w-full max-w-2xl rounded-lg bg-white p-6 shadow-xl mx-4 max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <FolderInput className="h-5 w-5" />
            Bulk Import Documents
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-6 w-6" />
          </button>
        </div>

        {!importResult ? (
          <>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Folder Path</label>
              <input
                type="text"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="/path/to/legal_dataset"
                className="w-full rounded-lg border p-2 font-mono text-sm"
              />
              <p className="text-xs text-gray-500 mt-1">
                Each subfolder will become a matter, and files within will be imported as documents.
              </p>
            </div>

            <button
              onClick={() => scanMutation.mutate()}
              disabled={scanMutation.isPending || !folderPath}
              className="flex items-center gap-2 rounded-lg bg-gray-100 px-4 py-2 text-gray-700 hover:bg-gray-200 disabled:opacity-50"
            >
              {scanMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FolderOpen className="h-4 w-4" />
              )}
              Scan Folder
            </button>

            {error && (
              <div className="mt-4 flex items-center gap-2 text-red-600 text-sm bg-red-50 p-3 rounded-lg">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            {scanResult && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium">Preview</h3>
                  <div className="flex items-center gap-3">
                    <button onClick={selectAll} className="text-xs text-blue-600 hover:underline">
                      Select All
                    </button>
                    <button onClick={selectNone} className="text-xs text-blue-600 hover:underline">
                      Select None
                    </button>
                    <span className="text-sm text-gray-500">
                      {selectedCount} of {scanResult.subfolders.length} selected
                    </span>
                  </div>
                </div>

                <div className="border rounded-lg max-h-64 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="w-10 p-2"></th>
                        <th className="text-left p-2 font-medium">Folder Name</th>
                        <th className="text-left p-2 font-medium">
                          Type
                          <span className="font-normal text-xs text-gray-400 ml-1">(click to edit)</span>
                        </th>
                        <th className="text-right p-2 font-medium">Documents</th>
                        <th className="text-left p-2 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scanResult.subfolders.map((folder) => (
                        <tr
                          key={folder.path}
                          className={clsx(
                            "border-t",
                            folder.already_imported && "bg-gray-50 text-gray-400",
                            folder.document_count === 0 && "bg-gray-50 text-gray-400"
                          )}
                        >
                          <td className="p-2 text-center">
                            <input
                              type="checkbox"
                              checked={selections[folder.path] || false}
                              onChange={() => toggleSelection(folder.path)}
                              disabled={folder.already_imported}
                              className="rounded border-gray-300"
                            />
                          </td>
                          <td className="p-2">
                            <div className="flex items-center gap-2">
                              <FolderOpen className={clsx(
                                "h-4 w-4",
                                folder.already_imported || folder.document_count === 0
                                  ? "text-gray-300"
                                  : "text-gray-400"
                              )} />
                              {folder.name}
                            </div>
                          </td>
                          <td className="p-2">
                            {editingType === folder.path && !folder.already_imported ? (
                              <select
                                value={getEffectiveType(folder)}
                                onChange={(e) => {
                                  setTypeOverrides(prev => ({
                                    ...prev,
                                    [folder.path]: e.target.value
                                  }))
                                  setEditingType(null)
                                }}
                                onBlur={() => setEditingType(null)}
                                autoFocus
                                className="text-xs rounded border border-blue-300 px-1 py-0.5 bg-white"
                              >
                                {availableTypes.map(type => (
                                  <option key={type} value={type}>{type}</option>
                                ))}
                              </select>
                            ) : (
                              <button
                                onClick={() => !folder.already_imported && setEditingType(folder.path)}
                                disabled={folder.already_imported}
                                className={clsx(
                                  "px-2 py-0.5 rounded text-xs transition-all",
                                  folder.already_imported && "opacity-50 cursor-not-allowed",
                                  !folder.already_imported && "hover:ring-2 hover:ring-blue-300 cursor-pointer",
                                  typeOverrides[folder.path] && "ring-2 ring-blue-400"
                                )}
                                style={{
                                  backgroundColor: `${matterTypeColors[getEffectiveType(folder)] || matterTypeColors['General']}20`,
                                  color: matterTypeColors[getEffectiveType(folder)] || matterTypeColors['General']
                                }}
                                title={folder.already_imported ? "Already imported" : "Click to change type"}
                              >
                                {getEffectiveType(folder)}
                                {typeOverrides[folder.path] && " *"}
                              </button>
                            )}
                          </td>
                          <td className={clsx(
                            "p-2 text-right",
                            folder.document_count === 0 && "text-gray-400"
                          )}>
                            {folder.document_count}
                          </td>
                          <td className="p-2 text-xs">
                            {folder.already_imported ? (
                              <span className="text-green-600 flex items-center gap-1">
                                <CheckCircle className="h-3 w-3" />
                                Imported
                              </span>
                            ) : folder.document_count === 0 ? (
                              <span className="text-gray-400">Empty</span>
                            ) : (
                              <span className="text-gray-500">Ready</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 flex justify-between items-center">
                  <span className="text-sm text-gray-500">
                    {selectedDocCount} documents to import
                  </span>
                  <button
                    onClick={() => importMutation.mutate()}
                    disabled={importMutation.isPending || selectedCount === 0}
                    className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
                  >
                    {importMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Plus className="h-4 w-4" />
                    )}
                    Import {selectedCount} Matters
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-6">
            <CheckCircle className="mx-auto h-12 w-12 text-green-500" />
            <h3 className="mt-4 text-lg font-medium text-green-800">Import Complete!</h3>
            <p className="mt-2 text-gray-600">
              Created {importResult.matters_created} matters with {importResult.documents_imported} documents
            </p>

            {importResult.errors.length > 0 && (
              <div className="mt-4 text-left bg-yellow-50 p-3 rounded-lg">
                <p className="text-sm font-medium text-yellow-800">Warnings:</p>
                <ul className="mt-1 text-sm text-yellow-700 list-disc list-inside">
                  {importResult.errors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            <button
              onClick={onClose}
              className="mt-6 px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700"
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function MatterTypeCard({
  stats,
  onProcessNew,
  onProcessAll,
  isProcessing
}: {
  stats: TypeStats
  onProcessNew: () => void
  onProcessAll: () => void
  isProcessing: boolean
}) {
  const color = matterTypeColors[stats.matter_type] || matterTypeColors['General']
  const hasNew = stats.pending_count > 0
  const hasAny = stats.document_count > 0

  return (
    <div className="rounded-lg bg-white p-6 shadow hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className="rounded-lg p-3"
            style={{ backgroundColor: `${color}20` }}
          >
            <FolderOpen className="h-6 w-6" style={{ color }} />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{stats.matter_type}</h3>
            <p className="text-sm text-gray-500">{stats.matter_count} matters</p>
          </div>
        </div>
        <Link
          to={`/matters/type/${encodeURIComponent(stats.matter_type)}`}
          className="text-gray-400 hover:text-gray-600"
        >
          <ChevronRight className="h-5 w-5" />
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-5 gap-2 mb-4">
        <div className="text-center">
          <p className="text-lg font-bold">{stats.document_count}</p>
          <p className="text-xs text-gray-500">Total</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold text-gray-500">{stats.pending_count}</p>
          <p className="text-xs text-gray-500">Pending</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold text-green-600">{stats.completed_count}</p>
          <p className="text-xs text-gray-500">Done</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold text-red-600">{stats.failed_count}</p>
          <p className="text-xs text-gray-500">Failed</p>
        </div>
        <div className="text-center">
          <p className={clsx(
            "text-lg font-bold",
            stats.average_confidence != null && stats.average_confidence >= 0.7 && "text-green-600",
            stats.average_confidence != null && stats.average_confidence >= 0.5 && stats.average_confidence < 0.7 && "text-yellow-600",
            stats.average_confidence != null && stats.average_confidence < 0.5 && "text-red-600",
            stats.average_confidence == null && "text-gray-400"
          )}>
            {stats.average_confidence != null ? `${(stats.average_confidence * 100).toFixed(0)}%` : '-'}
          </p>
          <p className="text-xs text-gray-500">Conf.</p>
        </div>
      </div>

      {/* Progress bar */}
      {stats.document_count > 0 && (
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden mb-4">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${(stats.completed_count / stats.document_count) * 100}%` }}
          />
        </div>
      )}

      {/* Process buttons */}
      <div className="flex gap-2">
        <button
          onClick={onProcessNew}
          disabled={!hasNew || isProcessing}
          className={clsx(
            "flex-1 flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
            hasNew
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-100 text-gray-400 cursor-not-allowed"
          )}
        >
          {isProcessing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          New ({stats.pending_count})
        </button>
        <button
          onClick={onProcessAll}
          disabled={!hasAny || isProcessing}
          className={clsx(
            "flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
            hasAny
              ? "bg-gray-100 text-gray-700 hover:bg-gray-200"
              : "bg-gray-100 text-gray-400 cursor-not-allowed"
          )}
        >
          <Clock className="h-4 w-4" />
          All
        </button>
      </div>
    </div>
  )
}

export default function Matters() {
  const [showImportModal, setShowImportModal] = useState(false)
  const [processAllTarget, setProcessAllTarget] = useState<{ type: string | null, count: number } | null>(null)
  const [fastMode, setFastMode] = useState(true)  // Fast mode enabled by default
  const [showStatusPanel, setShowStatusPanel] = useState(true)  // Processing status panel
  const queryClient = useQueryClient()

  // Check if any processing is happening to increase refresh rate
  const isProcessing = (stats?: TypeStats[]) =>
    stats?.some(s => s.processing_count > 0) || false

  const { data: stats, isLoading, error } = useQuery<TypeStats[]>({
    queryKey: ['matter-stats'],
    queryFn: () => api.get('/api/matters/stats/by-type').then(r => r.data),
    refetchInterval: (query) => isProcessing(query.state.data) ? 2000 : 5000,
  })

  const processNewMutation = useMutation({
    mutationFn: (matterType: string | null) =>
      api.post('/api/matters/process-batch', {
        matter_type: matterType,
        only_pending: true,
        fast_mode: fastMode
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const processAllMutation = useMutation({
    mutationFn: (matterType: string | null) =>
      api.post('/api/matters/process-batch', {
        matter_type: matterType,
        only_pending: false,
        fast_mode: fastMode
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      setProcessAllTarget(null)
    },
  })

  const totalStats = stats?.reduce(
    (acc, s) => ({
      matters: acc.matters + s.matter_count,
      documents: acc.documents + s.document_count,
      pending: acc.pending + s.pending_count,
      completed: acc.completed + s.completed_count,
    }),
    { matters: 0, documents: 0, pending: 0, completed: 0 }
  ) || { matters: 0, documents: 0, pending: 0, completed: 0 }

  if (error) {
    console.error('Failed to fetch matter stats:', error)
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Matters</h1>
          <p className="text-gray-500">
            {totalStats.matters} matters, {totalStats.documents} documents
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Fast Mode Toggle */}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={fastMode}
              onChange={(e) => setFastMode(e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-gray-600">Fast Mode</span>
            <span className="text-xs text-gray-400">(~3s/doc)</span>
          </label>
          <div className="flex gap-2">
          {totalStats.pending > 0 && (
            <button
              onClick={() => processNewMutation.mutate(null)}
              disabled={processNewMutation.isPending}
              className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-white hover:bg-green-700 disabled:opacity-50"
            >
              {processNewMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Process All New ({totalStats.pending})
            </button>
          )}
          {totalStats.documents > 0 && (
            <button
              onClick={() => setProcessAllTarget({ type: null, count: totalStats.documents })}
              className="flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
            >
              <Clock className="h-4 w-4" />
              Reprocess All
            </button>
          )}
          <button
            onClick={() => setShowImportModal(true)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700"
          >
            <FolderInput className="h-4 w-4" />
            Bulk Import
          </button>
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      ) : error ? (
        <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-500" />
          <h3 className="mt-2 font-medium text-red-800">Failed to load matters</h3>
          <p className="mt-1 text-sm text-red-600">Make sure the backend is running</p>
        </div>
      ) : stats && stats.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {stats.map((typeStat) => (
            <MatterTypeCard
              key={typeStat.matter_type}
              stats={typeStat}
              onProcessNew={() => processNewMutation.mutate(typeStat.matter_type)}
              onProcessAll={() => setProcessAllTarget({
                type: typeStat.matter_type,
                count: typeStat.document_count
              })}
              isProcessing={processNewMutation.isPending || processAllMutation.isPending}
            />
          ))}
        </div>
      ) : (
        <div className="rounded-lg bg-white p-12 shadow text-center">
          <FolderOpen className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No Matters yet</h3>
          <p className="mt-2 text-gray-500">Import documents from a folder to get started</p>
          <button
            onClick={() => setShowImportModal(true)}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white"
          >
            <FolderInput className="h-4 w-4" />
            Bulk Import
          </button>
        </div>
      )}

      {showImportModal && <BulkImportModal onClose={() => setShowImportModal(false)} />}

      {processAllTarget && (
        <ProcessWarningModal
          matterType={processAllTarget.type}
          documentCount={processAllTarget.count}
          onConfirm={() => processAllMutation.mutate(processAllTarget.type)}
          onCancel={() => setProcessAllTarget(null)}
          isPending={processAllMutation.isPending}
        />
      )}

      {/* Processing Status Panel - floating bottom-right */}
      <ProcessingStatusPanel
        stats={stats}
        isExpanded={showStatusPanel}
        onToggle={() => setShowStatusPanel(!showStatusPanel)}
      />
    </div>
  )
}
