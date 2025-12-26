import { useState, useEffect } from 'react'
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

interface MatterWithStats {
  id: string
  name: string
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
  'Investment Funds': '#3b82f6',
  'M&A / Corporate': '#8b5cf6',
  'Leveraged Finance': '#10b981',
  'Securities / Capital Markets': '#f59e0b',
  'Contracts / Commercial': '#6b7280',
  'Real Estate': '#ec4899',
  'Intellectual Property': '#06b6d4',
  'Employment': '#84cc16',
  'Tax': '#14b8a6',
  'Litigation': '#ef4444'
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
              const color = matterTypeColors[s.matter_type] || '#6b7280'

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

interface BrowseFolder {
  name: string
  path: string
}

interface BrowseResult {
  current_path: string
  parent_path: string | null
  folders: BrowseFolder[]
}

function FolderBrowser({
  onSelect,
  onCancel
}: {
  onSelect: (path: string) => void
  onCancel: () => void
}) {
  const [currentPath, setCurrentPath] = useState<string | null>(null)
  const [folders, setFolders] = useState<BrowseFolder[]>([])
  const [parentPath, setParentPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const browseTo = async (path: string | null) => {
    setLoading(true)
    try {
      const response = await api.post('/api/matters/browse-folder', { path })
      const data: BrowseResult = response.data
      setCurrentPath(data.current_path)
      setParentPath(data.parent_path)
      setFolders(data.folders)
    } catch (err) {
      console.error('Browse error:', err)
    } finally {
      setLoading(false)
    }
  }

  // Load initial folder on mount
  useEffect(() => {
    browseTo(null)
  }, [])

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-lg bg-white p-4 shadow-xl mx-4 max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold">Browse Folders</h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Current path */}
        <div className="text-xs text-gray-500 bg-gray-50 px-2 py-1 rounded mb-2 font-mono truncate">
          {currentPath || 'Loading...'}
        </div>

        {/* Folder list */}
        <div className="flex-1 border rounded overflow-auto min-h-[200px] max-h-[400px]">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
            </div>
          ) : (
            <div className="divide-y">
              {/* Parent folder */}
              {parentPath && (
                <button
                  onClick={() => browseTo(parentPath)}
                  className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-50 text-left"
                >
                  <ChevronUp className="h-4 w-4 text-gray-400" />
                  <span className="text-gray-600">..</span>
                </button>
              )}
              {/* Subfolders */}
              {folders.map((folder) => (
                <button
                  key={folder.path}
                  onClick={() => browseTo(folder.path)}
                  className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-50 text-left"
                >
                  <FolderOpen className="h-4 w-4 text-blue-500" />
                  <span className="truncate">{folder.name}</span>
                </button>
              ))}
              {folders.length === 0 && !parentPath && (
                <div className="text-center text-gray-400 py-8">No folders found</div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 mt-3">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={() => currentPath && onSelect(currentPath)}
            disabled={!currentPath}
            className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            Select This Folder
          </button>
        </div>
      </div>
    </div>
  )
}

function BulkImportModal({ onClose }: { onClose: () => void }) {
  const [folderPath, setFolderPath] = useState('')
  const [showBrowser, setShowBrowser] = useState(false)
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [selections, setSelections] = useState<Record<string, boolean>>({})
  const [typeOverrides, setTypeOverrides] = useState<Record<string, string>>({})
  const [editingType, setEditingType] = useState<string | null>(null)
  const [importResult, setImportResult] = useState<BulkImportResult | null>(null)
  const [showReimportWarning, setShowReimportWarning] = useState(false)
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
      // Build name overrides from scan result
      const nameOverrides: Record<string, string> = {}
      scanResult?.subfolders.forEach(folder => {
        if (selections[folder.path]) {
          nameOverrides[folder.path] = folder.name
        }
      })
      return api.post('/api/matters/bulk-import', {
        folder_path: folderPath,
        selected_folders: selectedFolders,
        type_overrides: typeOverrides,
        name_overrides: nameOverrides
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

  // Count selected folders that are already imported (reimports)
  const selectedReimports = scanResult?.subfolders
    .filter(f => selections[f.path] && f.already_imported) || []
  const reimportCount = selectedReimports.length

  // Get effective type (override or inferred)
  const getEffectiveType = (folder: SubfolderInfo) =>
    typeOverrides[folder.path] || folder.matter_type

  // Available types from taxonomy + fallback
  const availableTypes = areasOfLaw?.map(a => a.name) || Object.keys(matterTypeColors)

  // Handle import click - check for reimports first
  const handleImportClick = () => {
    if (reimportCount > 0) {
      setShowReimportWarning(true)
    } else {
      importMutation.mutate()
    }
  }

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
              <div className="flex gap-2">
                <input
                  type="text"
                  value={folderPath}
                  onChange={(e) => setFolderPath(e.target.value)}
                  placeholder="/path/to/legal_dataset"
                  className="flex-1 rounded-lg border p-2 font-mono text-sm"
                />
                <button
                  onClick={() => setShowBrowser(true)}
                  className="px-3 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 text-gray-700"
                  title="Browse folders"
                >
                  <FolderOpen className="h-4 w-4" />
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Each subfolder will become a matter, and files within will be imported as documents.
              </p>
            </div>

            <button
              onClick={() => scanMutation.mutate()}
              disabled={scanMutation.isPending || !folderPath}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"
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
                              disabled={folder.document_count === 0}
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
                            {editingType === folder.path ? (
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
                                onClick={() => setEditingType(folder.path)}
                                className={clsx(
                                  "px-2 py-0.5 rounded text-xs transition-all hover:ring-2 hover:ring-blue-300 cursor-pointer",
                                  typeOverrides[folder.path] && "ring-2 ring-blue-400",
                                  getEffectiveType(folder) === 'TBD' && "border border-dashed border-gray-400"
                                )}
                                style={{
                                  backgroundColor: getEffectiveType(folder) === 'TBD'
                                    ? '#f3f4f6'
                                    : `${matterTypeColors[getEffectiveType(folder)] || '#6b7280'}20`,
                                  color: getEffectiveType(folder) === 'TBD'
                                    ? '#9ca3af'
                                    : matterTypeColors[getEffectiveType(folder)] || '#6b7280'
                                }}
                                title={getEffectiveType(folder) === 'TBD'
                                  ? "Click to set type manually. Future: will be imported from MatterDB"
                                  : "Click to change type"
                                }
                              >
                                {getEffectiveType(folder)}
                                {typeOverrides[folder.path] && " ✓"}
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
                  <div className="text-sm text-gray-500">
                    <span>{selectedDocCount} documents to import</span>
                    {reimportCount > 0 && (
                      <span className="ml-2 text-amber-600">
                        ({reimportCount} already imported)
                      </span>
                    )}
                  </div>
                  <button
                    onClick={handleImportClick}
                    disabled={importMutation.isPending || selectedCount === 0}
                    className={clsx(
                      "flex items-center gap-2 rounded-lg px-4 py-2 text-white disabled:opacity-50",
                      reimportCount > 0 ? "bg-amber-600 hover:bg-amber-700" : "bg-blue-600 hover:bg-blue-700"
                    )}
                  >
                    {importMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Plus className="h-4 w-4" />
                    )}
                    {reimportCount > 0 ? `Import (${reimportCount} reimport)` : `Import ${selectedCount} Matters`}
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

      {/* Reimport Warning Modal */}
      {showReimportWarning && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="rounded-full bg-amber-100 p-2">
                <AlertTriangle className="h-6 w-6 text-amber-600" />
              </div>
              <h2 className="text-xl font-semibold">Re-import Matters?</h2>
            </div>

            <p className="text-gray-600 mb-4">
              You've selected <strong>{reimportCount}</strong> matter{reimportCount > 1 ? 's' : ''} that {reimportCount > 1 ? 'have' : 'has'} already been imported:
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4 max-h-32 overflow-auto">
              <ul className="text-sm text-amber-800 space-y-1">
                {selectedReimports.map(f => (
                  <li key={f.path} className="flex justify-between">
                    <span>{f.name}</span>
                    <span className="text-amber-600">{f.document_count} docs</span>
                  </li>
                ))}
              </ul>
            </div>

            <p className="text-sm text-gray-500 mb-4">
              Re-importing will <strong>overwrite</strong> existing documents and reset processing status.
              Use this after updating the taxonomy to reprocess with new tags.
            </p>

            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowReimportWarning(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowReimportWarning(false)
                  importMutation.mutate()
                }}
                disabled={importMutation.isPending}
                className="flex items-center gap-2 rounded-lg bg-amber-600 px-4 py-2 text-white hover:bg-amber-700 disabled:opacity-50"
              >
                {importMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Plus className="h-4 w-4" />
                )}
                Re-import {reimportCount} Matter{reimportCount > 1 ? 's' : ''}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Folder Browser Modal */}
      {showBrowser && (
        <FolderBrowser
          onSelect={(path) => {
            setFolderPath(path)
            setShowBrowser(false)
          }}
          onCancel={() => setShowBrowser(false)}
        />
      )}
    </div>
  )
}

function RetryButton({ matterId, count }: { matterId: string; count: number }) {
  const queryClient = useQueryClient()
  const retryMutation = useMutation({
    mutationFn: () => api.post(`/api/matters/${matterId}/retry-failed`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matters'] })
      queryClient.invalidateQueries({ queryKey: ['failed-docs', matterId] })
    },
  })

  return (
    <button
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        retryMutation.mutate()
      }}
      disabled={retryMutation.isPending}
      className="flex items-center gap-1 px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
    >
      {retryMutation.isPending ? (
        <Loader2 className="h-3 w-3 animate-spin" />
      ) : (
        <Play className="h-3 w-3" />
      )}
      Retry {count} failed
    </button>
  )
}

function MatterCard({
  matter,
  topTags
}: {
  matter: MatterWithStats
  topTags?: { tag: string; average_confidence: number }[]
}) {
  const [showErrors, setShowErrors] = useState(false)
  const progress = matter.document_count > 0
    ? Math.round((matter.completed_count / matter.document_count) * 100)
    : 0

  // Fetch failed documents when clicking on failed count
  const { data: failedDocs } = useQuery<{ id: string; filename: string; error_message: string }[]>({
    queryKey: ['failed-docs', matter.id],
    queryFn: () => api.get(`/api/matters/${matter.id}/failed-documents`).then(r => r.data),
    enabled: showErrors && matter.failed_count > 0,
  })

  return (
    <div className="rounded-lg bg-white p-4 shadow hover:shadow-md transition-shadow">
      <Link to={`/matters/${matter.id}`}>
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm truncate" title={matter.name}>
              {matter.name}
            </h3>
            <p className="text-xs text-gray-500">
              {matter.document_count} docs
            </p>
          </div>
          <div className="flex items-center gap-2 ml-2">
            {matter.average_confidence != null && (
              <span className={clsx(
                "text-xs font-medium px-1.5 py-0.5 rounded",
                matter.average_confidence >= 0.7 && "bg-green-100 text-green-700",
                matter.average_confidence >= 0.5 && matter.average_confidence < 0.7 && "bg-yellow-100 text-yellow-700",
                matter.average_confidence < 0.5 && "bg-red-100 text-red-700"
              )}>
                {(matter.average_confidence * 100).toFixed(0)}%
              </span>
            )}
            <ChevronRight className="h-4 w-4 text-gray-400" />
          </div>
        </div>
      </Link>

      {/* Progress bar */}
      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden mb-2">
        <div
          className={clsx(
            "h-full rounded-full transition-all",
            matter.failed_count > 0 ? "bg-red-500" : "bg-green-500"
          )}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Status */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        {matter.pending_count > 0 && (
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {matter.pending_count} pending
          </span>
        )}
        {matter.completed_count > 0 && (
          <span className="flex items-center gap-1 text-green-600">
            <CheckCircle className="h-3 w-3" />
            {matter.completed_count} done
          </span>
        )}
        {matter.failed_count > 0 && (
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.preventDefault()
                setShowErrors(!showErrors)
              }}
              className="flex items-center gap-1 text-red-600 hover:text-red-800"
              title="Click to see error details"
            >
              <XCircle className="h-3 w-3" />
              {matter.failed_count} failed
              {showErrors ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            <RetryButton matterId={matter.id} count={matter.failed_count} />
          </div>
        )}
      </div>

      {/* Error details */}
      {showErrors && matter.failed_count > 0 && (
        <div className="mt-2 p-2 bg-red-50 rounded text-xs">
          {failedDocs ? (
            <>
              <ul className="space-y-1">
                {failedDocs.slice(0, 3).map(doc => (
                  <li key={doc.id} className="text-red-700">
                    <span className="font-medium">{doc.filename}:</span>{' '}
                    <span className="text-red-600">{doc.error_message?.slice(0, 80)}...</span>
                  </li>
                ))}
                {failedDocs.length > 3 && (
                  <li className="text-red-500">...and {failedDocs.length - 3} more</li>
                )}
              </ul>
            </>
          ) : (
            <span className="text-red-600">Loading...</span>
          )}
        </div>
      )}

      {/* Top tags */}
      {topTags && topTags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {topTags.slice(0, 3).map((t) => (
            <span
              key={t.tag}
              className="text-xs bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded"
            >
              {t.tag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Matters() {
  const [showImportModal, setShowImportModal] = useState(false)
  const [processAllTarget, setProcessAllTarget] = useState<{ type: string | null, count: number } | null>(null)
  const [processingMode, setProcessingMode] = useState<'fast' | 'smart' | 'auto'>('auto')
  const [showStatusPanel, setShowStatusPanel] = useState(true)
  const queryClient = useQueryClient()

  // Fetch individual matters
  const { data: matters, isLoading, error } = useQuery<MatterWithStats[]>({
    queryKey: ['matters'],
    queryFn: () => api.get('/api/matters').then(r => r.data),
    refetchInterval: 3000,
  })

  // Also fetch stats for the status panel
  const { data: stats } = useQuery<TypeStats[]>({
    queryKey: ['matter-stats'],
    queryFn: () => api.get('/api/matters/stats/by-type').then(r => r.data),
    refetchInterval: 3000,
  })

  const processNewMutation = useMutation({
    mutationFn: (matterType: string | null) =>
      api.post('/api/matters/process-batch', {
        matter_type: matterType,
        only_pending: true,
        fast_mode: processingMode === 'fast',
        smart_mode: processingMode === 'smart',
        auto_mode: processingMode === 'auto'
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matters'] })
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
    },
  })

  const processAllMutation = useMutation({
    mutationFn: (matterType: string | null) =>
      api.post('/api/matters/process-batch', {
        matter_type: matterType,
        only_pending: false,
        fast_mode: processingMode === 'fast',
        smart_mode: processingMode === 'smart',
        auto_mode: processingMode === 'auto'
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['matters'] })
      queryClient.invalidateQueries({ queryKey: ['matter-stats'] })
      setProcessAllTarget(null)
    },
  })

  const totalStats = matters?.reduce(
    (acc, m) => ({
      matters: acc.matters + 1,
      documents: acc.documents + m.document_count,
      pending: acc.pending + m.pending_count,
      completed: acc.completed + m.completed_count,
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
          {/* Processing Mode Toggle */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setProcessingMode('fast')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm font-medium transition-colors',
                processingMode === 'fast'
                  ? 'bg-white shadow text-gray-900'
                  : 'text-gray-600 hover:text-gray-900'
              )}
              title="Fast text-only semantic tagging"
            >
              Fast
            </button>
            <button
              onClick={() => setProcessingMode('smart')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm font-medium transition-colors',
                processingMode === 'smart'
                  ? 'bg-white shadow text-purple-700'
                  : 'text-gray-600 hover:text-gray-900'
              )}
              title="Uses Gemini AI for smarter tagging (requires GOOGLE_API_KEY)"
            >
              Smart
              <span className="text-xs text-gray-400 ml-1">(AI)</span>
            </button>
            <button
              onClick={() => setProcessingMode('auto')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm font-medium transition-colors',
                processingMode === 'auto'
                  ? 'bg-white shadow text-blue-700'
                  : 'text-gray-600 hover:text-gray-900'
              )}
              title="Auto-select pipeline based on document analysis (OCR, vision, zone, or fast)"
            >
              Auto
              <span className="text-xs text-gray-400 ml-1">✨</span>
            </button>
          </div>
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
      ) : matters && matters.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {matters.map((matter) => (
            <MatterCard
              key={matter.id}
              matter={matter}
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
