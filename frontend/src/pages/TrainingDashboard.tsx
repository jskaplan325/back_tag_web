import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Database,
  FileText,
  Tag,
  Download,
  Loader2,
  BarChart3,
  CheckCircle,
  HelpCircle,
  Ban,
  ExternalLink,
  PieChart
} from 'lucide-react'
import api from '../api'

interface TrainingSummary {
  total_annotations: number
  annotated_documents: number
  total_documents: number
  coverage_percent: number
  by_type: {
    positive: number
    negative: number
    uncertain: number
  }
  by_color: {
    green: number
    yellow: number
    red: number
  }
  ignore_regions: number
  top_tags: Array<{
    tag: string
    area_of_law: string
    count: number
  }>
}

interface AnnotatedDocument {
  id: string
  filename: string
  matter_id: string | null
  status: string
  uploaded_at: string | null
  annotation_count: number
  green_count: number
  yellow_count: number
  red_count: number
  last_annotated: string | null
}

interface DocumentsResponse {
  total: number
  documents: AnnotatedDocument[]
}

function StatCard({
  icon: Icon,
  label,
  value,
  subValue,
  color = 'blue'
}: {
  icon: React.ElementType
  label: string
  value: string | number
  subValue?: string
  color?: 'blue' | 'green' | 'yellow' | 'red' | 'purple'
}) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    yellow: 'bg-yellow-50 text-yellow-600',
    red: 'bg-red-50 text-red-600',
    purple: 'bg-purple-50 text-purple-600'
  }

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subValue && <p className="text-xs text-gray-400">{subValue}</p>}
        </div>
      </div>
    </div>
  )
}

function ProgressBar({ value, max, color }: { value: number; max: number; color: string }) {
  const percent = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="w-full bg-gray-100 rounded-full h-2">
      <div
        className={`h-2 rounded-full ${color}`}
        style={{ width: `${Math.min(percent, 100)}%` }}
      />
    </div>
  )
}

export default function TrainingDashboard() {
  // Fetch training summary
  const { data: summary, isLoading: summaryLoading } = useQuery<TrainingSummary>({
    queryKey: ['training-summary'],
    queryFn: () => api.get('/api/training/summary').then(r => r.data),
    refetchInterval: 30000
  })

  // Fetch annotated documents
  const { data: documentsData, isLoading: docsLoading } = useQuery<DocumentsResponse>({
    queryKey: ['training-documents'],
    queryFn: () => api.get('/api/training/documents?limit=20').then(r => r.data)
  })

  const handleExport = async (format: 'json' | 'csv') => {
    try {
      const response = await api.get(`/api/training/export?format=${format}`, {
        responseType: 'blob'
      })
      const blob = new Blob([response.data], {
        type: format === 'json' ? 'application/json' : 'text/csv'
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `training_data.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  const isLoading = summaryLoading || docsLoading

  if (isLoading && !summary) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  const totalByType = (summary?.by_type.positive || 0) +
    (summary?.by_type.negative || 0) +
    (summary?.by_type.uncertain || 0)

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Database className="h-6 w-6" />
            Training Data
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Review and export your annotated training data for ML model improvement
          </p>
        </div>

        {/* Export buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => handleExport('json')}
            className="px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export JSON
          </button>
          <button
            onClick={() => handleExport('csv')}
            className="px-3 py-2 bg-white border text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          icon={Tag}
          label="Total Annotations"
          value={summary?.total_annotations || 0}
          color="blue"
        />
        <StatCard
          icon={FileText}
          label="Annotated Documents"
          value={summary?.annotated_documents || 0}
          subValue={`${summary?.coverage_percent || 0}% coverage`}
          color="purple"
        />
        <StatCard
          icon={CheckCircle}
          label="Positive Examples"
          value={summary?.by_type.positive || 0}
          color="green"
        />
        <StatCard
          icon={Ban}
          label="Ignore Regions"
          value={summary?.ignore_regions || 0}
          color="red"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Breakdown & Top Tags */}
        <div className="lg:col-span-1 space-y-6">
          {/* Annotation Type Breakdown */}
          <div className="bg-white rounded-lg border shadow-sm p-4">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2 mb-4">
              <PieChart className="h-4 w-4" />
              Annotation Types
            </h3>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
                    Positive
                  </span>
                  <span className="font-medium">{summary?.by_type.positive || 0}</span>
                </div>
                <ProgressBar value={summary?.by_type.positive || 0} max={totalByType} color="bg-green-500" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                    Uncertain
                  </span>
                  <span className="font-medium">{summary?.by_type.uncertain || 0}</span>
                </div>
                <ProgressBar value={summary?.by_type.uncertain || 0} max={totalByType} color="bg-yellow-500" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                    Negative/Ignore
                  </span>
                  <span className="font-medium">{summary?.by_type.negative || 0}</span>
                </div>
                <ProgressBar value={summary?.by_type.negative || 0} max={totalByType} color="bg-red-500" />
              </div>
            </div>
          </div>

          {/* Top Tags */}
          <div className="bg-white rounded-lg border shadow-sm p-4">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2 mb-4">
              <BarChart3 className="h-4 w-4" />
              Top Tags by Annotations
            </h3>

            {summary?.top_tags && summary.top_tags.length > 0 ? (
              <div className="space-y-2">
                {summary.top_tags.filter(t => t.tag !== '__IGNORE__').slice(0, 10).map((item, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{item.tag}</p>
                      <p className="text-xs text-gray-500 truncate">{item.area_of_law}</p>
                    </div>
                    <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                      {item.count}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                No annotations yet
              </p>
            )}
          </div>
        </div>

        {/* Right Column - Annotated Documents */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg border shadow-sm">
            <div className="px-4 py-3 border-b flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">Annotated Documents</h3>
              <span className="text-sm text-gray-500">{documentsData?.total || 0} documents</span>
            </div>

            {documentsData?.documents && documentsData.documents.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 border-b">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium text-gray-500">Document</th>
                      <th className="px-4 py-3 text-center font-medium text-gray-500">
                        <div className="flex items-center justify-center gap-1">
                          <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
                          Green
                        </div>
                      </th>
                      <th className="px-4 py-3 text-center font-medium text-gray-500">
                        <div className="flex items-center justify-center gap-1">
                          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                          Yellow
                        </div>
                      </th>
                      <th className="px-4 py-3 text-center font-medium text-gray-500">
                        <div className="flex items-center justify-center gap-1">
                          <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                          Red
                        </div>
                      </th>
                      <th className="px-4 py-3 text-center font-medium text-gray-500">Total</th>
                      <th className="px-4 py-3 text-left font-medium text-gray-500">Last Updated</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {documentsData.documents.map((doc) => (
                      <tr key={doc.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3">
                          <Link
                            to={`/documents/${doc.id}`}
                            className="font-medium text-blue-600 hover:underline flex items-center gap-1"
                          >
                            {doc.filename.length > 45 ? doc.filename.slice(0, 45) + '...' : doc.filename}
                            <ExternalLink className="h-3 w-3" />
                          </Link>
                        </td>
                        <td className="px-4 py-3 text-center">
                          {doc.green_count > 0 ? (
                            <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-medium">
                              {doc.green_count}
                            </span>
                          ) : (
                            <span className="text-gray-300">-</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          {doc.yellow_count > 0 ? (
                            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs font-medium">
                              {doc.yellow_count}
                            </span>
                          ) : (
                            <span className="text-gray-300">-</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          {doc.red_count > 0 ? (
                            <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs font-medium">
                              {doc.red_count}
                            </span>
                          ) : (
                            <span className="text-gray-300">-</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-medium">
                            {doc.annotation_count}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500">
                          {doc.last_annotated
                            ? new Date(doc.last_annotated).toLocaleDateString()
                            : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                <FileText className="h-12 w-12 mb-3" />
                <p className="text-lg font-medium text-gray-600">No annotated documents yet</p>
                <p className="text-sm mt-1">Start by annotating documents in the viewer</p>
                <Link
                  to="/documents"
                  className="mt-4 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700"
                >
                  View Documents
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Help Section */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h4 className="font-medium text-blue-900 flex items-center gap-2">
          <HelpCircle className="h-4 w-4" />
          About Training Data
        </h4>
        <div className="mt-2 text-sm text-blue-800 space-y-1">
          <p><strong>Positive (green):</strong> Examples where the tag correctly applies to the region</p>
          <p><strong>Uncertain (yellow):</strong> Examples needing review - useful for active learning</p>
          <p><strong>Negative/Ignore (red):</strong> Regions to exclude from ML training (headers, footers, noise)</p>
          <p className="mt-2">Export your annotations to fine-tune models or train custom classifiers.</p>
        </div>
      </div>
    </div>
  )
}
