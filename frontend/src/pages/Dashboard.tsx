import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { FileText, CheckCircle, TrendingUp, Clock, Tag, Filter, X, AlertTriangle, EyeOff } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Link } from 'react-router-dom'
import api from '../api'

interface DashboardSummary {
  total_documents: number
  total_processed: number
  total_ignored: number
  total_failed: number
  avg_confidence: number
  avg_processing_time: number
  total_tags_detected: number
  models_registered: number
  models_approved: number
}

interface ProcessingTrend {
  date: string
  documents_processed: number
  avg_processing_time: number
  avg_confidence: number
}

interface RecentDocument {
  id: string
  filename: string
  status: string
  uploaded_at: string
}

function StatCard({ title, value, icon: Icon, color }: {
  title: string
  value: string | number
  icon: React.ElementType
  color: string
}) {
  return (
    <div className="rounded-lg bg-white p-6 shadow">
      <div className="flex items-center gap-4">
        <div className={`rounded-lg p-3 ${color}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-semibold">{value}</p>
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [filterType, setFilterType] = useState<string | null>(null)

  const { data: matterTypes } = useQuery<string[]>({
    queryKey: ['metrics', 'matter-types'],
    queryFn: () => api.get('/api/metrics/matter-types').then(r => r.data),
  })

  const { data: summary } = useQuery<DashboardSummary>({
    queryKey: ['metrics', 'summary', filterType],
    queryFn: () => api.get(`/api/metrics/summary${filterType ? `?matter_type=${encodeURIComponent(filterType)}` : ''}`).then(r => r.data),
  })

  const { data: trends } = useQuery<ProcessingTrend[]>({
    queryKey: ['metrics', 'processing', filterType],
    queryFn: () => api.get(`/api/metrics/processing?days=14${filterType ? `&matter_type=${encodeURIComponent(filterType)}` : ''}`).then(r => r.data),
  })

  const { data: recentDocs } = useQuery<RecentDocument[]>({
    queryKey: ['documents', 'recent'],
    queryFn: () => api.get('/api/documents?limit=5').then(r => r.data),
  })

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900">
          Dashboard
          {filterType && (
            <span className="ml-2 text-lg font-normal text-gray-500">- {filterType}</span>
          )}
        </h1>
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-400" />
          <select
            value={filterType || ''}
            onChange={(e) => setFilterType(e.target.value || null)}
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
          >
            <option value="">All Types</option>
            {matterTypes?.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          {filterType && (
            <button
              onClick={() => setFilterType(null)}
              className="text-gray-400 hover:text-gray-600"
              title="Clear filter"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <StatCard
          title="Total Documents"
          value={summary?.total_documents ?? 0}
          icon={FileText}
          color="bg-blue-500"
        />
        <StatCard
          title="Processed"
          value={summary?.total_processed ?? 0}
          icon={CheckCircle}
          color="bg-green-500"
        />
        <StatCard
          title="Ignored"
          value={summary?.total_ignored ?? 0}
          icon={EyeOff}
          color="bg-yellow-500"
        />
        <StatCard
          title="Failed"
          value={summary?.total_failed ?? 0}
          icon={AlertTriangle}
          color="bg-red-500"
        />
        <StatCard
          title="Avg Confidence"
          value={`${((summary?.avg_confidence ?? 0) * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color="bg-purple-500"
        />
        <StatCard
          title="Avg Time"
          value={`${(summary?.avg_processing_time ?? 0).toFixed(0)}s`}
          icon={Clock}
          color="bg-orange-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Processing Trends Chart */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-semibold mb-4">Processing Trends (14 days)</h2>
          <div className="h-64">
            {trends && trends.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={trends}>
                  <XAxis
                    dataKey="date"
                    tickFormatter={(v) => v.slice(5)}
                    fontSize={12}
                  />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="documents_processed" fill="#3b82f6" name="Documents" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-400">
                No data yet
              </div>
            )}
          </div>
        </div>

        {/* Recent Documents */}
        <div className="rounded-lg bg-white p-6 shadow">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Recent Documents</h2>
            <Link to="/documents" className="text-sm text-blue-600 hover:underline">
              View all
            </Link>
          </div>
          <div className="space-y-3">
            {recentDocs && recentDocs.length > 0 ? (
              recentDocs.map((doc) => (
                <Link
                  key={doc.id}
                  to={`/documents/${doc.id}`}
                  className="flex items-center justify-between rounded-lg border p-3 hover:bg-gray-50"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-gray-400" />
                    <span className="font-medium">{doc.filename}</span>
                  </div>
                  <span className={`text-sm px-2 py-1 rounded ${
                    doc.status === 'completed' ? 'bg-green-100 text-green-700' :
                    doc.status === 'failed' ? 'bg-red-100 text-red-700' :
                    doc.status === 'processing' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {doc.status}
                  </span>
                </Link>
              ))
            ) : (
              <div className="text-center text-gray-400 py-8">
                No documents yet. Upload one to get started!
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Stats */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard
          title="Models Registered"
          value={summary?.models_registered ?? 0}
          icon={Tag}
          color="bg-indigo-500"
        />
        <StatCard
          title="Models Approved"
          value={summary?.models_approved ?? 0}
          icon={CheckCircle}
          color="bg-teal-500"
        />
        <StatCard
          title="Total Tags Detected"
          value={summary?.total_tags_detected ?? 0}
          icon={Tag}
          color="bg-pink-500"
        />
      </div>
    </div>
  )
}
