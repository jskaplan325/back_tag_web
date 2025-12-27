import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts'
import { Clock, TrendingUp, Activity, Loader2, Filter, X } from 'lucide-react'
import api from '../api'

interface DashboardSummary {
  total_documents: number
  total_processed: number
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

interface ModelUsage {
  model_name: string
  model_type: string
  usage_count: number
  total_processing_time: number
  approved: boolean
}

interface LeaderboardEntry {
  rank: number
  document_name: string
  document_id: string
  matter_name: string | null
  processing_time: number
  word_count: number | null
  tag_count: number
  confidence: number | null
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

function StatCard({ title, value, icon: Icon, color, subtitle }: {
  title: string
  value: string | number
  icon: React.ElementType
  color: string
  subtitle?: string
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
          {subtitle && <p className="text-xs text-gray-400">{subtitle}</p>}
        </div>
      </div>
    </div>
  )
}

export default function Metrics() {
  const [filterType, setFilterType] = useState<string | null>(null)

  const { data: matterTypes } = useQuery<string[]>({
    queryKey: ['metrics', 'matter-types'],
    queryFn: () => api.get('/api/metrics/matter-types').then(r => r.data),
  })

  const { data: summary, isLoading: summaryLoading } = useQuery<DashboardSummary>({
    queryKey: ['metrics', 'summary', filterType],
    queryFn: () => api.get(`/api/metrics/summary${filterType ? `?matter_type=${encodeURIComponent(filterType)}` : ''}`).then(r => r.data),
  })

  const { data: trends, isLoading: trendsLoading } = useQuery<ProcessingTrend[]>({
    queryKey: ['metrics', 'processing', filterType],
    queryFn: () => api.get(`/api/metrics/processing?days=30${filterType ? `&matter_type=${encodeURIComponent(filterType)}` : ''}`).then(r => r.data),
  })

  const { data: modelUsage, isLoading: usageLoading } = useQuery<ModelUsage[]>({
    queryKey: ['metrics', 'models'],
    queryFn: () => api.get('/api/metrics/models').then(r => r.data),
  })

  const { data: leaderboard, isLoading: leaderboardLoading } = useQuery<LeaderboardEntry[]>({
    queryKey: ['metrics', 'leaderboard'],
    queryFn: () => api.get('/api/metrics/leaderboard?limit=10&sort_by=time').then(r => r.data),
  })

  const isLoading = summaryLoading || trendsLoading || usageLoading || leaderboardLoading

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900">
          Metrics & Analytics
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

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Avg Processing Time"
          value={`${(summary?.avg_processing_time || 0).toFixed(1)}s`}
          icon={Clock}
          color="bg-blue-500"
          subtitle="per document"
        />
        <StatCard
          title="Avg Confidence"
          value={`${((summary?.avg_confidence || 0) * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color="bg-green-500"
          subtitle="tag accuracy"
        />
        <StatCard
          title="Tags Detected"
          value={summary?.total_tags_detected?.toLocaleString() || 0}
          icon={TrendingUp}
          color="bg-purple-500"
          subtitle="total extractions"
        />
        <StatCard
          title="Success Rate"
          value={`${summary?.total_processed ? ((summary.total_processed / (summary.total_processed + (summary.total_failed || 0))) * 100).toFixed(0) : 0}%`}
          icon={Activity}
          color="bg-orange-500"
          subtitle={`${summary?.total_failed || 0} failed`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Processing Time Trend */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-semibold mb-4">Processing Time Trend (30 days)</h2>
          <div className="h-64">
            {trends && trends.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends}>
                  <XAxis
                    dataKey="date"
                    tickFormatter={(v) => v.slice(5)}
                    fontSize={12}
                  />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="avg_processing_time"
                    stroke="#3b82f6"
                    name="Avg Time (s)"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-400">
                No data yet
              </div>
            )}
          </div>
        </div>

        {/* Confidence Trend */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-semibold mb-4">Confidence Trend (30 days)</h2>
          <div className="h-64">
            {trends && trends.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends}>
                  <XAxis
                    dataKey="date"
                    tickFormatter={(v) => v.slice(5)}
                    fontSize={12}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    fontSize={12}
                  />
                  <Tooltip
                    formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="avg_confidence"
                    stroke="#10b981"
                    name="Avg Confidence"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-400">
                No data yet
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Model Usage */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-semibold mb-4">Model Usage</h2>
          <div className="h-64">
            {modelUsage && modelUsage.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={modelUsage}
                    dataKey="usage_count"
                    nameKey="model_name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label={({ model_name, percent }) =>
                      `${model_name.split('/').pop()}: ${(percent * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {modelUsage.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-400">
                No model usage data yet
              </div>
            )}
          </div>
        </div>

        {/* Processing Leaderboard */}
        <div className="rounded-lg bg-white p-6 shadow">
          <h2 className="text-lg font-semibold mb-4">Processing Leaderboard</h2>
          <div className="h-64 overflow-auto">
            {leaderboard && leaderboard.length > 0 ? (
              <table className="w-full text-sm">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-gray-500">#</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-500">Document</th>
                    <th className="px-3 py-2 text-right font-medium text-gray-500">Time</th>
                    <th className="px-3 py-2 text-right font-medium text-gray-500">Words</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {leaderboard.map((entry) => (
                    <tr key={entry.document_id} className="hover:bg-gray-50">
                      <td className="px-3 py-2 font-bold text-gray-400">{entry.rank}</td>
                      <td className="px-3 py-2">
                        <div className="truncate max-w-[180px]" title={entry.document_name}>
                          {entry.document_name.length > 25
                            ? entry.document_name.slice(0, 25) + '...'
                            : entry.document_name}
                        </div>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-blue-600">
                        {entry.processing_time >= 60
                          ? `${Math.floor(entry.processing_time / 60)}m ${(entry.processing_time % 60).toFixed(0)}s`
                          : `${entry.processing_time.toFixed(1)}s`}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-500">
                        {entry.word_count?.toLocaleString() || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="flex h-full items-center justify-center text-gray-400">
                No processing data yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Processing Stats Table */}
      <div className="mt-8 rounded-lg bg-white p-6 shadow">
        <h2 className="text-lg font-semibold mb-4">Model Performance</h2>
        {modelUsage && modelUsage.length > 0 ? (
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Model</th>
                <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Usage Count</th>
                <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Avg Time</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {modelUsage.map((m) => (
                <tr key={m.model_name}>
                  <td className="px-6 py-4 font-medium">{m.model_name}</td>
                  <td className="px-6 py-4 text-gray-500">{m.usage_count}</td>
                  <td className="px-6 py-4 text-gray-500">{m.total_processing_time.toFixed(1)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-center py-8 text-gray-400">
            No model performance data yet
          </div>
        )}
      </div>
    </div>
  )
}
