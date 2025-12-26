import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  ExternalLink,
  Download,
  Heart,
  Clock,
  Shield,
  AlertTriangle,
  BookOpen,
  Database,
  Settings,
  Check,
  RefreshCw,
  Loader2,
  Tag,
  Building,
  GitBranch,
  Scale
} from 'lucide-react'
import clsx from 'clsx'
import api from '../api'

interface ModelCardDetail {
  id: string
  name: string
  type: string
  huggingface_url: string | null
  size_gb: number | null
  downloads: number | null
  license: string | null
  last_updated: string | null
  approved: boolean
  approved_by: string | null
  approved_at: string | null
  created_at: string
  usage_count: number
  executive_summary: string | null
  intended_uses: string | null
  limitations: string | null
  training_data: string | null
  training_procedure: string | null
  tags: string[]
  pipeline_tag: string | null
  library_name: string | null
  likes: number | null
}

function Section({
  title,
  icon: Icon,
  children,
  variant = 'default'
}: {
  title: string
  icon: React.ElementType
  children: React.ReactNode
  variant?: 'default' | 'warning' | 'info'
}) {
  return (
    <div className={clsx(
      'rounded-lg border p-6',
      variant === 'warning' && 'border-yellow-200 bg-yellow-50',
      variant === 'info' && 'border-blue-200 bg-blue-50',
      variant === 'default' && 'border-gray-200 bg-white'
    )}>
      <div className="flex items-center gap-2 mb-4">
        <Icon className={clsx(
          'h-5 w-5',
          variant === 'warning' && 'text-yellow-600',
          variant === 'info' && 'text-blue-600',
          variant === 'default' && 'text-gray-600'
        )} />
        <h3 className="font-semibold text-lg">{title}</h3>
      </div>
      <div className="prose prose-sm max-w-none">
        {children}
      </div>
    </div>
  )
}

function MarkdownContent({ content }: { content: string }) {
  // Simple markdown rendering - convert basic markdown to HTML
  const renderMarkdown = (text: string) => {
    return text
      // Code blocks
      .replace(/```[\s\S]*?```/g, (match) => {
        const code = match.replace(/```\w*\n?/g, '').trim()
        return `<pre class="bg-gray-100 p-3 rounded text-xs overflow-x-auto"><code>${code}</code></pre>`
      })
      // Inline code
      .replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>')
      // Bold
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" class="text-blue-600 hover:underline">$1</a>')
      // Line breaks
      .replace(/\n\n/g, '</p><p class="mt-3">')
      .replace(/\n/g, '<br/>')
  }

  return (
    <div
      className="text-gray-700 leading-relaxed"
      dangerouslySetInnerHTML={{ __html: `<p>${renderMarkdown(content)}</p>` }}
    />
  )
}

export default function ModelDetail() {
  const { id } = useParams<{ id: string }>()
  const queryClient = useQueryClient()

  const { data: model, isLoading, error } = useQuery<ModelCardDetail>({
    queryKey: ['model', id, 'card'],
    queryFn: () => api.get(`/api/models/${id}/card`).then(r => r.data),
  })

  const approveMutation = useMutation({
    mutationFn: () => api.patch(`/api/models/${id}`, { approved: true, approved_by: 'admin' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model', id] })
    },
  })

  const refreshMutation = useMutation({
    mutationFn: () => api.post(`/api/models/${id}/refresh`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['model', id] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  if (error || !model) {
    return (
      <div className="p-8">
        <Link to="/models" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Models
        </Link>
        <div className="text-center py-12">
          <p className="text-red-600">Failed to load model details</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <Link to="/models" className="flex items-center gap-2 text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft className="h-4 w-4" />
          Back to Models
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{model.name.split('/').pop()}</h1>
            <p className="text-gray-500 mt-1">{model.name.split('/')[0]}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => refreshMutation.mutate()}
              disabled={refreshMutation.isPending}
              className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              <RefreshCw className={clsx('h-4 w-4', refreshMutation.isPending && 'animate-spin')} />
              Refresh
            </button>
            {!model.approved && (
              <button
                onClick={() => approveMutation.mutate()}
                disabled={approveMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                <Check className="h-4 w-4" />
                Approve Model
              </button>
            )}
          </div>
        </div>

        {/* Status badges */}
        <div className="flex flex-wrap items-center gap-2 mt-4">
          <span className={clsx(
            'px-3 py-1 rounded-full text-sm font-medium',
            model.approved ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
          )}>
            {model.approved ? 'RAI Approved' : 'Pending Review'}
          </span>
          <span className={clsx(
            'px-3 py-1 rounded-full text-sm font-medium',
            model.type === 'semantic' && 'bg-blue-100 text-blue-700',
            model.type === 'vision' && 'bg-purple-100 text-purple-700',
            model.type === 'ocr' && 'bg-green-100 text-green-700'
          )}>
            {model.type}
          </span>
          {model.pipeline_tag && (
            <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700">
              {model.pipeline_tag}
            </span>
          )}
          {model.library_name && (
            <span className="px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-700">
              {model.library_name}
            </span>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Download className="h-4 w-4" />
            Downloads
          </div>
          <p className="text-2xl font-semibold mt-1">{model.downloads?.toLocaleString() || '-'}</p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Heart className="h-4 w-4" />
            Likes
          </div>
          <p className="text-2xl font-semibold mt-1">{model.likes?.toLocaleString() || '-'}</p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Clock className="h-4 w-4" />
            Usage Count
          </div>
          <p className="text-2xl font-semibold mt-1">{model.usage_count}</p>
        </div>
        <div className="bg-white rounded-lg border p-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Database className="h-4 w-4" />
            Size
          </div>
          <p className="text-2xl font-semibold mt-1">{model.size_gb ? `${model.size_gb} GB` : '-'}</p>
        </div>
      </div>

      {/* Tags */}
      {model.tags && model.tags.length > 0 && (
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Tag className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-500">Tags</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {model.tags.map((tag) => (
              <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Main Content Sections */}
      <div className="space-y-6">
        {/* Executive Summary */}
        {model.executive_summary && (
          <Section title="Executive Summary" icon={BookOpen}>
            <MarkdownContent content={model.executive_summary} />
          </Section>
        )}

        {/* Intended Uses */}
        {model.intended_uses && (
          <Section title="Intended Uses & Capabilities" icon={Settings} variant="info">
            <MarkdownContent content={model.intended_uses} />
          </Section>
        )}

        {/* Limitations & Security (Important!) */}
        {model.limitations && (
          <Section title="Limitations, Bias & Security Considerations" icon={AlertTriangle} variant="warning">
            <MarkdownContent content={model.limitations} />
          </Section>
        )}

        {/* Training Data */}
        {model.training_data && (
          <Section title="Training Data" icon={Database}>
            <MarkdownContent content={model.training_data} />
          </Section>
        )}

        {/* Training Procedure */}
        {model.training_procedure && (
          <Section title="Training Procedure" icon={Settings}>
            <MarkdownContent content={model.training_procedure} />
          </Section>
        )}

        {/* Origin & Provenance */}
        <Section title="Origin & Provenance" icon={Building}>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-gray-500 text-sm">Organization</span>
                <p className="font-medium">{model.name.split('/')[0] || 'Unknown'}</p>
              </div>
              <div>
                <span className="text-gray-500 text-sm">Model Name</span>
                <p className="font-medium">{model.name.split('/').pop()}</p>
              </div>
              <div>
                <span className="text-gray-500 text-sm">Full Identifier</span>
                <p className="font-mono text-sm">{model.name}</p>
              </div>
              {model.library_name && (
                <div>
                  <span className="text-gray-500 text-sm">Library</span>
                  <p className="font-medium">{model.library_name}</p>
                </div>
              )}
            </div>

            <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-start gap-2">
                <Scale className="h-4 w-4 text-blue-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-800">Responsible AI Review Note</p>
                  <p className="text-blue-700 mt-1">
                    Before approving, verify the organization's reputation, check for known issues
                    with the model, and review any geopolitical considerations related to the
                    model's origin and funding sources.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Section>

        {/* Architecture Overview (if available from tags) */}
        {model.pipeline_tag && (
          <Section title="Architecture" icon={GitBranch}>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Pipeline:</span>
                <span className="font-medium">{model.pipeline_tag}</span>
              </div>
              {model.library_name && (
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Framework:</span>
                  <span className="font-medium">{model.library_name}</span>
                </div>
              )}
              {model.tags && model.tags.some(t => t.includes('transformer')) && (
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Architecture Type:</span>
                  <span className="font-medium">Transformer-based</span>
                </div>
              )}
              <p className="text-sm text-gray-500 mt-2">
                For detailed architecture information, view the full model card on HuggingFace.
              </p>
            </div>
          </Section>
        )}

        {/* License Info */}
        <Section title="License & Legal" icon={Shield}>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">License:</span>
              <span className="font-medium">{model.license || 'Not specified'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Last Updated:</span>
              <span className="font-medium">
                {model.last_updated ? new Date(model.last_updated).toLocaleDateString() : '-'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Added to Registry:</span>
              <span className="font-medium">
                {new Date(model.created_at).toLocaleDateString()}
              </span>
            </div>
            {model.approved_at && (
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Approved:</span>
                <span className="font-medium">
                  {new Date(model.approved_at).toLocaleDateString()} by {model.approved_by}
                </span>
              </div>
            )}
          </div>
        </Section>

        {/* No content fallback */}
        {!model.executive_summary && !model.intended_uses && !model.limitations && (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <p className="text-gray-500">No detailed model card available from HuggingFace.</p>
            <a
              href={model.huggingface_url || '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 mt-4 text-blue-600 hover:underline"
            >
              View on HuggingFace
              <ExternalLink className="h-4 w-4" />
            </a>
          </div>
        )}
      </div>

      {/* HuggingFace Link */}
      <div className="mt-8 pt-6 border-t">
        <a
          href={model.huggingface_url?.startsWith('http')
            ? model.huggingface_url
            : `https://huggingface.co/${model.name}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-blue-600 hover:underline"
        >
          View full model card on HuggingFace
          <ExternalLink className="h-4 w-4" />
        </a>
      </div>
    </div>
  )
}
