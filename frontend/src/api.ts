import axios, { AxiosResponse } from 'axios'
import {
  DEMO_MODE,
  demoMetricsSummary,
  demoMatters,
  demoDocuments,
  demoModels,
  demoTaxonomy,
  demoProcessingTrends,
  demoModelUsage,
  demoMatterTypes,
  demoTaxonomyScoreboard,
  demoDocumentDetail,
  demoMatterDetail,
} from './demoData'

// Create mock response helper
const mockResponse = <T>(data: T): Promise<AxiosResponse<T>> => {
  return Promise.resolve({
    data,
    status: 200,
    statusText: 'OK',
    headers: {},
    config: {} as any,
  })
}

// Demo API that returns mock data
const demoApi = {
  get: (url: string): Promise<AxiosResponse<any>> => {
    // Metrics
    if (url.includes('/api/metrics/summary')) return mockResponse(demoMetricsSummary)
    if (url.includes('/api/metrics/processing')) return mockResponse(demoProcessingTrends)
    if (url.includes('/api/metrics/models')) return mockResponse(demoModelUsage)
    if (url.includes('/api/metrics/matter-types')) return mockResponse(demoMatterTypes)

    // Matters
    if (url.match(/\/api\/matters\/[^/]+\/tags/)) {
      return mockResponse(demoMatterDetail.top_tags)
    }
    if (url.match(/\/api\/matters\/[^/]+$/)) {
      const id = url.split('/').pop()
      const matter = demoMatters.find(m => m.id === id) || demoMatterDetail
      return mockResponse({
        ...matter,
        documents: demoDocuments.filter(d => d.matter_id === id),
      })
    }
    if (url.includes('/api/matters/stats/by-type')) return mockResponse(demoMatterTypes)
    if (url.includes('/api/matters')) return mockResponse(demoMatters)

    // Documents
    if (url.match(/\/api\/documents\/[^/]+\/text/)) {
      return mockResponse({ text: demoDocumentDetail.text_preview })
    }
    if (url.match(/\/api\/documents\/[^/]+\/tags/)) {
      return mockResponse(demoDocumentDetail.tags)
    }
    if (url.match(/\/api\/documents\/[^/]+$/)) {
      const id = url.split('/').pop()
      const doc = demoDocuments.find(d => d.id === id) || demoDocumentDetail
      return mockResponse({ ...doc, tags: demoDocumentDetail.tags })
    }
    if (url.includes('/api/documents')) return mockResponse(demoDocuments)

    // Models
    if (url.match(/\/api\/models\/[^/]+\/usage/)) {
      return mockResponse({ total_usages: 231, recent_usages: [] })
    }
    if (url.match(/\/api\/models\/[^/]+$/)) {
      const id = url.split('/').pop()
      return mockResponse(demoModels.find(m => m.id === id) || demoModels[0])
    }
    if (url.includes('/api/models')) return mockResponse(demoModels)

    // Taxonomy
    if (url.includes('/api/taxonomy/scoreboard')) return mockResponse(demoTaxonomyScoreboard)
    if (url.match(/\/api\/taxonomy\/[^/]+$/)) {
      const id = url.split('/').pop()
      return mockResponse(demoTaxonomy.find(t => t.id === id) || demoTaxonomy[0])
    }
    if (url.includes('/api/taxonomy')) return mockResponse(demoTaxonomy)

    // Root health check
    if (url === '/' || url === '') return mockResponse({ status: 'demo' })

    // Default
    console.log('[Demo Mode] Unhandled GET:', url)
    return mockResponse({})
  },

  post: (url: string, data?: any): Promise<AxiosResponse<any>> => {
    console.log('[Demo Mode] POST blocked:', url, data)
    return mockResponse({ status: 'demo_mode', message: 'Write operations disabled in demo' })
  },

  patch: (url: string, data?: any): Promise<AxiosResponse<any>> => {
    console.log('[Demo Mode] PATCH blocked:', url, data)
    return mockResponse({ status: 'demo_mode', message: 'Write operations disabled in demo' })
  },

  delete: (url: string): Promise<AxiosResponse<any>> => {
    console.log('[Demo Mode] DELETE blocked:', url)
    return mockResponse({ status: 'demo_mode', message: 'Write operations disabled in demo' })
  },
}

// Real API for local development
const realApi = axios.create({
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Export the appropriate API based on environment
const api = DEMO_MODE ? demoApi : realApi

export default api

// Export demo mode flag for components that need to show demo banner
export { DEMO_MODE }
