import { Routes, Route, Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  FileText,
  Box,
  BarChart3,
  Scale,
  FolderOpen,
  AlertCircle,
  AlertTriangle,
  Database
} from 'lucide-react'
import clsx from 'clsx'
import { DEMO_MODE } from './api'

// Pages
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import DocumentViewer from './pages/DocumentViewer'
import Models from './pages/Models'
import ModelDetail from './pages/ModelDetail'
import Metrics from './pages/Metrics'
import Taxonomy from './pages/Taxonomy'
import TaxonomyDetail from './pages/TaxonomyDetail'
import Matters from './pages/Matters'
import MatterDetail from './pages/MatterDetail'
import MattersByType from './pages/MattersByType'
import ReviewQueue from './pages/ReviewQueue'
import TrainingDashboard from './pages/TrainingDashboard'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Documents', href: '/documents', icon: FileText },
  { name: 'Matters', href: '/matters', icon: FolderOpen },
  { name: 'Review Queue', href: '/review', icon: AlertTriangle },
  { name: 'Training Data', href: '/training', icon: Database },
  { name: 'Taxonomy', href: '/taxonomy', icon: Scale },
  { name: 'Models', href: '/models', icon: Box },
  { name: 'Metrics', href: '/metrics', icon: BarChart3 },
]

function Sidebar() {
  const location = useLocation()

  return (
    <div className="flex h-screen w-64 flex-col bg-gray-900">
      <div className="flex h-16 items-center px-6">
        <h1 className="text-xl font-bold text-white">Doc Intelligence</h1>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href ||
            (item.href !== '/' && location.pathname.startsWith(item.href))
          return (
            <Link
              key={item.name}
              to={item.href}
              className={clsx(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>
      <div className="px-4 py-3 border-t border-gray-800">
        <p className="text-xs text-gray-500">v0.2.0</p>
      </div>
    </div>
  )
}

function DemoBanner() {
  if (!DEMO_MODE) return null
  return (
    <div className="bg-amber-500 text-amber-950 px-4 py-2 flex items-center justify-center gap-2 text-sm font-medium">
      <AlertCircle className="h-4 w-4" />
      <span>Demo Mode - Viewing sample data. Connect to backend for full functionality.</span>
    </div>
  )
}

function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DemoBanner />
        <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/documents/:id" element={<DocumentViewer />} />
          <Route path="/matters" element={<Matters />} />
          <Route path="/matters/type/:type" element={<MattersByType />} />
          <Route path="/matters/:id" element={<MatterDetail />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/training" element={<TrainingDashboard />} />
          <Route path="/taxonomy" element={<Taxonomy />} />
          <Route path="/taxonomy/:id" element={<TaxonomyDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/:id" element={<ModelDetail />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
