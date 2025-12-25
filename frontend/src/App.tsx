import { Routes, Route, Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  FileText,
  Box,
  BarChart3,
  Upload
} from 'lucide-react'
import clsx from 'clsx'

// Pages
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import DocumentViewer from './pages/DocumentViewer'
import Models from './pages/Models'
import ModelDetail from './pages/ModelDetail'
import Metrics from './pages/Metrics'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Documents', href: '/documents', icon: FileText },
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
      <div className="p-4">
        <Link
          to="/documents?upload=true"
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          <Upload className="h-4 w-4" />
          Upload Document
        </Link>
      </div>
    </div>
  )
}

function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/documents/:id" element={<DocumentViewer />} />
          <Route path="/models" element={<Models />} />
          <Route path="/models/:id" element={<ModelDetail />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
