import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { AuthProvider, useAuth } from './context/AuthContext'
import LandingPage  from './pages/LandingPage'
import AuthPage     from './pages/AuthPage'
import Dashboard    from './pages/Dashboard'
import GeneratePage from './pages/GeneratePage'
import RoadmapPage  from './pages/RoadmapPage'
import HistoryPage  from './pages/HistoryPage'
import Layout       from './components/Layout'

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingScreen />
  if (!user)   return <Navigate to="/auth" replace />
  return children
}

function LoadingScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-screen gap-4"
         style={{ background: 'var(--bg-primary)' }}>
      <div
        className="w-10 h-10 rounded-full"
        style={{
          border: '2px solid var(--border-subtle)',
          borderTopColor: 'var(--accent-green)',
          animation: 'spin 0.8s linear infinite',
        }}
      />
      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '13px' }}>
        initializing...
      </span>
    </div>
  )
}

function AppRoutes() {
  const { user } = useAuth()
  return (
    <Routes>
      <Route path="/"    element={user ? <Navigate to="/dashboard" /> : <LandingPage />} />
      <Route path="/auth" element={user ? <Navigate to="/dashboard" /> : <AuthPage />} />
      <Route path="/dashboard" element={
        <ProtectedRoute><Layout><Dashboard /></Layout></ProtectedRoute>
      } />
      <Route path="/generate" element={
        <ProtectedRoute><Layout><GeneratePage /></Layout></ProtectedRoute>
      } />
      <Route path="/roadmap/:id" element={
        <ProtectedRoute><Layout><RoadmapPage /></Layout></ProtectedRoute>
      } />
      <Route path="/history" element={
        <ProtectedRoute><Layout><HistoryPage /></Layout></ProtectedRoute>
      } />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'var(--bg-elevated)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-medium)',
              fontFamily: 'var(--font-body)',
              fontSize: '14px',
            },
            success: { iconTheme: { primary: 'var(--accent-green)', secondary: 'var(--bg-elevated)' } },
            error:   { iconTheme: { primary: 'var(--accent-red)',   secondary: 'var(--bg-elevated)' } },
          }}
        />
      </AuthProvider>
    </BrowserRouter>
  )
}