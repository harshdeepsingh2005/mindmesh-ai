import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useState, createContext, useContext } from 'react'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Students from './pages/Students'
import Alerts from './pages/Alerts'
import Analytics from './pages/Analytics'
import Models from './pages/Models'
import Journal from './pages/Journal'
import StudentProgress from './pages/StudentProgress'
import ReportConcern from './pages/ReportConcern'
import Sidebar from './components/Sidebar'
import SOSModule from './components/SOSModule'
import './index.css'

export const AuthContext = createContext(null)

export function useAuth() {
  return useContext(AuthContext)
}

function ProtectedLayout({ children }) {
  const { user } = useAuth()
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        {children}
      </main>
      {user?.role === 'student' && <SOSModule />}
    </div>
  )
}

function ProtectedRoute({ children }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  return <ProtectedLayout>{children}</ProtectedLayout>
}

/**
 * RoleRoute — Restrict access by role.
 * If the user's role is not in `allowedRoles`, redirect them:
 *   - Students → /journal
 *   - Admins  → /
 */
function RoleRoute({ allowedRoles, children }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />

  if (!allowedRoles.includes(user.role)) {
    // Redirect to their default page
    const defaultPath = user.role === 'student' ? '/journal' : '/'
    return <Navigate to={defaultPath} replace />
  }

  return <ProtectedLayout>{children}</ProtectedLayout>
}

/**
 * DefaultRedirect — Send user to the right landing page based on role.
 */
function DefaultRedirect() {
  const { user } = useAuth()
  if (!user) return <Navigate to="/login" replace />
  if (user.role === 'student') return <Navigate to="/journal" replace />
  return <Navigate to="/dashboard" replace />
}

export default function App() {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('mindmesh_user')
    return saved ? JSON.parse(saved) : null
  })

  const [token, setToken] = useState(() => {
    return localStorage.getItem('mindmesh_token') || null
  })

  const login = (userData, accessToken) => {
    setUser(userData)
    setToken(accessToken)
    localStorage.setItem('mindmesh_user', JSON.stringify(userData))
    localStorage.setItem('mindmesh_token', accessToken)
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('mindmesh_user')
    localStorage.removeItem('mindmesh_token')
  }

  const STAFF_ROLES = ['admin', 'teacher', 'counselor']

  return (
    <AuthContext.Provider value={{ user, token, login, logout }}>
      <BrowserRouter>
        <Routes>
          {/* Public */}
          <Route path="/login" element={user ? <DefaultRedirect /> : <Login />} />

          {/* ─── Admin / Teacher / Counselor Portal ────────── */}
          <Route path="/dashboard" element={
            <RoleRoute allowedRoles={STAFF_ROLES}><Dashboard /></RoleRoute>
          } />
          <Route path="/students" element={
            <RoleRoute allowedRoles={STAFF_ROLES}><Students /></RoleRoute>
          } />
          <Route path="/alerts" element={
            <RoleRoute allowedRoles={STAFF_ROLES}><Alerts /></RoleRoute>
          } />
          <Route path="/analytics" element={
            <RoleRoute allowedRoles={STAFF_ROLES}><Analytics /></RoleRoute>
          } />
          <Route path="/models" element={
            <RoleRoute allowedRoles={['admin']}><Models /></RoleRoute>
          } />

          {/* ─── Student Portal ────────────────────────────── */}
          <Route path="/journal" element={
            <RoleRoute allowedRoles={['student', ...STAFF_ROLES]}><Journal /></RoleRoute>
          } />
          <Route path="/checkin" element={
            <RoleRoute allowedRoles={['student', ...STAFF_ROLES]}><Journal /></RoleRoute>
          } />
          <Route path="/my-progress" element={
            <RoleRoute allowedRoles={['student', ...STAFF_ROLES]}><StudentProgress /></RoleRoute>
          } />
          <Route path="/report-concern" element={
            <RoleRoute allowedRoles={['student', ...STAFF_ROLES]}><ReportConcern /></RoleRoute>
          } />

          {/* ─── Default redirect based on role ────────────── */}
          <Route path="/" element={<DefaultRedirect />} />
          <Route path="*" element={<DefaultRedirect />} />
        </Routes>
      </BrowserRouter>
    </AuthContext.Provider>
  )
}
