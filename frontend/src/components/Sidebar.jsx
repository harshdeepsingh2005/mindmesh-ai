import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import {
  LayoutDashboard, Users, Bell, BarChart3, Brain, BookOpen,
  LogOut, Shield, Activity, SmilePlus, User, Home
} from 'lucide-react'

// ── Role-based navigation config ───────────────────────────
const ADMIN_NAV = [
  { section: 'Overview' },
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', end: true },
  { to: '/students', icon: Users, label: 'Students' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { section: 'Intelligence' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/models', icon: Brain, label: 'ML Models' },
]

const STUDENT_NAV = [
  { section: 'My Portal' },
  { to: '/journal', icon: BookOpen, label: 'Journal', end: true },
  { to: '/checkin', icon: SmilePlus, label: 'Mood Check-in' },
  { to: '/my-progress', icon: Activity, label: 'My Progress' },
  { section: 'Support' },
  { to: '/report-concern', icon: Shield, label: 'Report Concern' },
]

function getNavItems(role) {
  if (role === 'student') return STUDENT_NAV
  // admin, teacher, counselor all get admin nav
  return ADMIN_NAV
}

export default function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const initials = user?.name
    ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : 'U'

  const navItems = getNavItems(user?.role)

  const roleLabel = {
    admin: 'Administrator',
    teacher: 'Teacher',
    counselor: 'Counselor',
    student: 'Student',
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <Brain size={20} />
        </div>
        <div className="sidebar-brand">
          <h1>MindMesh AI</h1>
          <span>
            {user?.role === 'student' ? 'Student Wellness' : 'Mental Health Intelligence'}
          </span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item, i) => {
          if (item.section) {
            return (
              <span key={`section-${i}`} className="sidebar-section-label">
                {item.section}
              </span>
            )
          }

          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} />
              {item.label}
            </NavLink>
          )
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="sidebar-avatar">{initials}</div>
          <div className="sidebar-user-info">
            <span className="name">{user?.name || 'User'}</span>
            <span className="role">{roleLabel[user?.role] || user?.role || 'User'}</span>
          </div>
          <button
            className="btn btn-ghost btn-icon"
            onClick={handleLogout}
            title="Log out"
            style={{ marginLeft: 'auto' }}
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </aside>
  )
}
