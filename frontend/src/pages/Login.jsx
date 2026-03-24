import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../App'
import { api } from '../lib/api'
import { Brain, Eye, EyeOff, ArrowRight, Shield } from 'lucide-react'
import { motion } from 'framer-motion'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const data = await api.login(email, password)
      login(
        { id: data.user_id, name: email.split('@')[0], email, role: data.role },
        data.access_token
      )
      navigate('/')
    } catch (err) {
      setError(err.message || 'Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  const handleDemoAdmin = () => {
    login(
      { id: 'demo-admin', name: 'Dr. Sarah Chen', email: 'admin@mindmesh.ai', role: 'admin' },
      'demo-token'
    )
    navigate('/dashboard')
  }

  const handleDemoStudent = () => {
    login(
      { id: 'demo-student', name: 'Arjun Mehta', email: 'arjun@school.edu', role: 'student' },
      'demo-token'
    )
    navigate('/journal')
  }

  return (
    <div className="login-page">
      <div className="login-bg" />

      {/* Floating particles */}
      <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            style={{
              position: 'absolute',
              width: 4 + i * 2,
              height: 4 + i * 2,
              borderRadius: '50%',
              background: `rgba(99,102,241,${0.1 + i * 0.04})`,
              left: `${15 + i * 14}%`,
              top: `${20 + (i % 3) * 25}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.3, 0.7, 0.3],
            }}
            transition={{
              duration: 4 + i,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: i * 0.5,
            }}
          />
        ))}
      </div>

      <motion.div
        className="login-card"
        initial={{ opacity: 0, y: 20, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="login-logo">
          <motion.div
            className="login-logo-icon"
            initial={{ rotate: -10 }}
            animate={{ rotate: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <Brain size={24} />
          </motion.div>
          <h1>
            Mind<span>Mesh</span> AI
          </h1>
        </div>

        <p style={{
          textAlign: 'center',
          color: 'var(--text-muted)',
          fontSize: '14px',
          marginBottom: 'var(--space-xl)',
          lineHeight: 1.5,
        }}>
          AI-powered mental health intelligence<br />for school ecosystems
        </p>

        {error && (
          <motion.div
            className="alert-banner danger"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
          >
            <Shield size={16} />
            {error}
          </motion.div>
        )}

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="input-group">
            <label className="input-label">Email</label>
            <input
              className="input"
              type="email"
              placeholder="admin@mindmesh.ai"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
            />
          </div>

          <div className="input-group">
            <label className="input-label">Password</label>
            <div style={{ position: 'relative' }}>
              <input
                className="input"
                type={showPassword ? 'text' : 'password'}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                style={{ paddingRight: 44 }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: 'absolute',
                  right: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  padding: 4,
                }}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary login-btn"
            disabled={loading}
          >
            {loading ? (
              <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
            ) : (
              <>
                Sign In
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          margin: '24px 0 16px',
        }}>
          <div style={{ flex: 1, height: 1, background: 'var(--border-subtle)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>or</span>
          <div style={{ flex: 1, height: 1, background: 'var(--border-subtle)' }} />
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ flex: 1 }}
            onClick={handleDemoAdmin}
          >
            <Shield size={14} />
            Counselor Demo
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ flex: 1 }}
            onClick={handleDemoStudent}
          >
            <Brain size={14} />
            Student Demo
          </button>
        </div>

        <div className="login-footer">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 8 }}>
            <Shield size={12} />
            SIH 1433 — Mental Health Intelligence System
          </div>
        </div>
      </motion.div>
    </div>
  )
}
