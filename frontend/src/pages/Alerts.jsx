import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bell, AlertTriangle, CheckCircle, Clock, Shield, X, ChevronDown, Brain } from 'lucide-react'
import { api } from '../lib/api'

// Added SOS mock alert to show the feature
const MOCK_ALERTS = [
  { id: 'a0', student_name: 'John Doe', student_id: 's-0', risk_score: 100, alert_type: 'sos', message: 'SOS TRIGGERED. Location: Campus library. Notes: Requested via emergency SOS button', created_at: new Date().toISOString(), status: 'pending' },
  { id: 'pee1', student_name: 'Jane Smith', student_id: 's-pee1', risk_score: 85, alert_type: 'peer_concern', message: 'Peer Concern Reported: I saw them crying in the hallway and talking about dropping out.', created_at: new Date(Date.now() - 3600000).toISOString(), status: 'pending' },
  { id: 'a1', student_name: 'Arjun Mehta', student_id: 's-1', risk_score: 92, alert_type: 'high_risk', message: 'HIGH RISK CONTENT DETECTED in journal. Keywords found: self harm. Immediate counselor intervention required.', created_at: new Date(Date.now() - 2 * 3600000).toISOString(), status: 'pending' },
  { id: 'a2', student_name: 'Priya Sharma', student_id: 's-2', risk_score: 85, alert_type: 'high_risk', message: 'Anomaly detected: Isolation Forest flagged significant behavioral deviation. Emotional volatility increased 340% over baseline.', created_at: new Date(Date.now() - 5 * 3600000).toISOString(), status: 'pending' },
  { id: 'a3', student_name: 'Rohan Patel', student_id: 's-3', risk_score: 78, alert_type: 'high_risk', message: 'Sustained negative sentiment trend detected over 14 days. VADER compound score declining at -0.04/day.', created_at: new Date(Date.now() - 12 * 3600000).toISOString(), status: 'acknowledged' },
  { id: 'a4', student_name: 'Sneha Gupta', student_id: 's-4', risk_score: 74, alert_type: 'high_risk', message: 'Disengagement detected: Check-in frequency dropped 65% compared to prior 30-day baseline.', created_at: new Date(Date.now() - 24 * 3600000).toISOString(), status: 'acknowledged' },
  { id: 'a5', student_name: 'Vikram Singh', student_id: 's-5', risk_score: 71, alert_type: 'high_risk', message: 'K-Means cluster shift: Student moved from "positive" to "distress" emotion cluster in latest analysis window.', created_at: new Date(Date.now() - 48 * 3600000).toISOString(), status: 'resolved' },
  { id: 'a6', student_name: 'Ananya Das', student_id: 's-6', risk_score: 68, alert_type: 'info', message: 'Student behavioral profile reclassified by GMM clustering. New profile: "withdrawn-declining". Previous: "stable-engaged".', created_at: new Date(Date.now() - 72 * 3600000).toISOString(), status: 'resolved' },
  { id: 'a7', student_name: 'Karthik Reddy', student_id: 's-7', risk_score: 65, alert_type: 'info', message: 'NMF topic analysis detected recurring theme: "loneliness" appearing in 4 of last 5 journal entries.', created_at: new Date(Date.now() - 96 * 3600000).toISOString(), status: 'dismissed' },
]

function timeAgo(date) {
  const diff = Date.now() - new Date(date).getTime()
  const hours = Math.floor(diff / 3600000)
  if (hours < 1) return 'Just now'
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

export default function Alerts() {
  const [alerts, setAlerts] = useState(MOCK_ALERTS)
  const [statusFilter, setStatusFilter] = useState('all')
  const [expandedId, setExpandedId] = useState(null)
  const [primers, setPrimers] = useState({})
  const [loadingPrimer, setLoadingPrimer] = useState(null)

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getAlerts({ limit: 50 })
        if (data?.alerts?.length) setAlerts(data.alerts)
      } catch { /* keep mock */ }
    })()
  }, [])

  const handleExpand = async (id, alertType, message) => {
    if (expandedId === id) {
      setExpandedId(null)
      return
    }
    setExpandedId(id)
    
    if (!primers[id]) {
      setLoadingPrimer(id)
      try {
        const data = await api.getAlertPrimer(id)
        setPrimers(prev => ({ ...prev, [id]: data.primer }))
      } catch (err) {
        // Fallback mock primer just in case backend isn't running or endpoint fails
        let mockPrimer = ""
        if (alertType.toLowerCase() === "sos") {
          mockPrimer = "1. IMMEDIATE ACTION: Contact Campus Security and dispatch to student's location immediately.\n2. Keep the student on the line if they call; do not hang up.\n3. Ask direct questions: 'Are you safe right now?' 'Do you have a plan?'\n4. Validate their pain without judgment: 'I hear how overwhelmed you are.'"
        } else if (alertType.toLowerCase() === "peer_concern") {
          mockPrimer = "1. DISCREET CHECK-IN: Reach out to the student for a 'routine' check-in.\n2. Do NOT mention the peer who reported them to maintain trust.\n3. Open-ended questions: 'How have things been balancing lately?'\n4. Look for signs of the reported concern in their body language."
        } else {
          mockPrimer = "1. Approach with empathy: 'I noticed from your recent check-ins that things seem heavy right now.'\n2. Focus on their specific stressor mentioned in the alert: " + message.substring(0, 50) + "...\n3. Avoid toxic positivity (don't say 'it will get better'). Instead, say 'That sounds really difficult.'\n4. Collaborate on a small, actionable next step for today."
        }
        setPrimers(prev => ({ ...prev, [id]: mockPrimer }))
      } finally {
        setLoadingPrimer(null)
      }
    }
  }

  const filtered = alerts.filter(a =>
    statusFilter === 'all' || a.status === statusFilter
  )

  const handleAcknowledge = async (id) => {
    try { await api.acknowledgeAlert(id) } catch {}
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: 'acknowledged' } : a))
  }

  const handleResolve = async (id) => {
    try { await api.resolveAlert(id) } catch {}
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, status: 'resolved' } : a))
  }

  const statusCounts = {
    all: alerts.length,
    pending: alerts.filter(a => a.status === 'pending').length,
    acknowledged: alerts.filter(a => a.status === 'acknowledged').length,
    resolved: alerts.filter(a => a.status === 'resolved').length,
  }

  const statusStyles = {
    pending: { color: 'var(--color-danger)', bg: 'var(--color-danger-bg)', icon: AlertTriangle, label: 'Pending' },
    acknowledged: { color: 'var(--color-warning)', bg: 'var(--color-warning-bg)', icon: Clock, label: 'Acknowledged' },
    resolved: { color: 'var(--color-success)', bg: 'var(--color-success-bg)', icon: CheckCircle, label: 'Resolved' },
    dismissed: { color: 'var(--text-muted)', bg: 'var(--bg-tertiary)', icon: X, label: 'Dismissed' },
  }

  return (
    <div>
      <div className="page-header page-header-row">
        <div>
          <h2>Alerts</h2>
          <p>AI-generated risk alerts requiring counselor attention</p>
        </div>
        {statusCounts.pending > 0 && (
          <div className="alert-banner danger" style={{ marginBottom: 0, padding: '8px 16px' }}>
            <Shield size={16} />
            <span style={{ fontWeight: 600 }}>{statusCounts.pending} alerts need immediate attention</span>
          </div>
        )}
      </div>

      {/* Status Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 'var(--space-lg)', flexWrap: 'wrap' }}>
        {['all', 'pending', 'acknowledged', 'resolved'].map(status => (
          <button
            key={status}
            className={`btn btn-sm ${statusFilter === status ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setStatusFilter(status)}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
            <span style={{ marginLeft: 6, fontFamily: 'var(--font-mono)', fontSize: 11, opacity: 0.8 }}>
              {statusCounts[status] || 0}
            </span>
          </button>
        ))}
      </div>

      {/* Alert Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <AnimatePresence mode="popLayout">
          {filtered.map((alert, i) => {
            const style = statusStyles[alert.status] || statusStyles.pending
            const StatusIcon = style.icon
            const expanded = expandedId === alert.id

            return (
              <motion.div
                key={alert.id}
                className="card"
                layout
                {...anim}
                transition={{ ...anim.transition, delay: i * 0.04 }}
                exit={{ opacity: 0, scale: 0.95 }}
                style={{
                  borderLeft: `3px solid ${style.color}`,
                  cursor: 'pointer',
                }}
                onClick={() => handleExpand(alert.id, alert.alert_type, alert.message)}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-md)' }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 'var(--radius-sm)',
                    background: style.bg, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: style.color, flexShrink: 0,
                  }}>
                    <StatusIcon size={18} />
                  </div>

                  <div style={{ flex: 1 }}>
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      marginBottom: 6,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span style={{ fontWeight: 700, fontSize: 14 }}>
                          {alert.student_name || alert.student_id}
                        </span>
                        <span className={`risk-badge ${alert.risk_score >= 70 ? 'high' : alert.risk_score >= 40 ? 'medium' : 'low'}`}
                              style={{ fontSize: 10, padding: '2px 8px' }}>
                          Score: {alert.risk_score}
                        </span>
                        <span className="chip" style={{ fontSize: 10 }}>
                          {alert.alert_type === 'high_risk' ? '🔴 Critical' : 'ℹ️ Info'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                          {timeAgo(alert.created_at)}
                        </span>
                        <ChevronDown size={14} style={{
                          color: 'var(--text-muted)',
                          transform: expanded ? 'rotate(180deg)' : 'rotate(0)',
                          transition: 'transform 0.2s',
                        }} />
                      </div>
                    </div>

                    <p style={{
                      fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5,
                      display: expanded ? 'block' : '-webkit-box',
                      WebkitLineClamp: expanded ? 'unset' : 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: expanded ? 'visible' : 'hidden',
                    }}>
                      {alert.message}
                    </p>

                    {expanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)', cursor: 'default' }}
                        onClick={e => e.stopPropagation()}
                      >
                        <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
                          <Brain size={16} /> AI De-escalation Primer
                        </h4>
                        {loadingPrimer === alert.id ? (
                          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Generating actionable advice...</div>
                        ) : (
                          <div style={{ 
                            fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6, 
                            background: 'var(--bg-tertiary)', padding: '16px', borderRadius: '8px',
                            whiteSpace: 'pre-line', border: '1px solid var(--border)'
                          }}>
                            {primers[alert.id]}
                          </div>
                        )}
                        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
                          {alert.status === 'pending' && (
                            <>
                              <button className="btn btn-primary btn-sm" onClick={() => handleAcknowledge(alert.id)}>
                                <CheckCircle size={14} />
                                Acknowledge
                              </button>
                              <button className="btn btn-secondary btn-sm" onClick={() => handleResolve(alert.id)}>
                                <X size={14} />
                                Resolve
                              </button>
                            </>
                          )}
                          {alert.status === 'acknowledged' && (
                             <button className="btn btn-primary btn-sm" onClick={() => handleResolve(alert.id)}>
                               <CheckCircle size={14} />
                               Mark Resolved
                             </button>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>

        {filtered.length === 0 && (
          <div className="empty-state">
            <Bell size={48} />
            <h3>No alerts found</h3>
            <p>No alerts match the current filter</p>
          </div>
        )}
      </div>
    </div>
  )
}
