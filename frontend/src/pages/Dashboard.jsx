import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Users, AlertTriangle, Brain, Activity, TrendingDown,
  TrendingUp, ShieldAlert, Heart, BarChart3
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { api } from '../lib/api'

// ── Mock data for demo mode ────────────────────────────────
const MOCK_OVERVIEW = {
  total_students: 1247,
  active_students: 892,
  total_records: 15832,
  high_risk_count: 23,
  medium_risk_count: 89,
  avg_emotion_score: 0.62,
  avg_sentiment_score: 0.34,
  open_alerts: 7,
}

const MOCK_EMOTION_TREND = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
  value: 0.5 + Math.sin(i * 0.3) * 0.15 + Math.random() * 0.1,
}))

const MOCK_SENTIMENT_TREND = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
  value: 0.25 + Math.cos(i * 0.2) * 0.2 + Math.random() * 0.15,
}))

const MOCK_RISK_DIST = [
  { label: 'Low Risk', count: 780, percentage: 69.5, color: '#10b981' },
  { label: 'Medium Risk', count: 251, percentage: 22.4, color: '#f59e0b' },
  { label: 'High Risk', count: 91, percentage: 8.1, color: '#ef4444' },
]

const MOCK_EMOTION_DIST = [
  { emotion: 'positive', count: 4521, percentage: 38.2, color: '#34d399' },
  { emotion: 'neutral', count: 3890, percentage: 32.9, color: '#94a3b8' },
  { emotion: 'anxiety', count: 1567, percentage: 13.2, color: '#fbbf24' },
  { emotion: 'distress', count: 1234, percentage: 10.4, color: '#f87171' },
  { emotion: 'anger', count: 620, percentage: 5.3, color: '#fb923c' },
]

const MOCK_AT_RISK = [
  { name: 'Arjun Mehta', school: 'DPS Bangalore', risk_score: 87, risk_level: 'high', emotion: 'distress', alerts: 3 },
  { name: 'Priya Sharma', school: 'KV Delhi', risk_score: 82, risk_level: 'high', emotion: 'anxiety', alerts: 2 },
  { name: 'Rohan Patel', school: 'DAV Mumbai', risk_score: 78, risk_level: 'high', emotion: 'distress', alerts: 2 },
  { name: 'Sneha Gupta', school: 'JNV Lucknow', risk_score: 74, risk_level: 'high', emotion: 'anxiety', alerts: 1 },
  { name: 'Vikram Singh', school: 'DPS Jaipur', risk_score: 71, risk_level: 'high', emotion: 'anger', alerts: 1 },
]

const MOCK_RADAR = [
  { metric: 'Emotion', value: 72, fullMark: 100 },
  { metric: 'Sentiment', value: 65, fullMark: 100 },
  { metric: 'Engagement', value: 84, fullMark: 100 },
  { metric: 'Consistency', value: 58, fullMark: 100 },
  { metric: 'Risk Signal', value: 31, fullMark: 100 },
  { metric: 'Social', value: 76, fullMark: 100 },
]

const cardAnim = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius-sm)',
      padding: '10px 14px',
      fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: 600 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [overview, setOverview] = useState(MOCK_OVERVIEW)
  const [emotionTrend, setEmotionTrend] = useState(MOCK_EMOTION_TREND)
  const [sentimentTrend, setSentimentTrend] = useState(MOCK_SENTIMENT_TREND)
  const [riskDist, setRiskDist] = useState(MOCK_RISK_DIST)
  const [emotionDist, setEmotionDist] = useState(MOCK_EMOTION_DIST)
  const [atRisk, setAtRisk] = useState(MOCK_AT_RISK)
  const [period, setPeriod] = useState(30)

  useEffect(() => {
    (async () => {
      try {
        const dash = await api.getDashboard(period)
        if (dash?.overview) setOverview(dash.overview)
        if (dash?.emotion_trend?.data_points?.length) {
          setEmotionTrend(dash.emotion_trend.data_points)
        }
        if (dash?.risk_distribution?.buckets?.length) {
          setRiskDist(dash.risk_distribution.buckets.map(b => ({
            label: `${b.level} Risk`,
            count: b.count,
            percentage: b.percentage,
            color: b.level === 'high' ? '#ef4444' : b.level === 'medium' ? '#f59e0b' : '#10b981',
          })))
        }
        if (dash?.at_risk_students?.length) setAtRisk(dash.at_risk_students)
      } catch {
        // Demo mode — keep mock data
      }
    })()
  }, [period])

  const stats = [
    {
      label: 'Total Students',
      value: overview.total_students?.toLocaleString(),
      icon: Users,
      color: 'purple',
      change: '+12%',
      positive: true,
    },
    {
      label: 'Active Today',
      value: overview.active_students?.toLocaleString(),
      icon: Activity,
      color: 'green',
      change: '+8%',
      positive: true,
    },
    {
      label: 'High Risk',
      value: overview.high_risk_count,
      icon: AlertTriangle,
      color: 'red',
      change: '-3',
      positive: true,
    },
    {
      label: 'Open Alerts',
      value: overview.open_alerts,
      icon: ShieldAlert,
      color: 'amber',
      change: '+2',
      positive: false,
    },
    {
      label: 'Avg Emotion',
      value: (overview.avg_emotion_score * 100).toFixed(0) + '%',
      icon: Heart,
      color: 'blue',
      change: '+5%',
      positive: true,
    },
  ]

  return (
    <div>
      <div className="page-header page-header-row">
        <div>
          <h2>Dashboard</h2>
          <p>Real-time mental health intelligence across your school ecosystem</p>
        </div>
        <select
          className="select"
          value={period}
          onChange={(e) => setPeriod(Number(e.target.value))}
        >
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>

      {/* Stat Cards */}
      <div className="stats-grid stagger">
        {stats.map((stat, i) => (
          <motion.div key={i} className="stat-card" {...cardAnim} transition={{ ...cardAnim.transition, delay: i * 0.06 }}>
            <div className={`stat-icon ${stat.color}`}>
              <stat.icon size={20} />
            </div>
            <div className="stat-label">{stat.label}</div>
            <div className="stat-value">{stat.value}</div>
            <div className={`stat-change ${stat.positive ? 'positive' : 'negative'}`}>
              {stat.positive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
              {stat.change}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts Row 1 */}
      <div className="grid-2-1" style={{ marginBottom: 'var(--space-lg)' }}>
        <motion.div className="card" {...cardAnim} transition={{ ...cardAnim.transition, delay: 0.3 }}>
          <div className="card-header">
            <span className="card-title">Emotion & Sentiment Trends</span>
          </div>
          <div className="chart-container" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={emotionTrend}>
                <defs>
                  <linearGradient id="gradEmotion" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradSentiment" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="date"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  tickFormatter={(v) => v?.slice(5)}
                />
                <YAxis
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  domain={[0, 1]}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="value"
                  name="Emotion Score"
                  stroke="#6366f1"
                  fill="url(#gradEmotion)"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#6366f1' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div className="card" {...cardAnim} transition={{ ...cardAnim.transition, delay: 0.36 }}>
          <div className="card-header">
            <span className="card-title">Risk Distribution</span>
          </div>
          <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDist}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  dataKey="count"
                  paddingAngle={3}
                  stroke="none"
                >
                  {riskDist.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
            {riskDist.map((item, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 13 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: item.color }} />
                  <span style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                </div>
                <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                  {item.count}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
        <motion.div className="card" {...cardAnim} transition={{ ...cardAnim.transition, delay: 0.42 }}>
          <div className="card-header">
            <span className="card-title">Emotion Cluster Distribution</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis
                  type="category"
                  dataKey="emotion"
                  stroke="var(--text-muted)"
                  fontSize={12}
                  tickLine={false}
                  width={70}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" name="Count" radius={[0, 6, 6, 0]} barSize={24}>
                  {emotionDist.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div className="card" {...cardAnim} transition={{ ...cardAnim.transition, delay: 0.48 }}>
          <div className="card-header">
            <span className="card-title">Behavioral Radar</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={MOCK_RADAR}>
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis dataKey="metric" stroke="var(--text-muted)" fontSize={11} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar
                  name="Score"
                  dataKey="value"
                  stroke="#6366f1"
                  fill="rgba(99,102,241,0.2)"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#6366f1' }}
                />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* At Risk Students Table */}
      <motion.div className="card" {...cardAnim} transition={{ ...cardAnim.transition, delay: 0.54 }}>
        <div className="card-header">
          <span className="card-title">Students Requiring Attention</span>
          <span className="chip" style={{ color: 'var(--color-danger)', borderColor: 'rgba(239,68,68,0.2)' }}>
            <AlertTriangle size={12} style={{ marginRight: 4 }} />
            {atRisk.length} flagged
          </span>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Student</th>
                <th>School</th>
                <th>Risk Score</th>
                <th>Risk Level</th>
                <th>Emotion Cluster</th>
                <th>Alerts</th>
              </tr>
            </thead>
            <tbody>
              {atRisk.map((student, i) => (
                <tr key={i}>
                  <td>
                    <div className="student-cell">
                      <div className="student-avatar">
                        {(student.name || student.student_identifier || '?').slice(0, 2).toUpperCase()}
                      </div>
                      <span style={{ fontWeight: 600 }}>{student.name || student.student_identifier}</span>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{student.school || '—'}</td>
                  <td>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      fontSize: 14,
                      color: student.risk_score >= 80 ? 'var(--color-danger)' :
                             student.risk_score >= 60 ? 'var(--color-warning)' :
                             'var(--color-success)',
                    }}>
                      {student.risk_score}
                    </span>
                  </td>
                  <td>
                    <span className={`risk-badge ${student.risk_level}`}>
                      <span className="dot" />
                      {student.risk_level}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                    {student.emotion || student.latest_emotion || '—'}
                  </td>
                  <td>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: (student.alerts || student.open_alerts || 0) > 0 ? 'var(--color-danger)' : 'var(--text-muted)',
                    }}>
                      {student.alerts || student.open_alerts || 0}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  )
}
