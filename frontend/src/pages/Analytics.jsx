import { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Activity, Zap } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line,
  ScatterChart, Scatter, ZAxis, Legend
} from 'recharts'

// Simulated 30-day emotion trend
const emotionTrend = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0].slice(5),
  positive: 0.35 + Math.sin(i * 0.25) * 0.1 + Math.random() * 0.05,
  neutral: 0.30 + Math.cos(i * 0.3) * 0.05,
  distress: 0.12 + Math.sin(i * 0.4 + 2) * 0.05 + Math.random() * 0.03,
  anxiety: 0.13 + Math.cos(i * 0.35) * 0.04,
}))

// Sentiment over time
const sentimentTrend = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0].slice(5),
  compound: 0.2 + Math.sin(i * 0.2) * 0.3 + Math.random() * 0.1,
  positive: 0.4 + Math.random() * 0.2,
  negative: 0.15 + Math.random() * 0.1,
}))

// Activity breakdown
const activityBreakdown = Array.from({ length: 14 }, (_, i) => ({
  date: new Date(Date.now() - (13 - i) * 86400000).toISOString().split('T')[0].slice(5),
  journal: Math.floor(30 + Math.random() * 40),
  checkin: Math.floor(60 + Math.random() * 50),
  survey: Math.floor(10 + Math.random() * 20),
}))

// Risk score distribution
const riskHistogram = [
  { range: '0-10', count: 120, color: '#10b981' },
  { range: '10-20', count: 210, color: '#10b981' },
  { range: '20-30', count: 185, color: '#10b981' },
  { range: '30-40', count: 145, color: '#34d399' },
  { range: '40-50', count: 98, color: '#f59e0b' },
  { range: '50-60', count: 76, color: '#f59e0b' },
  { range: '60-70', count: 54, color: '#fb923c' },
  { range: '70-80', count: 32, color: '#ef4444' },
  { range: '80-90', count: 18, color: '#ef4444' },
  { range: '90-100', count: 7, color: '#dc2626' },
]

// Anomaly scatter
const anomalyScatter = Array.from({ length: 80 }, (_, i) => ({
  sentiment: -0.5 + Math.random() * 1.5,
  emotion: Math.random(),
  anomaly_score: Math.random() < 0.1 ? -0.3 - Math.random() * 0.7 : 0.1 + Math.random() * 0.5,
  size: 40 + Math.random() * 100,
  label: i,
}))

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius-sm)', padding: '10px 14px', fontSize: 12,
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

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

export default function Analytics() {
  const [period, setPeriod] = useState(30)

  return (
    <div>
      <div className="page-header page-header-row">
        <div>
          <h2>Analytics</h2>
          <p>Deep dive into behavioral patterns and ML model outputs</p>
        </div>
        <select className="select" value={period} onChange={e => setPeriod(Number(e.target.value))}>
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>

      {/* Emotion Cluster Trends */}
      <motion.div className="card" {...anim} style={{ marginBottom: 'var(--space-lg)' }}>
        <div className="card-header">
          <span className="card-title">
            <Activity size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
            Emotion Cluster Distribution Over Time
          </span>
          <span className="chip">K-Means Clustering</span>
        </div>
        <div style={{ height: 340 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={emotionTrend}>
              <defs>
                <linearGradient id="gPositive" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gNeutral" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#94a3b8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gDistress" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f87171" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#f87171" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gAnxiety" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#fbbf24" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={[0, 0.6]} />
              <Tooltip content={<ChartTooltip />} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
              <Area type="monotone" dataKey="positive" stroke="#34d399" fill="url(#gPositive)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="neutral" stroke="#94a3b8" fill="url(#gNeutral)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="distress" stroke="#f87171" fill="url(#gDistress)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="anxiety" stroke="#fbbf24" fill="url(#gAnxiety)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
        {/* VADER Sentiment Trend */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.06 }}>
          <div className="card-header">
            <span className="card-title">
              <TrendingUp size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              VADER Sentiment Analysis
            </span>
            <span className="chip">VADER</span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sentimentTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={[-0.5, 1]} />
                <Tooltip content={<ChartTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="compound" stroke="#6366f1" strokeWidth={2} dot={false} name="Compound" />
                <Line type="monotone" dataKey="positive" stroke="#10b981" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Positive" />
                <Line type="monotone" dataKey="negative" stroke="#ef4444" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Negative" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Risk Score Distribution */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.12 }}>
          <div className="card-header">
            <span className="card-title">
              <BarChart3 size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Risk Score Histogram
            </span>
            <span className="chip">Isolation Forest</span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskHistogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="range" stroke="var(--text-muted)" fontSize={10} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="count" name="Students" radius={[4, 4, 0, 0]} barSize={28}>
                  {riskHistogram.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
        {/* Activity Breakdown */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.18 }}>
          <div className="card-header">
            <span className="card-title">Daily Activity Breakdown</span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activityBreakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <Tooltip content={<ChartTooltip />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="checkin" stackId="a" fill="#6366f1" name="Check-ins" radius={[0, 0, 0, 0]} />
                <Bar dataKey="journal" stackId="a" fill="#8b5cf6" name="Journals" />
                <Bar dataKey="survey" stackId="a" fill="#a78bfa" name="Surveys" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Anomaly Detection Scatter Plot */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.24 }}>
          <div className="card-header">
            <span className="card-title">
              <Zap size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Anomaly Detection Scatter
            </span>
            <span className="chip">Isolation Forest</span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis
                  dataKey="sentiment"
                  name="Sentiment"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  domain={[-0.5, 1.2]}
                  label={{ value: 'Sentiment Score', position: 'bottom', offset: -5, fill: 'var(--text-muted)', fontSize: 10 }}
                />
                <YAxis
                  dataKey="emotion"
                  name="Emotion"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickLine={false}
                  domain={[0, 1]}
                  label={{ value: 'Emotion Score', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 10 }}
                />
                <ZAxis dataKey="size" range={[20, 150]} />
                <Tooltip content={<ChartTooltip />} />
                <Scatter
                  data={anomalyScatter.filter(d => d.anomaly_score > 0)}
                  fill="#6366f1"
                  fillOpacity={0.6}
                  name="Normal"
                />
                <Scatter
                  data={anomalyScatter.filter(d => d.anomaly_score <= 0)}
                  fill="#ef4444"
                  fillOpacity={0.9}
                  name="Anomaly"
                  shape="diamond"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
