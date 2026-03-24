import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, TrendingUp, Heart, Calendar, Award, Sparkles } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'
import { useAuth } from '../App'

// Mock weekly mood data
const MOCK_MOOD_HISTORY = Array.from({ length: 21 }, (_, i) => ({
  date: new Date(Date.now() - (20 - i) * 86400000).toISOString().split('T')[0].slice(5),
  mood: Math.round(4 + Math.sin(i * 0.5) * 2.5 + Math.random() * 1.5),
  sentiment: 0.2 + Math.sin(i * 0.3) * 0.3 + Math.random() * 0.15,
}))

const MOCK_WELLNESS = [
  { metric: 'Mood', value: 72, fullMark: 100 },
  { metric: 'Sleep', value: 65, fullMark: 100 },
  { metric: 'Social', value: 80, fullMark: 100 },
  { metric: 'Academic', value: 58, fullMark: 100 },
  { metric: 'Energy', value: 70, fullMark: 100 },
  { metric: 'Calm', value: 64, fullMark: 100 },
]

const MOCK_STREAKS = {
  current: 5,
  longest: 14,
  total_entries: 38,
  this_month: 12,
}

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

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
          {p.name}: {typeof p.value === 'number' ? (Number.isInteger(p.value) ? p.value : p.value.toFixed(2)) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function StudentProgress() {
  const { user } = useAuth()

  return (
    <div>
      <div className="page-header">
        <h2>My Progress</h2>
        <p>Track your wellness journey over time, {user?.name?.split(' ')[0] || 'Student'}</p>
      </div>

      {/* Streak Cards */}
      <div className="stats-grid stagger" style={{ marginBottom: 'var(--space-xl)' }}>
        <motion.div className="stat-card" {...anim}>
          <div className="stat-icon purple"><Sparkles size={20} /></div>
          <div className="stat-label">Current Streak</div>
          <div className="stat-value">{MOCK_STREAKS.current} days</div>
          <div className="stat-change positive">
            <TrendingUp size={12} />Keep it up!
          </div>
        </motion.div>

        <motion.div className="stat-card" {...anim} transition={{ ...anim.transition, delay: 0.06 }}>
          <div className="stat-icon green"><Award size={20} /></div>
          <div className="stat-label">Longest Streak</div>
          <div className="stat-value">{MOCK_STREAKS.longest} days</div>
        </motion.div>

        <motion.div className="stat-card" {...anim} transition={{ ...anim.transition, delay: 0.12 }}>
          <div className="stat-icon blue"><Calendar size={20} /></div>
          <div className="stat-label">This Month</div>
          <div className="stat-value">{MOCK_STREAKS.this_month}</div>
          <div className="card-subtitle">entries logged</div>
        </motion.div>

        <motion.div className="stat-card" {...anim} transition={{ ...anim.transition, delay: 0.18 }}>
          <div className="stat-icon amber"><Heart size={20} /></div>
          <div className="stat-label">Total Check-ins</div>
          <div className="stat-value">{MOCK_STREAKS.total_entries}</div>
        </motion.div>
      </div>

      <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
        {/* Mood Over Time */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.24 }}>
          <div className="card-header">
            <span className="card-title">
              <Activity size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Your Mood Journey
            </span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={MOCK_MOOD_HISTORY}>
                <defs>
                  <linearGradient id="moodGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={[0, 10]}
                  ticks={[0, 2, 4, 6, 8, 10]} />
                <Tooltip content={<ChartTooltip />} />
                <Area
                  type="monotone" dataKey="mood" name="Mood"
                  stroke="#6366f1" fill="url(#moodGrad)" strokeWidth={2}
                  dot={false} activeDot={{ r: 4, fill: '#6366f1' }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Wellness Radar */}
        <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.30 }}>
          <div className="card-header">
            <span className="card-title">
              <Heart size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Wellness Overview
            </span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={MOCK_WELLNESS}>
                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                <PolarAngleAxis dataKey="metric" stroke="var(--text-muted)" fontSize={12} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar
                  name="Score" dataKey="value"
                  stroke="#10b981" fill="rgba(16,185,129,0.2)" strokeWidth={2}
                  dot={{ r: 3, fill: '#10b981' }}
                />
                <Tooltip content={<ChartTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Encouragement Card */}
      <motion.div
        className="card"
        {...anim}
        transition={{ ...anim.transition, delay: 0.36 }}
        style={{
          background: 'linear-gradient(135deg, rgba(99,102,241,0.08), rgba(16,185,129,0.08))',
          borderColor: 'rgba(99,102,241,0.2)',
          textAlign: 'center',
          padding: 'var(--space-2xl)',
        }}
      >
        <Sparkles size={32} style={{ color: 'var(--accent-primary)', marginBottom: 'var(--space-md)' }} />
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
          You're doing great, {user?.name?.split(' ')[0] || 'friend'}! 💪
        </h3>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', maxWidth: 480, margin: '0 auto', lineHeight: 1.6 }}>
          You've been consistent with your check-ins. Remember, it's okay to have
          tough days — what matters is that you keep showing up. Your counselor is
          always here to help.
        </p>
      </motion.div>
    </div>
  )
}
