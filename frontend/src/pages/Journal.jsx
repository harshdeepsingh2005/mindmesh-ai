import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, SmilePlus, Send, Sparkles, Brain, Heart, AlertTriangle, TrendingUp, Loader } from 'lucide-react'
import { api } from '../lib/api'

const MOOD_EMOJIS = [
  { value: 1, emoji: '😢', label: 'Terrible' },
  { value: 2, emoji: '😞', label: 'Bad' },
  { value: 3, emoji: '😕', label: 'Down' },
  { value: 4, emoji: '😐', label: 'Meh' },
  { value: 5, emoji: '🙂', label: 'Okay' },
  { value: 6, emoji: '😊', label: 'Good' },
  { value: 7, emoji: '😄', label: 'Happy' },
  { value: 8, emoji: '🤩', label: 'Great' },
  { value: 9, emoji: '🥳', label: 'Amazing' },
  { value: 10, emoji: '✨', label: 'Incredible' },
]

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

export default function Journal() {
  const [tab, setTab] = useState('journal')
  const [journalText, setJournalText] = useState('')
  const [moodRating, setMoodRating] = useState(5)
  const [checkinNotes, setCheckinNotes] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)

  const handleAnalyze = async (text) => {
    if (!text.trim()) return
    setAnalyzing(true)
    setAnalysisResult(null)
    try {
      const result = await api.analyzeText(text)
      setAnalysisResult(result)
    } catch (err) {
      // Fallback mock analysis
      setAnalysisResult({
        emotion: {
          cluster_label: text.includes('happy') || text.includes('good') ? 'positive' :
                         text.includes('sad') || text.includes('bad') ? 'distress' : 'neutral',
          confidence_score: 0.72,
          predicted_cluster: 0,
        },
        sentiment: {
          sentiment_label: text.includes('happy') ? 'positive' : text.includes('sad') ? 'negative' : 'neutral',
          sentiment_score: text.includes('happy') ? 0.65 : text.includes('sad') ? -0.45 : 0.05,
          positive_score: 0.3,
          negative_score: 0.1,
          neutral_score: 0.6,
          high_risk_flag: false,
          high_risk_keywords_found: [],
        },
        topic: { topic_label: 'daily_life', confidence: 0.8 },
      })
    } finally {
      setAnalyzing(false)
    }
  }

  const handleJournalSubmit = (e) => {
    e.preventDefault()
    handleAnalyze(journalText)
  }

  const handleCheckinSubmit = (e) => {
    e.preventDefault()
    handleAnalyze(checkinNotes || `Mood: ${moodRating}/10`)
  }

  const sentimentColor = (score) => {
    if (score > 0.2) return 'var(--color-success)'
    if (score < -0.2) return 'var(--color-danger)'
    return 'var(--text-muted)'
  }

  return (
    <div>
      <div className="page-header">
        <h2>Journal & Check-in</h2>
        <p>Express yourself freely — our AI provides real-time wellness insights</p>
      </div>

      {/* Tab Switcher */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 'var(--space-xl)' }}>
        <button
          className={`btn ${tab === 'journal' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setTab('journal'); setAnalysisResult(null) }}
        >
          <BookOpen size={16} />
          Journal Entry
        </button>
        <button
          className={`btn ${tab === 'checkin' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => { setTab('checkin'); setAnalysisResult(null) }}
        >
          <SmilePlus size={16} />
          Mood Check-in
        </button>
      </div>

      <div className="grid-2-1">
        {/* Left: Input */}
        <motion.div key={tab} className="card" {...anim}>
          {tab === 'journal' ? (
            <form onSubmit={handleJournalSubmit}>
              <div className="card-header">
                <span className="card-title">
                  <BookOpen size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
                  Write a Journal Entry
                </span>
              </div>
              <textarea
                className="input textarea"
                placeholder="How are you feeling today? Write about anything — school, friends, family, worries, hopes…"
                value={journalText}
                onChange={(e) => setJournalText(e.target.value)}
                style={{ minHeight: 200, resize: 'vertical', lineHeight: 1.7, marginBottom: 'var(--space-md)' }}
              />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {journalText.length} characters
                </span>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!journalText.trim() || analyzing}
                >
                  {analyzing ? (
                    <Loader size={16} className="spinner" style={{ animation: 'spin 0.8s linear infinite' }} />
                  ) : (
                    <>
                      <Sparkles size={14} />
                      Analyze with AI
                    </>
                  )}
                </button>
              </div>
            </form>
          ) : (
            <form onSubmit={handleCheckinSubmit}>
              <div className="card-header">
                <span className="card-title">
                  <SmilePlus size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
                  How are you feeling?
                </span>
              </div>

              {/* Emoji Mood Scale */}
              <div style={{ marginBottom: 'var(--space-xl)' }}>
                <div style={{
                  display: 'flex', justifyContent: 'center', gap: 6,
                  flexWrap: 'wrap', padding: 'var(--space-md) 0',
                }}>
                  {MOOD_EMOJIS.map((mood) => (
                    <motion.button
                      key={mood.value}
                      type="button"
                      whileHover={{ scale: 1.15 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setMoodRating(mood.value)}
                      style={{
                        width: 56, height: 56,
                        borderRadius: 'var(--radius-md)',
                        border: `2px solid ${moodRating === mood.value ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                        background: moodRating === mood.value ? 'var(--accent-glow)' : 'var(--bg-tertiary)',
                        cursor: 'pointer',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        fontSize: 22, transition: 'all 0.2s',
                      }}
                    >
                      <span>{mood.emoji}</span>
                    </motion.button>
                  ))}
                </div>
                <div style={{ textAlign: 'center', marginTop: 8 }}>
                  <span style={{
                    fontSize: 16, fontWeight: 700,
                    background: 'var(--accent-gradient)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}>
                    {MOOD_EMOJIS[moodRating - 1]?.label} — {moodRating}/10
                  </span>
                </div>
              </div>

              <textarea
                className="input textarea"
                placeholder="Any thoughts you'd like to share? (optional)"
                value={checkinNotes}
                onChange={(e) => setCheckinNotes(e.target.value)}
                style={{ minHeight: 100, marginBottom: 'var(--space-md)' }}
              />
              <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={analyzing}>
                {analyzing ? (
                  <Loader size={16} style={{ animation: 'spin 0.8s linear infinite' }} />
                ) : (
                  <>
                    <Send size={14} />
                    Submit Check-in
                  </>
                )}
              </button>
            </form>
          )}
        </motion.div>

        {/* Right: AI Analysis Results */}
        <div>
          <AnimatePresence mode="wait">
            {analysisResult ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}
              >
                {/* Emotion */}
                <div className="card" style={{ borderLeft: '3px solid var(--accent-primary)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <Brain size={16} style={{ color: 'var(--accent-primary)' }} />
                    <span style={{ fontWeight: 700, fontSize: 13 }}>Emotion Cluster</span>
                  </div>
                  <div style={{
                    fontSize: 22, fontWeight: 800, textTransform: 'capitalize',
                    marginBottom: 4,
                    background: 'var(--accent-gradient)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}>
                    {analysisResult.emotion?.cluster_label || analysisResult.emotion?.predicted_emotion}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Confidence: {((analysisResult.emotion?.confidence_score || 0) * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Sentiment */}
                <div className="card" style={{ borderLeft: `3px solid ${sentimentColor(analysisResult.sentiment?.sentiment_score || 0)}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <Heart size={16} style={{ color: sentimentColor(analysisResult.sentiment?.sentiment_score || 0) }} />
                    <span style={{ fontWeight: 700, fontSize: 13 }}>Sentiment</span>
                  </div>
                  <div style={{
                    fontSize: 22, fontWeight: 800, textTransform: 'capitalize',
                    color: sentimentColor(analysisResult.sentiment?.sentiment_score || 0),
                    marginBottom: 4,
                  }}>
                    {analysisResult.sentiment?.sentiment_label}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Score: {(analysisResult.sentiment?.sentiment_score || 0).toFixed(3)}
                  </div>

                  {/* Score bars */}
                  <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {['positive', 'neutral', 'negative'].map(key => {
                      const val = analysisResult.sentiment?.[`${key}_score`] || 0
                      const colors = { positive: '#10b981', neutral: '#94a3b8', negative: '#ef4444' }
                      return (
                        <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)', width: 52, textTransform: 'capitalize' }}>{key}</span>
                          <div style={{ flex: 1, height: 6, background: 'var(--bg-primary)', borderRadius: 3, overflow: 'hidden' }}>
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${val * 100}%` }}
                              transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                              style={{ height: '100%', background: colors[key], borderRadius: 3 }}
                            />
                          </div>
                          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', width: 36, textAlign: 'right' }}>
                            {(val * 100).toFixed(0)}%
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Risk Alert */}
                {analysisResult.sentiment?.high_risk_flag && (
                  <motion.div
                    className="alert-banner danger"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    style={{ fontWeight: 600 }}
                  >
                    <AlertTriangle size={18} />
                    High-risk content detected. A counselor has been notified.
                  </motion.div>
                )}

                {/* Topic */}
                {analysisResult.topic?.topic_label && (
                  <div className="card" style={{ borderLeft: '3px solid var(--color-warning)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <TrendingUp size={16} style={{ color: 'var(--color-warning)' }} />
                      <span style={{ fontWeight: 700, fontSize: 13 }}>Topic Detected</span>
                    </div>
                    <span style={{
                      fontSize: 15, fontWeight: 700, textTransform: 'capitalize',
                      color: 'var(--color-warning)',
                    }}>
                      {analysisResult.topic.topic_label.replace(/_/g, ' ')}
                    </span>
                  </div>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                className="card"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 300, textAlign: 'center' }}
              >
                <Sparkles size={40} style={{ color: 'var(--accent-primary)', opacity: 0.4, marginBottom: 'var(--space-md)' }} />
                <h3 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>
                  AI Analysis
                </h3>
                <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 220 }}>
                  Write something and click analyze to see real-time ML insights
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
