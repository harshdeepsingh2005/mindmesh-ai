import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Cpu, Play, CheckCircle, Clock, Zap, BarChart3, Network, Layers } from 'lucide-react'
import { api } from '../lib/api'

const MODEL_ICONS = {
  text_embeddings: Layers,
  emotion_detection: Brain,
  sentiment_analysis: BarChart3,
  anomaly_detection: Zap,
  student_clustering: Network,
  topic_discovery: Cpu,
  risk_scoring: CheckCircle,
}

const MODEL_COLORS = {
  text_embeddings: '#6366f1',
  emotion_detection: '#8b5cf6',
  sentiment_analysis: '#3b82f6',
  anomaly_detection: '#ef4444',
  student_clustering: '#10b981',
  topic_discovery: '#f59e0b',
  risk_scoring: '#ec4899',
}

const MOCK_MODELS = [
  {
    model_name: 'text_embeddings', version: '3.0.0', status: 'active',
    description: 'TF-IDF text embedding engine — bigrams, sublinear TF, 5000 features',
    config: { type: 'tfidf', max_features: 5000, ngram_range: [1, 2] },
    metrics: { vocabulary_size: 4821 },
  },
  {
    model_name: 'emotion_detection', version: '3.0.0', status: 'active',
    description: 'K-Means emotion cluster discovery — 5 clusters, auto-labelled via seed terms',
    config: { type: 'kmeans_clustering', n_clusters: 5 },
    metrics: { silhouette_score: 0.412, n_clusters: 5 },
  },
  {
    model_name: 'sentiment_analysis', version: '3.0.0', status: 'active',
    description: 'VADER sentiment analyser — compound scoring with high-risk keyword override',
    config: { type: 'vader', library: 'nltk' },
    metrics: { type: 'rule_based_unsupervised' },
  },
  {
    model_name: 'anomaly_detection', version: '1.0.0', status: 'active',
    description: 'Isolation Forest behavioral anomaly detection — 10% contamination rate',
    config: { type: 'isolation_forest', contamination: 0.1, n_estimators: 100 },
    metrics: { anomaly_rate: 0.092 },
  },
  {
    model_name: 'student_clustering', version: '1.0.0', status: 'active',
    description: 'Gaussian Mixture Model — probabilistic student behavioral profiling',
    config: { type: 'gmm', covariance_type: 'full' },
    metrics: { silhouette_score: 0.358, n_clusters: 4 },
  },
  {
    model_name: 'topic_discovery', version: '1.0.0', status: 'active',
    description: 'NMF topic modeling — discovers latent themes in student text',
    config: { type: 'nmf', n_topics: 8, init: 'nndsvd' },
    metrics: { reconstruction_error: 12.45, n_topics: 8 },
  },
  {
    model_name: 'risk_scoring', version: '2.0.0', status: 'active',
    description: 'Composite risk scorer — Isolation Forest anomaly (30%) + VADER sentiment (20%)',
    config: { type: 'anomaly_composite', primary_signal: 'isolation_forest' },
    metrics: { anomaly_weight: 0.30, sentiment_weight: 0.20 },
  },
]

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

export default function Models() {
  const [models, setModels] = useState(MOCK_MODELS)
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState(null)

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getModels()
        if (data?.models?.length) setModels(data.models)
      } catch { /* keep mock */ }
    })()
  }, [])

  const handleTrainAll = async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const result = await api.trainModels({ model_name: 'all', corpus_size: 500, feature_size: 100 })
      setTrainResult(result)
      
      // Pull the freshly trained models with their new live metrics!
      const data = await api.getModels()
      if (data?.models?.length) {
        setModels(data.models)
      }
    } catch (err) {
      setTrainResult({ error: err.message })
    } finally {
      setTraining(false)
    }
  }

  return (
    <div>
      <div className="page-header page-header-row">
        <div>
          <h2>ML Models</h2>
          <p>Unsupervised machine learning pipeline — model registry & training</p>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleTrainAll}
          disabled={training}
        >
          {training ? (
            <>
              <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Training…
            </>
          ) : (
            <>
              <Play size={14} />
              Train All Models
            </>
          )}
        </button>
      </div>

      {trainResult && (
        <motion.div
          className={`alert-banner ${trainResult.error ? 'danger' : 'info'}`}
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
        >
          {trainResult.error ? (
            <>{trainResult.error}</>
          ) : (
            <>{trainResult.total_models} models trained successfully</>
          )}
        </motion.div>
      )}

      {/* Pipeline Visualization */}
      <motion.div className="card" {...anim} style={{ marginBottom: 'var(--space-lg)', padding: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">Unsupervised ML Pipeline Architecture</span>
        </div>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: 8, flexWrap: 'wrap', paddingTop: 'var(--space-md)',
        }}>
          {['TF-IDF', '→', 'K-Means', '→', 'NMF', '→', 'Isolation Forest', '→', 'GMM', '→', 'Risk Score'].map((step, i) => (
            step === '→' ? (
              <span key={i} style={{ color: 'var(--text-muted)', fontSize: 18, fontWeight: 300 }}>→</span>
            ) : (
              <div key={i} style={{
                padding: '10px 18px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-md)',
                fontSize: 13,
                fontWeight: 600,
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
              }}>
                {step}
              </div>
            )
          ))}
        </div>
      </motion.div>

      {/* Model Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
        gap: 'var(--space-md)',
      }}>
        {models.map((model, i) => {
          const Icon = MODEL_ICONS[model.model_name] || Brain
          const color = MODEL_COLORS[model.model_name] || '#6366f1'

          return (
            <motion.div
              key={model.model_name}
              className="card"
              {...anim}
              transition={{ ...anim.transition, delay: i * 0.06 }}
              style={{ borderTop: `3px solid ${color}` }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
                <div style={{
                  width: 40, height: 40, borderRadius: 'var(--radius-md)',
                  background: `${color}18`, color,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0,
                }}>
                  <Icon size={20} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{
                      fontSize: 14.5, fontWeight: 700, color: 'var(--text-primary)',
                      fontFamily: 'var(--font-mono)',
                    }}>
                      {model.model_name}
                    </span>
                    <span className="chip" style={{ fontSize: 10 }}>v{model.version}</span>
                  </div>
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4,
                    fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                    color: model.status === 'active' ? 'var(--color-success)' : 'var(--text-muted)',
                    background: model.status === 'active' ? 'var(--color-success-bg)' : 'var(--bg-tertiary)',
                    padding: '2px 8px', borderRadius: 'var(--radius-full)',
                  }}>
                    {model.status === 'active' && <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--color-success)' }} />}
                    {model.status}
                  </div>
                </div>
              </div>

              <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 'var(--space-md)' }}>
                {model.description}
              </p>

              {/* Config */}
              <div style={{ marginBottom: 'var(--space-md)' }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                  Configuration
                </span>
                <div style={{
                  marginTop: 6, padding: '8px 12px',
                  background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-mono)', fontSize: 11,
                  color: 'var(--text-secondary)', lineHeight: 1.6,
                  border: '1px solid var(--border-subtle)',
                  maxHeight: 80, overflow: 'auto',
                }}>
                  {Object.entries(model.config || {}).map(([k, v]) => (
                    <div key={k}>
                      <span style={{ color: 'var(--text-muted)' }}>{k}:</span>{' '}
                      <span style={{ color }}>
                        {Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Metrics */}
              <div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
                  Metrics
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 6 }}>
                  {Object.entries(model.metrics || {}).map(([k, v]) => (
                    <div key={k} style={{
                      padding: '4px 10px', background: 'var(--bg-tertiary)',
                      borderRadius: 'var(--radius-sm)', fontSize: 11,
                      border: '1px solid var(--border-subtle)',
                    }}>
                      <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                      <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color }}>
                        {typeof v === 'number' ? v.toFixed(3) : (typeof v === 'object' && v !== null ? JSON.stringify(v).replace(/[{}"]/g, '') : String(v))}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
