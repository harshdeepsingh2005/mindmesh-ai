import { useState } from 'react'
import { api } from '../lib/api'
import { ShieldAlert, Send, HeartHandshake } from 'lucide-react'

export default function ReportConcern() {
  const [peerId, setPeerId] = useState('')
  const [concern, setConcern] = useState('')
  const [status, setStatus] = useState('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!peerId || !concern) return
    
    setStatus('loading')
    setErrorMsg('')
    try {
      await api.reportPeer(peerId, concern)
      setStatus('success')
    } catch (err) {
      console.error(err)
      setErrorMsg(err.message || 'Something went wrong.')
      setStatus('error')
    }
  }

  // If we've successfully submitted, show a thank you message
  if (status === 'success') {
    return (
      <div className="page-container">
        <div style={{ maxWidth: '600px', margin: '4rem auto', textAlign: 'center', background: 'var(--surface)', padding: '3rem', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <HeartHandshake size={64} style={{ color: 'var(--primary)', margin: '0 auto 1.5rem' }} />
          <h2>Thank You For Reaching Out</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem', lineHeight: '1.6' }}>
            We have received your concern. A trained counselor will review the situation and reach out to your friend safely. Your identity remains completely anonymous.
          </p>
          <button 
            className="btn btn-primary"
            onClick={() => { setStatus('idle'); setPeerId(''); setConcern(''); }}
            style={{ marginTop: '2rem' }}
          >
            Submit Another Concern
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="page-container">
      <div className="page-header" style={{ marginBottom: '2rem' }}>
        <div>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <ShieldAlert style={{ color: 'var(--danger)' }} />
            I'm Worried about a Friend
          </h1>
          <p className="subtitle">
            If you notice a friend struggling, you can anonymously report your concern here. 
            A counselor will discreetly check in on them. Your name will not be shared.
          </p>
        </div>
      </div>

      <div style={{ maxWidth: '600px', background: 'var(--surface)', padding: '2rem', borderRadius: '12px', border: '1px solid var(--border)' }}>
        <form onSubmit={handleSubmit}>
          
          <div className="form-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
              Friend's Name or Student ID *
            </label>
            <input
              type="text"
              required
              className="input"
              placeholder="e.g. STU-1025 or John Doe"
              value={peerId}
              onChange={(e) => setPeerId(e.target.value)}
              style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--background)' }}
            />
          </div>

          <div className="form-group" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
              What have you noticed? *
            </label>
            <textarea
              required
              rows={6}
              className="input"
              placeholder="Please describe why you're worried..."
              value={concern}
              onChange={(e) => setConcern(e.target.value)}
              style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--background)', resize: 'vertical' }}
            />
          </div>

          {status === 'error' && (
            <div style={{ padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', marginBottom: '1.5rem' }}>
              Error reporting concern: {errorMsg}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={status === 'loading'}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              {status === 'loading' ? 'Sending...' : (
                <>
                  <Send size={18} />
                  Submit Anonymously
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      <div style={{ marginTop: '3rem', padding: '1.5rem', backgroundColor: 'var(--surface-hover)', borderRadius: '8px' }}>
        <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>If this is an immediate emergency:</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Do not wait for a counselor's response. Please contact Campus Security or use the SOS button at the bottom right immediately.
        </p>
      </div>
    </div>
  )
}
