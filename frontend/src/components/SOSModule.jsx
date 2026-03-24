import { useState } from 'react'
import { AlertCircle, Phone, X, ShieldAlert } from 'lucide-react'
import { api } from '../lib/api'

export default function SOSModule() {
  const [isOpen, setIsOpen] = useState(false)
  const [status, setStatus] = useState('idle')

  const triggerSOS = async () => {
    setStatus('loading')
    try {
      // In a real app we'd get geolocation here
      await api.triggerSOS('Campus library', 'Requested via emergency SOS button')
      setStatus('success')
    } catch (err) {
      console.error(err)
      setStatus('error')
    }
  }

  return (
    <>
      <button 
        className="sos-floating-btn"
        onClick={() => setIsOpen(true)}
        title="Get Help Now"
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          backgroundColor: '#ef4444',
          color: 'white',
          border: 'none',
          boxShadow: '0 10px 25px rgba(239, 68, 68, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          zIndex: 9999,
          transition: 'transform 0.2s',
        }}
        onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.05)'}
        onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
      >
        <AlertCircle size={32} />
      </button>

      {isOpen && (
        <div className="sos-modal-overlay" style={{
          position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 10000,
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <div className="sos-modal-content" style={{
            background: 'white', padding: '2.5rem', borderRadius: '16px', width: '90%', maxWidth: '420px',
            position: 'relative', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)'
          }}>
            <button 
              onClick={() => setIsOpen(false)}
              style={{ position: 'absolute', top: '16px', right: '16px', background: 'transparent', border: 'none', cursor: 'pointer', color: '#6b7280' }}
            >
              <X size={24} />
            </button>
            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <ShieldAlert size={56} color="#ef4444" style={{ margin: '0 auto' }} />
              <h2 style={{ color: '#111827', marginTop: '1rem', fontSize: '1.75rem', fontWeight: 'bold' }}>Emergency Assistance</h2>
              <p style={{ color: '#4b5563', marginTop: '0.5rem', lineHeight: '1.5' }}>
                If you are in immediate danger or experiencing a crisis, please request help immediately. You are not alone.
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <button 
                onClick={triggerSOS}
                disabled={status === 'loading' || status === 'success'}
                style={{
                  background: status === 'success' ? '#10b981' : '#ef4444',
                  color: 'white', padding: '1.25rem', borderRadius: '12px', border: 'none',
                  fontSize: '1.1rem', fontWeight: 'bold', cursor: 'pointer',
                  display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              >
                <AlertCircle size={22} />
                {status === 'loading' ? 'Dispatching Alert...' : 
                 status === 'success' ? 'Campus Team Alerted!' : 'Alert Campus Counselor'}
              </button>

              <a href="tel:988" style={{
                  background: '#f3f4f6', color: '#1f2937', padding: '1rem', borderRadius: '12px',
                  textDecoration: 'none', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', fontWeight: '600',
                  border: '1px solid #e5e7eb'
              }}>
                <Phone size={20} /> Call National Suicide Hotline (988)
              </a>
              <a href="tel:911" style={{
                  background: '#f3f4f6', color: '#1f2937', padding: '1rem', borderRadius: '12px',
                  textDecoration: 'none', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', fontWeight: '600',
                  border: '1px solid #e5e7eb'
              }}>
                <Phone size={20} /> Call Campus Security
              </a>
            </div>
            {status === 'success' && (
               <div style={{ padding: '1rem', background: '#d1fae5', borderRadius: '8px', marginTop: '1.5rem' }}>
                 <p style={{ textAlign: 'center', color: '#047857', fontWeight: '500', margin: 0 }}>
                   A campus counselor has been alerted with your location and will contact you. Please stay where you are.
                 </p>
               </div>
            )}
            {status === 'error' && (
               <p style={{ textAlign: 'center', color: '#ef4444', marginTop: '1rem' }}>
                 Failed to send alert automatically. Please call the numbers above.
               </p>
            )}
          </div>
        </div>
      )}
    </>
  )
}
