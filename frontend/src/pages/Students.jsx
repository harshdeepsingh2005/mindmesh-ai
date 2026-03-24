import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Users, Search, Filter, ChevronLeft, ChevronRight, UserPlus, Eye } from 'lucide-react'
import { api } from '../lib/api'

const MOCK_STUDENTS = Array.from({ length: 25 }, (_, i) => ({
  id: `s-${i + 1}`,
  student_identifier: `STU-${String(1001 + i)}`,
  name: ['Arjun Mehta', 'Priya Sharma', 'Rohan Patel', 'Sneha Gupta', 'Vikram Singh',
         'Ananya Das', 'Karthik Reddy', 'Meera Joshi', 'Rahul Kumar', 'Pooja Agarwal',
         'Aditya Verma', 'Riya Kapoor', 'Sanjay Nair', 'Divya Pillai', 'Amit Shah',
         'Kavya Iyer', 'Nikhil Rao', 'Swati Mishra', 'Deepak Choudhury', 'Lakshmi Sundaram',
         'Vivek Bansal', 'Tanya Saxena', 'Manish Tiwari', 'Anjali Pandey', 'Suresh Menon'][i],
  school: ['DPS Bangalore', 'KV Delhi', 'DAV Mumbai', 'JNV Lucknow', 'DPS Jaipur',
           'KV Chennai', 'DAV Pune', 'JNV Hyderabad', 'DPS Kolkata'][i % 9],
  grade: ['8-A', '9-B', '10-C', '7-A', '11-B', '8-C', '9-A', '10-B', '12-A'][i % 9],
  age: 12 + (i % 7),
  risk_level: i < 5 ? 'high' : i < 12 ? 'medium' : 'low',
  risk_score: i < 5 ? 85 - i * 3 : i < 12 ? 55 + (12 - i) * 2 : 20 + i,
  latest_emotion: ['distress', 'anxiety', 'neutral', 'positive', 'anger'][i % 5],
  records: Math.floor(Math.random() * 50) + 5,
}))

const anim = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
}

export default function Students() {
  const [students, setStudents] = useState(MOCK_STUDENTS)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const perPage = 15

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getStudentSummaries({ days: 30, limit: 100 })
        if (data?.students?.length) setStudents(data.students)
      } catch { /* keep mock */ }
    })()
  }, [])

  const filtered = students.filter(s => {
    const matchesFilter = filter === 'all' || s.risk_level === filter
    const matchesSearch = !search ||
      (s.name || s.student_identifier || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.school || '').toLowerCase().includes(search.toLowerCase())
    return matchesFilter && matchesSearch
  })

  const paginated = filtered.slice(page * perPage, (page + 1) * perPage)
  const totalPages = Math.ceil(filtered.length / perPage)

  const riskCounts = {
    all: students.length,
    high: students.filter(s => s.risk_level === 'high').length,
    medium: students.filter(s => s.risk_level === 'medium').length,
    low: students.filter(s => s.risk_level === 'low').length,
  }

  return (
    <div>
      <div className="page-header page-header-row">
        <div>
          <h2>Students</h2>
          <p>Monitor and manage student mental health profiles</p>
        </div>
        <button className="btn btn-primary btn-sm">
          <UserPlus size={14} />
          Add Student
        </button>
      </div>

      {/* Filter Tabs */}
      <motion.div {...anim} style={{
        display: 'flex', gap: 8, marginBottom: 'var(--space-lg)',
        flexWrap: 'wrap',
      }}>
        {[
          { key: 'all', label: 'All Students' },
          { key: 'high', label: 'High Risk' },
          { key: 'medium', label: 'Medium Risk' },
          { key: 'low', label: 'Low Risk' },
        ].map(tab => (
          <button
            key={tab.key}
            className={`btn btn-sm ${filter === tab.key ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => { setFilter(tab.key); setPage(0) }}
          >
            {tab.label}
            <span style={{
              marginLeft: 6, fontFamily: 'var(--font-mono)', fontSize: 11,
              opacity: 0.8,
            }}>
              {riskCounts[tab.key]}
            </span>
          </button>
        ))}
      </motion.div>

      {/* Search */}
      <motion.div {...anim} transition={{ ...anim.transition, delay: 0.06 }}
        style={{ position: 'relative', marginBottom: 'var(--space-lg)', maxWidth: 400 }}
      >
        <Search size={16} style={{
          position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--text-muted)',
        }} />
        <input
          className="input"
          placeholder="Search by name, school…"
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0) }}
          style={{ paddingLeft: 40 }}
        />
      </motion.div>

      {/* Table */}
      <motion.div className="card" {...anim} transition={{ ...anim.transition, delay: 0.12 }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Student</th>
                <th>School</th>
                <th>Grade</th>
                <th>Age</th>
                <th>Risk Score</th>
                <th>Risk Level</th>
                <th>Emotion</th>
                <th>Records</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {paginated.map((student, i) => (
                <tr key={student.id || i}>
                  <td>
                    <div className="student-cell">
                      <div className="student-avatar">
                        {(student.name || student.student_identifier || '?').slice(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 13.5 }}>
                          {student.name || student.student_identifier}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {student.student_identifier || student.id}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td style={{ color: 'var(--text-secondary)' }}>{student.school || '—'}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>{student.grade || '—'}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>{student.age || '—'}</td>
                  <td>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14,
                      color: (student.risk_score || 0) >= 70 ? 'var(--color-danger)' :
                             (student.risk_score || 0) >= 40 ? 'var(--color-warning)' : 'var(--color-success)',
                    }}>
                      {student.risk_score ?? '—'}
                    </span>
                  </td>
                  <td>
                    <span className={`risk-badge ${student.risk_level || 'low'}`}>
                      <span className="dot" />
                      {student.risk_level || 'low'}
                    </span>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', textTransform: 'capitalize', fontSize: 13 }}>
                    {student.latest_emotion || '—'}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5, color: 'var(--text-muted)' }}>
                    {student.records || student.behavioral_record_count || 0}
                  </td>
                  <td>
                    <button className="btn btn-ghost btn-icon btn-sm" title="View Profile">
                      <Eye size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: 'var(--space-md) 0 0',
            borderTop: '1px solid var(--border-subtle)',
            marginTop: 'var(--space-md)',
          }}>
            <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
              Showing {page * perPage + 1}–{Math.min((page + 1) * perPage, filtered.length)} of {filtered.length}
            </span>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                className="btn btn-ghost btn-icon btn-sm"
                disabled={page === 0}
                onClick={() => setPage(p => p - 1)}
              >
                <ChevronLeft size={16} />
              </button>
              <button
                className="btn btn-ghost btn-icon btn-sm"
                disabled={page >= totalPages - 1}
                onClick={() => setPage(p => p + 1)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  )
}
