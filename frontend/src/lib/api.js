const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function getToken() {
  return localStorage.getItem('mindmesh_token')
}

async function request(path, options = {}) {
  const token = getToken()
  const headers = {
    'Content-Type': 'application/json',
    'Bypass-Tunnel-Reminder': 'true',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...options.headers,
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })

  if (res.status === 401) {
    localStorage.removeItem('mindmesh_token')
    localStorage.removeItem('mindmesh_user')
    window.location.href = '/login'
    throw new Error('Unauthorized')
  }

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export const api = {
  // Auth
  login: (email, password) =>
    request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  // Dashboard
  getDashboard: (days = 30) =>
    request(`/analytics/dashboard?days=${days}`),

  getOverview: (days = 30) =>
    request(`/analytics/overview?days=${days}`),

  // Analytics
  getEmotionDistribution: (days = 30) =>
    request(`/analytics/emotions/distribution?days=${days}`),

  getRiskDistribution: () =>
    request('/analytics/risk/distribution'),

  getRiskHistogram: (bucketSize = 10) =>
    request(`/analytics/risk/histogram?bucket_size=${bucketSize}`),

  getEmotionTrend: (days = 30) =>
    request(`/analytics/trends/emotion?days=${days}`),

  getSentimentTrend: (days = 30) =>
    request(`/analytics/trends/sentiment?days=${days}`),

  getRiskTrend: (days = 30) =>
    request(`/analytics/trends/risk?days=${days}`),

  getActivityBreakdown: (days = 30) =>
    request(`/analytics/activity/breakdown?days=${days}`),

  getAlertSummary: (days = 30) =>
    request(`/analytics/alerts/summary?days=${days}`),

  getStudentSummaries: (params = {}) => {
    const query = new URLSearchParams({
      days: params.days || 30,
      skip: params.skip || 0,
      limit: params.limit || 50,
      ...(params.risk_level ? { risk_level: params.risk_level } : {}),
    })
    return request(`/analytics/students?${query}`)
  },

  getSchoolStats: (days = 30) =>
    request(`/analytics/schools?days=${days}`),

  // Students
  getStudents: (skip = 0, limit = 50) =>
    request(`/student/?skip=${skip}&limit=${limit}`),

  getStudent: (id) =>
    request(`/student/${id}`),

  createStudent: (data) =>
    request('/student/', { method: 'POST', body: JSON.stringify(data) }),

  // Analysis
  analyzeText: (text) =>
    request('/emotion/analyze', {
      method: 'POST',
      body: JSON.stringify({ text_input: text }),
    }),

  getStudentTrends: (studentId, days = 30) =>
    request(`/emotion/trends/${studentId}?days=${days}`),

  // Alerts
  getAlerts: (params = {}) => {
    const query = new URLSearchParams({
      skip: params.skip || 0,
      limit: params.limit || 50,
      ...(params.status ? { status: params.status } : {}),
    })
    return request(`/alerts/?${query}`)
  },

  acknowledgeAlert: (alertId) =>
    request(`/alerts/${alertId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status: 'acknowledged' }),
    }),

  resolveAlert: (alertId) =>
    request(`/alerts/${alertId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status: 'resolved' }),
    }),

  getAlertCount: () =>
    request('/alerts/count'),

  getAlertPrimer: (alertId) =>
    request(`/alerts/${alertId}/primer`),

  // Models
  getModels: () =>
    request('/models/'),

  getActiveModels: () =>
    request('/models/active'),

  trainModels: (data) =>
    request('/models/train', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  evaluateLiveText: (text) =>
    request('/models/evaluate', {
      method: 'POST',
      body: JSON.stringify({ text }),
    }),

  promoteModel: (modelName, version) =>
    request('/models/promote', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName, version }),
    }),

  // Risk
  assessRisk: (studentId, lookbackDays = 30) =>
    request('/student/risk/assess', {
      method: 'POST',
      body: JSON.stringify({ student_id: studentId, lookback_days: lookbackDays }),
    }),

  // Journal / Check-in
  submitCheckin: (studentId, moodRating, notes) =>
    request(`/student/${studentId}/checkin`, {
      method: 'POST',
      body: JSON.stringify({ mood_rating: moodRating, notes }),
    }),

  submitJournal: (studentId, text, moodTag) =>
    request(`/student/journal`, {
      method: 'POST',
      body: JSON.stringify({ text, mood_tag: moodTag }),
    }),

  // Crisis
  triggerSOS: (location, notes) =>
    request('/student/sos', {
      method: 'POST',
      body: JSON.stringify({ location, notes }),
    }),

  reportPeer: (peerIdentifier, concern) =>
    request('/student/report_peer', {
      method: 'POST',
      body: JSON.stringify({ peer_identifier: peerIdentifier, concern }),
    }),

  // Health
  health: () => request('/health'),
}
