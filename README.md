<div align="center">

# 🧠 MindMesh AI

### AI-Powered Mental Health Intelligence for School Ecosystems

**SIH 1433 — Smart India Hackathon**

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-6366f1?style=flat-square)](LICENSE)

*Proactive early-intervention system that detects at-risk students through unsupervised machine learning — without requiring any labelled training data.*

---

</div>

## 🎯 The Problem

**1 in 7 adolescents** experience mental health disorders globally. Schools lack scalable tools to identify struggling students before crisis points. Traditional approaches rely on:
- Manual counselor observation (limited scale)
- Self-reported surveys (response bias)
- Reactive intervention (too late)

## 💡 Our Solution

MindMesh AI continuously analyzes student behavioral signals — journal entries, check-ins, and engagement patterns — using a **fully unsupervised ML pipeline** that discovers patterns organically without pre-labelled data.

### How It Works

**Overview:** The system captures unstructured student inputs (daily emojis, journal texts) and transforms them into high-dimensional numerical vectors. It groups similar emotional states and extracts latent human stressors using clustering techniques (K-Means, NMF). Then, it feeds these quantified behavioral metrics into a **Three-Pillar Anomaly Fusion Engine** (Isolation Forest, GMM, LOF). If a student mathematically diverges from the school's normal behavioral baseline, the engines vote. Only when multiple mathematically distinct models reach a strict anomaly consensus does the system flag the student, generating an Explainable AI (XAI) alert for the counselor with exact geometric feature attributions.

```
Student Input → TF-IDF Embeddings → K-Means Emotion Clustering
                                  → NMF Topic Discovery
                                  → VADER Sentiment Analysis
                                  → Multi-View Anomaly Fusion (Isolation Forest + GMM + LOF)
                                  → Local Outlier & Probabilistic Profiling
                                  → Unsupervised Consensus Risk Score → Alert
```

When risk thresholds are crossed, counselors receive actionable alerts with **Explainable AI (XAI)** — not just a score, but mathematically backed feature attribution explaining *why* a student's behavior was flagged by the consensus protocol.

---

## 🏗️ Architecture

### Backend (FastAPI + Python ML)

| Module | Algorithm | Purpose |
|--------|-----------|---------|
| Text Embeddings | **TF-IDF** (scikit-learn) | Converts text → numerical vectors |
| Emotion Detection | **K-Means** Clustering | Discovers emotion clusters organically |
| Sentiment Analysis | **VADER** (NLTK) | Compound sentiment scoring |
| Anomaly Fusion | **Isolation Forest + GMM + LOF** | Multi-View behavioral outlier consensus |
| Topic Discovery | **NMF** | Latent theme extraction |
| XAI Attribution | **Simulated SHAP** | Mathematical feature importance proofs |
| Risk Scoring | **Consensus Validation** | Ensembled mathematical risk assessment |

### Frontend (React + Vite)

| Page | Features |
|------|----------|
| **Dashboard** | Real-time KPIs, emotion trends, risk distribution, radar charts |
| **Students** | Searchable/filterable student table with risk badges |
| **Alerts** | Expandable AI-generated alert cards with status management |
| **Analytics** | 6 interactive charts — scatter, area, bar, line, histogram |
| **ML Models** | Pipeline visualization, model registry, one-click training |
| **Journal** | Student portal with emoji mood check-in and real-time AI analysis |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker (Production)
```bash
docker compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/login` | JWT authentication |
| `GET` | `/analytics/dashboard` | Full dashboard payload |
| `POST` | `/emotion/analyze` | Real-time text analysis |
| `GET` | `/analytics/trends/emotion` | Emotion time-series |
| `GET` | `/analytics/risk/distribution` | Risk level distribution |
| `GET` | `/alerts/` | Paginated alert list |
| `POST` | `/models/train` | Train unsupervised models |
| `GET` | `/models/active` | Active model registry |
| `POST` | `/student/{id}/checkin` | Student mood check-in |

Full API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## 🔬 Unsupervised ML Pipeline

### Why Unsupervised?

Traditional supervised approaches require **labelled mental health data** — which is:
1. **Ethically problematic** to collect from minors
2. **Expensive** to annotate by clinical psychologists
3. **Not available** for most schools

Our unsupervised pipeline discovers patterns **from the data itself**:

- **K-Means** finds natural emotion clusters without predefined categories
- **Multi-View Anomaly Fusion Protocol** identifies behavioral outliers without labelled "at-risk" examples, relying on the mathematical consensus of Spatial, Density, and Neighborhood views (Isolation Forest, GMM, LOF).
- **NMF** extracts recurring themes students write about
- **Unsupervised Feature Attribution** provides SHAP-like explanations for complex anomaly outputs
- **VADER** provides rule-based sentiment as an unsupervised baseline

### Safety-Critical Override

Despite being unsupervised, the system **always checks for high-risk keywords** (e.g., self-harm, suicidal ideation) and immediately escalates to counselors — this is a hardcoded safety feature that overrides all ML predictions.

---

## 🛡️ Privacy & Ethics

- All data processed **on-premises** — no external API calls
- Student identifiers are anonymizable
- Journal content encrypted at rest (production)
- RBAC: Students see only their own data; counselors see aggregate + flagged
- Compliant with **FERPA** (US) and **DPDP Act** (India) guidelines

---

## 📁 Project Structure

```
mindmesh-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # Configuration management
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   ├── routes/                  # API route handlers
│   │   │   ├── analysis.py          # AI analysis endpoints
│   │   │   ├── analytics.py         # Dashboard data endpoints
│   │   │   ├── alerts.py            # Alert management
│   │   │   ├── models.py            # ML model management
│   │   │   └── student.py           # Student CRUD
│   │   ├── services/                # Business logic
│   │   │   ├── text_embeddings.py   # TF-IDF engine
│   │   │   ├── emotion_detection.py # K-Means clustering
│   │   │   ├── sentiment_analysis.py# VADER sentiment
│   │   │   ├── anomaly_detection.py # Isolation Forest
│   │   │   ├── topic_discovery.py   # NMF topic modeling
│   │   │   ├── student_clustering.py# GMM profiling
│   │   │   ├── risk_scoring.py      # Composite risk engine
│   │   │   ├── training_pipeline.py # Model training orchestrator
│   │   │   ├── model_evaluation.py  # Unsupervised metrics
│   │   │   └── model_registry.py    # Model versioning
│   │   └── database/                # DB schemas & connection
│   ├── tests/                       # Test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Router + auth context
│   │   ├── pages/                   # Dashboard, Students, Analytics…
│   │   ├── components/              # Sidebar, charts
│   │   └── lib/api.js               # API client
│   └── index.html
├── docker-compose.yml
├── Dockerfile
└── docs/
    ├── API_SPEC.md
    ├── DATA_SCHEMA.md
    └── TASKS.md
```

---

## 🏆 Built For

**Smart India Hackathon 2024 — Problem Statement SIH1433**
*AI-based platform for monitoring and enhancing mental well-being of students*

---

<div align="center">
<br>

**Built with ❤️ by Team MindMesh**

*Making mental health support proactive, not reactive.*

</div>
