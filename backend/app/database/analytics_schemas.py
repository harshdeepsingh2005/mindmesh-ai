"""MindMesh AI — Pydantic Schemas for Analytics API Responses.

Separated from the main schemas to keep the file manageable.
All schemas here are read-only response models for the /analytics endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


# ─── Overview ────────────────────────────────────────────────────


class OverviewStatsResponse(BaseModel):
    """Top-level dashboard overview statistics."""
    total_students: int
    total_records: int
    total_checkins: int
    total_journals: int
    total_risk_assessments: int
    open_alerts: int
    acknowledged_alerts: int
    high_risk_students: int
    medium_risk_students: int
    low_risk_students: int
    unassessed_students: int
    avg_risk_score: Optional[float] = None
    avg_emotion_score: Optional[float] = None
    avg_sentiment_score: Optional[float] = None
    period_days: int


# ─── Emotion Distribution ───────────────────────────────────────


class EmotionDistributionItem(BaseModel):
    """Single emotion bucket."""
    emotion: str
    count: int
    percentage: float


class EmotionDistributionResponse(BaseModel):
    """Full emotion distribution."""
    period_days: int
    student_id: Optional[str] = None
    distribution: List[EmotionDistributionItem]
    total_analyses: int


# ─── Risk Distribution / Heatmap ────────────────────────────────


class RiskBucketResponse(BaseModel):
    """Single risk-level bucket."""
    risk_level: str
    count: int
    percentage: float
    student_ids: List[str] = []


class RiskDistributionResponse(BaseModel):
    """Risk distribution across students."""
    buckets: List[RiskBucketResponse]
    total_assessed: int


class RiskHistogramBucket(BaseModel):
    """Single histogram bucket."""
    range: str
    low: int
    high: int
    count: int


class RiskHistogramResponse(BaseModel):
    """Risk score histogram."""
    bucket_size: int
    buckets: List[RiskHistogramBucket]
    total_assessed: int


# ─── Time Series ────────────────────────────────────────────────


class TimeSeriesPointResponse(BaseModel):
    """Single time-series data point."""
    date: str
    value: float
    count: int = 0


class TimeSeriesResponse(BaseModel):
    """Time-series data for charting."""
    metric: str
    period_days: int
    student_id: Optional[str] = None
    data_points: List[TimeSeriesPointResponse]
    total_points: int


# ─── Activity Breakdown ─────────────────────────────────────────


class ActivityDayResponse(BaseModel):
    """Activity counts for a single day."""
    date: str
    checkin: int = 0
    journal: int = 0
    survey: int = 0


class ActivityBreakdownResponse(BaseModel):
    """Daily activity breakdown for stacked bar chart."""
    period_days: int
    daily: List[ActivityDayResponse]
    total_days: int


# ─── Alert Summary ──────────────────────────────────────────────


class AlertDailyTrend(BaseModel):
    """Alert count for a single day."""
    date: str
    count: int


class AlertSummaryResponse(BaseModel):
    """Alert statistics for the dashboard."""
    period_days: int
    total_alerts: int
    by_status: Dict[str, int]
    by_type: Dict[str, int]
    daily_trend: List[AlertDailyTrend]


# ─── Student Summaries (Dashboard Table) ────────────────────────


class StudentSummaryResponse(BaseModel):
    """Compact per-student dashboard summary."""
    student_id: str
    student_identifier: str
    school: str
    grade: str
    latest_risk_score: Optional[int] = None
    latest_risk_level: Optional[str] = None
    avg_emotion_score: Optional[float] = None
    avg_sentiment_score: Optional[float] = None
    record_count: int = 0
    open_alert_count: int = 0
    last_activity: Optional[str] = None


class StudentSummaryListResponse(BaseModel):
    """Paginated student summary table."""
    students: List[StudentSummaryResponse]
    total: int
    period_days: int


# ─── School Stats ───────────────────────────────────────────────


class SchoolStatsItem(BaseModel):
    """Per-school aggregated statistics."""
    school: str
    student_count: int
    avg_emotion_score: Optional[float] = None
    record_count: int = 0
    open_alert_count: int = 0


class SchoolStatsResponse(BaseModel):
    """School-level statistics."""
    period_days: int
    schools: List[SchoolStatsItem]


# ─── Compound Dashboard Response ────────────────────────────────


class DashboardResponse(BaseModel):
    """All-in-one dashboard payload (reduces API calls).

    Combines overview, risk distribution, emotion distribution,
    recent emotion trend, alert summary, and top at-risk students.
    """
    overview: OverviewStatsResponse
    risk_distribution: RiskDistributionResponse
    emotion_distribution: EmotionDistributionResponse
    emotion_trend: TimeSeriesResponse
    alert_summary: AlertSummaryResponse
    at_risk_students: List[StudentSummaryResponse]
