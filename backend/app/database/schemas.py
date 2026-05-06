"""MindMesh AI — Pydantic Schemas for Request/Response Validation."""

from pydantic import BaseModel, EmailStr, Field
from typing import Any, Optional, Dict, List
from datetime import datetime


# ─── Auth Schemas ───────────────────────────────────────────


class TokenRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str


# ─── User Schemas ───────────────────────────────────────────


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    role: str = Field(..., pattern="^(student|teacher|admin)$")
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """Schema for user response."""
    id: str
    name: str
    email: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating a user profile."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None


class UserListResponse(BaseModel):
    """Schema for listing multiple users."""
    users: List[UserResponse]
    total: int


# ─── Student Schemas ────────────────────────────────────────


class StudentCreate(BaseModel):
    """Schema for creating a student profile."""
    student_identifier: str = Field(..., min_length=1, max_length=255)
    age: int = Field(..., ge=5, le=20)
    school: str = Field(..., min_length=1, max_length=255)
    grade: str = Field(..., min_length=1, max_length=50)
    guardian_contact: Optional[str] = None


class StudentUpdate(BaseModel):
    """Schema for updating a student profile."""
    age: Optional[int] = Field(None, ge=5, le=20)
    school: Optional[str] = Field(None, min_length=1, max_length=255)
    grade: Optional[str] = Field(None, min_length=1, max_length=50)
    guardian_contact: Optional[str] = None


class StudentResponse(BaseModel):
    """Schema for student response."""
    id: str
    student_identifier: str
    age: int
    school: str
    grade: str
    guardian_contact: Optional[str] = None

    class Config:
        from_attributes = True


class StudentDetailResponse(BaseModel):
    """Schema for detailed student response with user info."""
    id: str
    user_id: str
    student_identifier: str
    age: int
    school: str
    grade: str
    guardian_contact: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    behavioral_record_count: int = 0
    latest_risk_level: Optional[str] = None

    class Config:
        from_attributes = True


class StudentListResponse(BaseModel):
    """Schema for listing multiple students."""
    students: List[StudentResponse]
    total: int


# ─── Mood / Journal Schemas ─────────────────────────────────


class MoodCheckinRequest(BaseModel):
    """Schema for mood check-in submission."""
    mood_rating: int = Field(..., ge=1, le=10, description="Mood rating 1-10")
    notes: Optional[str] = Field(None, max_length=2000)


class JournalEntryRequest(BaseModel):
    """Schema for journal entry submission."""
    text: str = Field(..., min_length=1, max_length=5000)
    mood_tag: Optional[str] = Field(None, max_length=50)


# ─── Behavioral Record Schemas ──────────────────────────────


class BehavioralRecordCreate(BaseModel):
    """Schema for creating a behavioral record."""
    student_id: str
    text_input: Optional[str] = None
    activity_type: str = Field(..., pattern="^(survey|journal|checkin)$")
    emotion_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)


class BehavioralRecordResponse(BaseModel):
    """Schema for behavioral record response."""
    id: str
    student_id: str
    timestamp: datetime
    text_input: Optional[str] = None
    activity_type: str
    emotion_score: Optional[float] = None
    sentiment_score: Optional[float] = None

    class Config:
        from_attributes = True


class BehavioralRecordListResponse(BaseModel):
    """Schema for listing multiple behavioral records."""
    records: List[BehavioralRecordResponse]
    total: int


# ─── Emotion Analysis Schemas ───────────────────────────────


class EmotionAnalysisCreate(BaseModel):
    """Schema for creating emotion analysis."""
    record_id: str
    predicted_emotion: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class EmotionAnalysisResponse(BaseModel):
    """Schema for emotion analysis response."""
    id: str
    record_id: str
    predicted_emotion: str
    confidence_score: float
    model_version: str

    class Config:
        from_attributes = True


class EmotionAnalyzeRequest(BaseModel):
    """Schema for standalone emotion analysis request."""
    text_input: str = Field(..., min_length=1, max_length=5000)
    student_id: Optional[str] = None


class EmotionAnalyzeResponse(BaseModel):
    """Schema for standalone emotion analysis response."""
    emotion: Dict
    sentiment: Dict
    topic: Optional[Dict] = None
    metrics: Optional[Dict[str, Any]] = None


class TrendAnalysisResponse(BaseModel):
    """Schema for behavioral trend analysis response."""
    student_id: str
    period_days: int
    total_records: int
    data_points: int
    emotion_trend: str
    emotion_slope: float
    avg_emotion_score: Optional[float] = None
    emotion_volatility: float
    checkin_count: int
    journal_count: int
    checkin_frequency: float
    weekly_averages: List[Dict]
    declining_flag: bool
    disengagement_flag: bool


# ─── Risk Score Schemas ─────────────────────────────────────


class RiskScoreCreate(BaseModel):
    """Schema for creating a risk score."""
    student_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(..., pattern="^(low|medium|high)$")
    contributing_factors: Optional[Dict[str, float]] = None


class RiskScoreResponse(BaseModel):
    """Schema for risk score response."""
    id: str
    student_id: str
    risk_score: int
    risk_level: str
    contributing_factors: Optional[Dict[str, float]] = None
    calculated_at: datetime

    class Config:
        from_attributes = True


class RiskScoreListResponse(BaseModel):
    """Schema for a list of risk scores."""
    student_id: str
    scores: List[RiskScoreResponse]
    total: int


class RiskAssessmentRequest(BaseModel):
    """Schema for requesting a risk assessment."""
    student_id: str
    lookback_days: Optional[int] = Field(default=30, ge=7, le=180)


class RiskAssessmentResponse(BaseModel):
    """Schema for risk assessment result."""
    student_id: str
    composite_score: int = Field(..., ge=0, le=100)
    risk_level: str
    contributing_factors: Dict[str, float]
    assessed_at: datetime


class BatchRiskAssessmentRequest(BaseModel):
    """Schema for batch risk assessment."""
    student_ids: List[str] = Field(..., min_length=1, max_length=50)
    lookback_days: Optional[int] = Field(default=30, ge=7, le=180)


class BatchRiskAssessmentResponse(BaseModel):
    """Schema for batch risk assessment results."""
    assessments: List[RiskAssessmentResponse]
    total: int


class RiskThresholdsResponse(BaseModel):
    """Schema for risk threshold configuration."""
    thresholds: Dict[str, Dict[str, int]]
    factor_weights: Dict[str, float]


# ─── Alert Schemas ──────────────────────────────────────────


class AlertCreate(BaseModel):
    """Schema for creating an alert manually."""
    student_id: str
    risk_score: int = Field(..., ge=0, le=100)
    alert_type: str = Field(..., pattern="^(high_risk|info|sos|peer_concern)$")
    message: str = Field(..., min_length=1, max_length=2000)


class AlertResponse(BaseModel):
    """Schema for alert response."""
    id: str
    student_id: str
    risk_score: int
    alert_type: str
    message: str
    created_at: datetime
    status: str

    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Schema for listing multiple alerts."""
    alerts: List[AlertResponse]
    total: int


class AlertStatusUpdate(BaseModel):
    """Schema for updating an alert's status."""
    status: str = Field(
        ...,
        pattern="^(acknowledged|resolved|dismissed)$",
        description="New status: acknowledged, resolved, or dismissed",
    )


class SOSRequest(BaseModel):
    """Schema for SOS request."""
    location: Optional[str] = None
    notes: Optional[str] = None


class PeerReportRequest(BaseModel):
    """Schema for reporting a peer."""
    peer_identifier: str = Field(..., min_length=1, max_length=255)
    concern: str = Field(..., min_length=10, max_length=2000)



class AlertBulkAcknowledge(BaseModel):
    """Schema for bulk-acknowledging alerts."""
    alert_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of alert IDs to acknowledge",
    )


class AlertCountResponse(BaseModel):
    """Schema for open alert count (dashboard badge)."""
    open_count: int


class AlertNotificationResponse(BaseModel):
    """Schema for notification dispatch result."""
    alert_id: str
    notifications_sent: int
    notifications: List[Dict[str, Any]]


# ─── Health Check ───────────────────────────────────────────


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    app_name: str
    version: str
    database: str
    timestamp: datetime
