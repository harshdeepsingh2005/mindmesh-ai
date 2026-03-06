"""MindMesh AI — Pydantic Schemas for Request/Response Validation."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, List
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


# ─── Alert Schemas ──────────────────────────────────────────


class AlertCreate(BaseModel):
    """Schema for creating an alert."""
    student_id: str
    risk_score: int = Field(..., ge=0, le=100)
    alert_type: str = Field(..., pattern="^(high_risk|info)$")
    message: str


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


# ─── Health Check ───────────────────────────────────────────


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    app_name: str
    version: str
    database: str
    timestamp: datetime
