"""MindMesh AI — Risk Prediction Engine (Unsupervised).

Scores student risk using Isolation Forest anomaly detection
combined with sentiment trend analysis.  Unlike the previous
version with hardcoded factor weights, this engine LEARNS what
constitutes "abnormal" behaviour from the data itself.

Risk pipeline:
  1. Build a BehavioralFeatureVector from recent records
  2. Run Isolation Forest anomaly detection
  3. Combine anomaly score with VADER sentiment signals
  4. Map to risk level (low / medium / high)
  5. Persist RiskScore + auto-generate Alert on high risk

Risk Levels:
    LOW      — 0-39   (routine monitoring)
    MEDIUM   — 40-69  (increased attention)
    HIGH     — 70-100 (immediate intervention recommended)
"""

from __future__ import annotations

import uuid
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.student import Student
from ..models.behavioral_record import BehavioralRecord
from ..models.emotion_analysis import EmotionAnalysis
from ..models.risk_score import RiskScore
from ..models.alert import Alert
from ..logging_config import logger

from .sentiment_analysis import analyze_sentiment
from .emotion_detection import detect_emotion
from .anomaly_detection import (
    AnomalyDetectionEngine,
    BehavioralFeatureVector,
    AnomalyResult,
    get_anomaly_engine,
)

# ── Constants ────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "low": (0, 39),
    "medium": (40, 69),
    "high": (70, 100),
}

LOOKBACK_DAYS = 30  # sliding window for feature computation

FACTOR_WEIGHTS = {
    "anomaly_score": 0.30,
    "sentiment_score": 0.20,
    "emotion_intensity": 0.15,
    "high_risk_keywords": 0.15,
    "mood_variability": 0.10,
    "behavioral_frequency": 0.05,
    "trend_direction": 0.05,
}


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class RiskFactors:
    """Individual factor scores (each 0-100)."""
    anomaly_score: float = 0.0         # Isolation Forest output
    sentiment_score: float = 0.0       # avg negative sentiment
    emotion_intensity: float = 0.0     # proportion of distress
    behavioral_frequency: float = 0.0  # activity level
    mood_variability: float = 0.0      # mood stability
    high_risk_keywords: float = 0.0    # flagged content
    trend_direction: float = 0.0       # deteriorating trend

    def to_dict(self) -> Dict[str, float]:
        return {
            "anomaly_score": round(self.anomaly_score, 2),
            "sentiment_score": round(self.sentiment_score, 2),
            "emotion_intensity": round(self.emotion_intensity, 2),
            "behavioral_frequency": round(self.behavioral_frequency, 2),
            "mood_variability": round(self.mood_variability, 2),
            "high_risk_keywords": round(self.high_risk_keywords, 2),
            "trend_direction": round(self.trend_direction, 2),
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    student_id: str
    composite_score: int
    risk_level: str
    factors: RiskFactors
    anomaly_result: Optional[AnomalyResult] = None
    contributing_features: List[Tuple[str, float]] = field(default_factory=list)
    assessed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high"


# ── Helpers ──────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def get_dynamic_thresholds() -> Dict[str, Tuple[int, int]]:
    """Adjust risk thresholds dynamically based on the academic calendar.
    
    During high-stress periods (like midterms in March/October, 
    finals in May/December), we lower the threshold for 'high' risk 
    to be more sensitive and catch at-risk students earlier.
    """
    now = datetime.now(timezone.utc)
    # E.g. March (Midterms), May (Finals), October (Midterms), December (Finals)
    is_high_stress_period = now.month in [3, 5, 10, 12]
    
    if is_high_stress_period:
        return {
            "low": (0, 30),
            "medium": (31, 59),
            "high": (60, 100),   # High risk triggers at 60 instead of 70
        }
    
    return {
        "low": (0, 39),
        "medium": (40, 69),
        "high": (70, 100),       # Standard threshold
    }


def _risk_level_from_score(score: int) -> str:
    thresholds = get_dynamic_thresholds()
    for level, (lo, hi) in thresholds.items():
        if lo <= score <= hi:
            return level
    return "high"


# ── Feature Vector Construction ──────────────────────────────────

async def build_feature_vector(
    db: AsyncSession,
    student_id: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> BehavioralFeatureVector:
    """Build a behavioural feature vector from database records.

    Aggregates sentiment, emotion, activity, and mood data over
    the lookback window into a single feature vector for ML models.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Fetch recent records
    stmt = (
        select(BehavioralRecord)
        .where(
            BehavioralRecord.student_id == student_id,
            BehavioralRecord.timestamp >= cutoff,
        )
        .order_by(desc(BehavioralRecord.timestamp))
    )
    result = await db.execute(stmt)
    records: List[BehavioralRecord] = list(result.scalars().all())

    fv = BehavioralFeatureVector(student_id=student_id)

    if not records:
        return fv

    # ── Sentiment Features ───────────────────────────────────
    sentiment_scores = []
    negative_count = 0
    high_risk_count = 0

    for r in records:
        text = r.text_input or ""
        if text:
            sr = analyze_sentiment(text)
            sentiment_scores.append(sr.sentiment_score)
            if sr.sentiment_label == "negative":
                negative_count += 1
            if sr.high_risk_flag:
                high_risk_count += 1

    if sentiment_scores:
        fv.avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        if len(sentiment_scores) > 1:
            mean_s = fv.avg_sentiment
            fv.sentiment_std = (
                sum((s - mean_s) ** 2 for s in sentiment_scores)
                / (len(sentiment_scores) - 1)
            ) ** 0.5

    fv.negative_ratio = negative_count / max(len(records), 1)
    fv.high_risk_flags = high_risk_count

    # ── Emotion Features ─────────────────────────────────────
    emotion_results = []
    for r in records:
        text = r.text_input or ""
        if text:
            er = detect_emotion(text)
            emotion_results.append(er)

    if emotion_results:
        # Dominant cluster
        cluster_counts = Counter(er.predicted_cluster for er in emotion_results)
        fv.dominant_cluster = cluster_counts.most_common(1)[0][0]

        # Entropy of cluster distribution
        total = len(emotion_results)
        probs = [c / total for c in cluster_counts.values()]
        fv.emotion_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Distress ratio (clusters with negative labels)
        distress_labels = {"distress", "anxiety", "anger"}
        distress_count = sum(
            1 for er in emotion_results
            if any(dl in er.cluster_label.lower() for dl in distress_labels)
        )
        fv.distress_ratio = distress_count / total

    # ── Activity Features ────────────────────────────────────
    weeks = max(lookback_days / 7, 1)
    fv.entries_per_week = len(records) / weeks

    journal_count = sum(1 for r in records if r.activity_type == "journal")
    fv.journal_ratio = journal_count / max(len(records), 1)

    if records:
        latest = max(r.timestamp for r in records)
        fv.days_since_last_entry = (
            datetime.now(timezone.utc) - latest
        ).total_seconds() / 86400

    # ── Mood Features (derived from emotion_score) ─────────
    mood_ratings = [
        float(r.emotion_score) for r in records
        if r.emotion_score is not None
    ]
    if mood_ratings:
        fv.avg_mood = sum(mood_ratings) / len(mood_ratings)
        if len(mood_ratings) > 1:
            mean_m = fv.avg_mood
            fv.mood_std = (
                sum((m - mean_m) ** 2 for m in mood_ratings)
                / (len(mood_ratings) - 1)
            ) ** 0.5

    return fv


# ── Risk Scoring ─────────────────────────────────────────────────

async def compute_risk_factors(
    db: AsyncSession,
    student_id: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> Tuple[RiskFactors, BehavioralFeatureVector, Optional[AnomalyResult]]:
    """Compute risk factors using anomaly detection + sentiment signals.

    Returns:
        Tuple of (RiskFactors, BehavioralFeatureVector, AnomalyResult).
    """
    fv = await build_feature_vector(db, student_id, lookback_days)
    factors = RiskFactors()

    # Run anomaly detection
    anomaly_engine = get_anomaly_engine()
    anomaly_result = anomaly_engine.predict(fv)

    # Map anomaly score to 0-100 (more anomalous = higher)
    # Isolation Forest scores: ~[-0.5, 0.5], negative = more anomalous
    anomaly_normalized = _clamp((0.5 - anomaly_result.anomaly_score) * 100)
    factors.anomaly_score = anomaly_normalized

    # Sentiment-based risk (negative sentiment = higher risk)
    factors.sentiment_score = _clamp((1 - (fv.avg_sentiment + 1) / 2) * 100)

    # Emotion intensity (distress ratio)
    factors.emotion_intensity = _clamp(fv.distress_ratio * 100)

    # Activity frequency (both extremes are concerning)
    if fv.entries_per_week < 1:
        factors.behavioral_frequency = _clamp((1 - fv.entries_per_week) * 50)
    elif fv.entries_per_week > 10:
        factors.behavioral_frequency = _clamp((fv.entries_per_week - 10) * 10)
    else:
        factors.behavioral_frequency = 0.0

    # Mood variability
    factors.mood_variability = _clamp(fv.mood_std * 40)

    # High-risk keyword flags (immediate escalation)
    factors.high_risk_keywords = _clamp(fv.high_risk_flags * 50)

    return factors, fv, anomaly_result


def calculate_composite_score(
    factors: RiskFactors,
    anomaly_result: Optional[AnomalyResult] = None,
) -> int:
    """Calculate composite risk score.

    Anomaly detection is the PRIMARY signal (40% weight).
    Sentiment and emotion are SECONDARY signals.
    High-risk keywords override everything.
    """
    fd = factors.to_dict()

    # Weights: anomaly detection is the primary unsupervised signal
    weighted = sum(fd.get(k, 0) * w for k, w in FACTOR_WEIGHTS.items())

    # High-risk keyword override: floor at 70
    if fd.get("high_risk_keywords", 0) > 0:
        weighted = max(weighted, 70)

    return int(_clamp(weighted))


# ── Main API ─────────────────────────────────────────────────────

async def assess_student_risk(
    db: AsyncSession,
    student_id: str,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    create_alert_on_high: bool = True,
) -> RiskAssessment:
    """Run a full risk assessment for one student.

    1. Build behavioural feature vector from recent records
    2. Run Isolation Forest anomaly detection
    3. Compute risk factors
    4. Calculate composite score
    5. Persist RiskScore row
    6. Create Alert if high-risk
    """
    factors, fv, anomaly_result = await compute_risk_factors(
        db, student_id, lookback_days=lookback_days
    )
    composite = calculate_composite_score(factors, anomaly_result)
    level = _risk_level_from_score(composite)

    contributing = (
        anomaly_result.contributing_features
        if anomaly_result
        else []
    )

    assessment = RiskAssessment(
        student_id=student_id,
        composite_score=composite,
        risk_level=level,
        factors=factors,
        anomaly_result=anomaly_result,
        contributing_features=contributing,
    )

    # ── Persist RiskScore ────────────────────────────────────
    risk_row = RiskScore(
        id=str(uuid.uuid4()),
        student_id=student_id,
        risk_score=composite,
        risk_level=level,
        contributing_factors=factors.to_dict(),
    )
    db.add(risk_row)

    # ── Auto-alert on HIGH risk ──────────────────────────────
    if assessment.is_high_risk and create_alert_on_high:
        contributing_str = ", ".join(
            f"{name}: {score:.2f}" for name, score in contributing[:3]
        )
        alert = Alert(
            id=str(uuid.uuid4()),
            student_id=student_id,
            risk_score=composite,
            alert_type="high_risk",
            message=(
                f"Student {student_id} scored {composite}/100 "
                f"(level: {level}). "
                f"Key factors: {contributing_str}. "
                f"Immediate attention recommended."
            ),
        )
        db.add(alert)
        logger.warning(
            f"HIGH RISK ALERT — student={student_id}, "
            f"score={composite}, level={level}, "
            f"anomaly_score={factors.anomaly_score:.2f}"
        )

    await db.commit()
    await db.refresh(risk_row)

    logger.info(
        f"Risk assessment: student={student_id}, "
        f"score={composite}, level={level}, "
        f"anomaly={factors.anomaly_score:.2f}"
    )

    return assessment


# ── Query Helpers ────────────────────────────────────────────────

async def get_risk_history(
    db: AsyncSession,
    student_id: str,
    *,
    limit: int = 20,
) -> List[RiskScore]:
    """Return the most recent risk scores for a student."""
    stmt = (
        select(RiskScore)
        .where(RiskScore.student_id == student_id)
        .order_by(desc(RiskScore.calculated_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_latest_risk_score(
    db: AsyncSession,
    student_id: str,
) -> Optional[RiskScore]:
    """Return the single most-recent risk score."""
    stmt = (
        select(RiskScore)
        .where(RiskScore.student_id == student_id)
        .order_by(desc(RiskScore.calculated_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def batch_assess_students(
    db: AsyncSession,
    student_ids: List[str],
    *,
    lookback_days: int = LOOKBACK_DAYS,
) -> List[RiskAssessment]:
    """Run risk assessments for multiple students."""
    assessments: List[RiskAssessment] = []
    for sid in student_ids:
        try:
            a = await assess_student_risk(
                db, sid, lookback_days=lookback_days
            )
            assessments.append(a)
        except Exception as exc:
            logger.error(f"Risk assessment failed for student={sid}: {exc}")
    return assessments
