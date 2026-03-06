"""MindMesh AI — Risk Prediction Engine.

Multi-factor composite risk scoring system.  Each student is scored on a 0-100
scale by weighting seven behavioral / emotional factors derived from recent
records.  Thresholds map the composite score to a human-readable risk level.

Risk Levels:
    LOW      — 0-39   (routine monitoring)
    MEDIUM   — 40-69  (increased attention)
    HIGH     — 70-100 (immediate intervention recommended)

Factor Weights (must sum to 1.0):
    sentiment_score        0.20
    emotion_intensity      0.20
    high_risk_keywords     0.15
    behavioral_frequency   0.15
    trend_direction        0.15
    mood_variability       0.10
    journal_sentiment      0.05

An alert is automatically created when the composite score ≥ 70.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

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
from .trend_analysis import analyze_trend

# ── Constants ────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "low": (0, 39),
    "medium": (40, 69),
    "high": (70, 100),
}

FACTOR_WEIGHTS: Dict[str, float] = {
    "sentiment_score": 0.20,
    "emotion_intensity": 0.20,
    "high_risk_keywords": 0.15,
    "behavioral_frequency": 0.15,
    "trend_direction": 0.15,
    "mood_variability": 0.10,
    "journal_sentiment": 0.05,
}

LOOKBACK_DAYS = 30  # sliding window for factor computation


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class RiskFactors:
    """Individual factor scores (each 0-100)."""

    sentiment_score: float = 0.0
    emotion_intensity: float = 0.0
    high_risk_keywords: float = 0.0
    behavioral_frequency: float = 0.0
    trend_direction: float = 0.0
    mood_variability: float = 0.0
    journal_sentiment: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "sentiment_score": round(self.sentiment_score, 2),
            "emotion_intensity": round(self.emotion_intensity, 2),
            "high_risk_keywords": round(self.high_risk_keywords, 2),
            "behavioral_frequency": round(self.behavioral_frequency, 2),
            "trend_direction": round(self.trend_direction, 2),
            "mood_variability": round(self.mood_variability, 2),
            "journal_sentiment": round(self.journal_sentiment, 2),
        }


@dataclass
class RiskAssessment:
    """Complete assessment result returned to callers."""

    student_id: str
    composite_score: int
    risk_level: str
    factors: RiskFactors
    assessed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "high"


# ── Helper Utilities ─────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _risk_level_from_score(score: int) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score <= hi:
            return level
    return "high"  # fallback


# ── Factor Computation ───────────────────────────────────────────

async def compute_risk_factors(
    db: AsyncSession,
    student_id: str,
    *,
    lookback_days: int = LOOKBACK_DAYS,
) -> RiskFactors:
    """Compute all seven risk factors from the student's recent data.

    Each factor is normalised to a 0-100 scale where **higher = more risk**.
    """

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    factors = RiskFactors()

    # ── 1. Recent behavioral records ─────────────────────────────
    stmt = (
        select(BehavioralRecord)
        .where(
            BehavioralRecord.student_id == student_id,
            BehavioralRecord.created_at >= cutoff,
        )
        .order_by(desc(BehavioralRecord.created_at))
    )
    result = await db.execute(stmt)
    records: List[BehavioralRecord] = list(result.scalars().all())

    if not records:
        return factors  # all zeros — no data

    # ── 2. Emotion analyses for those records ────────────────────
    record_ids = [r.id for r in records]
    stmt_ea = (
        select(EmotionAnalysis)
        .where(EmotionAnalysis.record_id.in_(record_ids))
    )
    ea_result = await db.execute(stmt_ea)
    analyses: List[EmotionAnalysis] = list(ea_result.scalars().all())

    # ── Factor: sentiment_score ──────────────────────────────────
    # Average negative sentiment across records (inverted: more negative → higher risk)
    if records:
        sentiment_scores = []
        for r in records:
            text = r.text_input or r.mood_rating or ""
            if text:
                sr = analyze_sentiment(str(text))
                sentiment_scores.append(sr.negative_score * 100)
        if sentiment_scores:
            factors.sentiment_score = _clamp(
                sum(sentiment_scores) / len(sentiment_scores)
            )

    # ── Factor: emotion_intensity ────────────────────────────────
    # High confidence in negative emotions → higher risk
    negative_emotions = {"sadness", "anger", "fear"}
    if analyses:
        neg_confs = [
            a.confidence_score * 100
            for a in analyses
            if a.predicted_emotion in negative_emotions
        ]
        if neg_confs:
            factors.emotion_intensity = _clamp(
                sum(neg_confs) / len(neg_confs)
            )

    # ── Factor: high_risk_keywords ───────────────────────────────
    # Proportion of records flagged for high-risk keywords
    flagged = 0
    for r in records:
        text = r.text_input or ""
        if text:
            sr = analyze_sentiment(text)
            if sr.high_risk_flag:
                flagged += 1
    if records:
        factors.high_risk_keywords = _clamp((flagged / len(records)) * 100)

    # ── Factor: behavioral_frequency ─────────────────────────────
    # More records in a shorter window → higher concern  (cap at 30 records)
    factors.behavioral_frequency = _clamp(
        (len(records) / 30) * 100
    )

    # ── Factor: trend_direction ──────────────────────────────────
    # Worsening trend → higher risk  (negative slope in mood trend)
    if len(records) >= 3:
        trend = await analyze_trend(db, student_id, days=lookback_days)
        if trend and trend.get("mood_trend"):
            slope = trend["mood_trend"].get("slope", 0)
            # negative slope means deteriorating, scale ×20 and clamp
            factors.trend_direction = _clamp(max(0, -slope * 20) * 100 / 5)

    # ── Factor: mood_variability ─────────────────────────────────
    mood_ratings = [
        int(r.mood_rating) for r in records
        if r.mood_rating and str(r.mood_rating).isdigit()
    ]
    if len(mood_ratings) >= 2:
        mean_m = sum(mood_ratings) / len(mood_ratings)
        variance = sum((m - mean_m) ** 2 for m in mood_ratings) / len(mood_ratings)
        std_dev = variance ** 0.5
        # Normalise: std_dev of 2 on a 1-5 scale → 100
        factors.mood_variability = _clamp((std_dev / 2) * 100)

    # ── Factor: journal_sentiment ────────────────────────────────
    journal_negs: list[float] = []
    for r in records:
        if r.record_type == "journal" and r.text_input:
            sr = analyze_sentiment(r.text_input)
            journal_negs.append(sr.negative_score * 100)
    if journal_negs:
        factors.journal_sentiment = _clamp(
            sum(journal_negs) / len(journal_negs)
        )

    return factors


# ── Composite Scoring ────────────────────────────────────────────

def calculate_composite_score(factors: RiskFactors) -> int:
    """Weighted sum of individual factor scores → 0-100 int."""
    fd = factors.to_dict()
    weighted = sum(fd[k] * FACTOR_WEIGHTS[k] for k in FACTOR_WEIGHTS)
    return int(_clamp(weighted))


# ── Main Assessment Entry Point ──────────────────────────────────

async def assess_student_risk(
    db: AsyncSession,
    student_id: str,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    create_alert_on_high: bool = True,
) -> RiskAssessment:
    """Run a full risk assessment for one student.

    1. Compute factors
    2. Calculate composite score
    3. Persist a RiskScore row
    4. Create an Alert if high-risk

    Returns:
        A RiskAssessment dataclass with all details.
    """

    factors = await compute_risk_factors(
        db, student_id, lookback_days=lookback_days
    )
    composite = calculate_composite_score(factors)
    level = _risk_level_from_score(composite)

    assessment = RiskAssessment(
        student_id=student_id,
        composite_score=composite,
        risk_level=level,
        factors=factors,
    )

    # ── Persist RiskScore ────────────────────────────────────────
    risk_row = RiskScore(
        id=str(uuid.uuid4()),
        student_id=student_id,
        risk_score=composite,
        risk_level=level,
        contributing_factors=factors.to_dict(),
    )
    db.add(risk_row)

    # ── Auto-alert on HIGH risk ──────────────────────────────────
    if assessment.is_high_risk and create_alert_on_high:
        alert = Alert(
            id=str(uuid.uuid4()),
            student_id=student_id,
            risk_score=composite,
            alert_type="high_risk",
            message=(
                f"Student {student_id} scored {composite}/100 "
                f"(level: {level}). Immediate attention recommended."
            ),
        )
        db.add(alert)
        logger.warning(
            f"HIGH RISK ALERT — student={student_id}, "
            f"score={composite}, level={level}"
        )

    await db.commit()
    await db.refresh(risk_row)

    logger.info(
        f"Risk assessment complete: student={student_id}, "
        f"score={composite}, level={level}"
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
    """Return the single most-recent risk score for a student."""
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
    """Run risk assessments for multiple students in sequence."""
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
