"""MindMesh AI — Behavioral Trend Analysis Module.

Analyzes temporal patterns in student behavioral data to detect
deterioration trends, stability, or improvement. Uses statistical
methods on time-series behavioral records.

Outputs trend indicators that feed into the risk scoring engine.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.behavioral_record import BehavioralRecord
from ..logging_config import logger


@dataclass
class TrendDataPoint:
    """A single data point in a trend series."""
    timestamp: datetime
    emotion_score: Optional[float]
    sentiment_score: Optional[float]
    activity_type: str


@dataclass
class TrendResult:
    """Result of behavioral trend analysis."""
    student_id: str
    period_days: int
    total_records: int
    data_points: int

    # Emotion trend
    emotion_trend: str  # improving, declining, stable, insufficient_data
    emotion_slope: float  # positive = improving, negative = declining
    avg_emotion_score: Optional[float]
    emotion_volatility: float  # standard deviation of scores

    # Activity trends
    checkin_count: int
    journal_count: int
    checkin_frequency: float  # avg checkins per week

    # Weekly aggregates
    weekly_averages: List[Dict]

    # Flags
    declining_flag: bool  # True if clear downward trend detected
    disengagement_flag: bool  # True if activity frequency dropped significantly


async def fetch_student_records(
    db: AsyncSession,
    student_id: str,
    days: int = 30,
) -> List[TrendDataPoint]:
    """Fetch behavioral records for trend analysis.

    Args:
        db: Async database session.
        student_id: The student's ID.
        days: Number of days to look back.

    Returns:
        List of TrendDataPoint objects sorted by timestamp.
    """
    since = datetime.utcnow() - timedelta(days=days)

    result = await db.execute(
        select(BehavioralRecord)
        .where(
            and_(
                BehavioralRecord.student_id == student_id,
                BehavioralRecord.timestamp >= since,
            )
        )
        .order_by(BehavioralRecord.timestamp.asc())
    )
    records = result.scalars().all()

    return [
        TrendDataPoint(
            timestamp=r.timestamp,
            emotion_score=r.emotion_score,
            sentiment_score=r.sentiment_score,
            activity_type=r.activity_type,
        )
        for r in records
    ]


def _calculate_slope(values: List[float]) -> float:
    """Calculate linear regression slope for a series of values.

    Uses simple least-squares linear regression.

    Args:
        values: Ordered list of float values.

    Returns:
        Slope coefficient. Positive = upward trend, negative = downward.
    """
    n = len(values)
    if n < 2:
        return 0.0

    x_vals = list(range(n))
    x_mean = sum(x_vals) / n
    y_mean = sum(values) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return variance ** 0.5


def _compute_weekly_averages(
    data_points: List[TrendDataPoint], days: int
) -> List[Dict]:
    """Compute weekly average emotion scores.

    Args:
        data_points: Sorted list of data points.
        days: Total period in days.

    Returns:
        List of weekly aggregate dicts.
    """
    if not data_points:
        return []

    weeks: Dict[int, List[float]] = {}
    now = datetime.utcnow()

    for dp in data_points:
        if dp.emotion_score is not None:
            days_ago = (now - dp.timestamp).days
            week_num = days_ago // 7
            if week_num not in weeks:
                weeks[week_num] = []
            weeks[week_num].append(dp.emotion_score)

    result = []
    for week_num in sorted(weeks.keys()):
        scores = weeks[week_num]
        result.append({
            "weeks_ago": week_num,
            "avg_emotion_score": round(sum(scores) / len(scores), 4),
            "record_count": len(scores),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
        })

    return result


async def analyze_trends(
    db: AsyncSession,
    student_id: str,
    days: int = 30,
) -> TrendResult:
    """Analyze behavioral trends for a student over a time period.

    Examines emotion scores, activity frequency, and weekly patterns
    to determine if the student's wellbeing is improving, declining,
    or stable.

    Args:
        db: Async database session.
        student_id: The student's ID.
        days: Number of days to analyze (default 30).

    Returns:
        TrendResult with comprehensive trend analysis.
    """
    data_points = await fetch_student_records(db, student_id, days)

    total_records = len(data_points)
    checkin_count = sum(1 for dp in data_points if dp.activity_type == "checkin")
    journal_count = sum(1 for dp in data_points if dp.activity_type == "journal")

    # Extract emotion scores
    emotion_scores = [
        dp.emotion_score for dp in data_points if dp.emotion_score is not None
    ]

    # Compute metrics
    if len(emotion_scores) >= 3:
        emotion_slope = _calculate_slope(emotion_scores)
        avg_emotion = sum(emotion_scores) / len(emotion_scores)
        volatility = _calculate_std(emotion_scores)

        # Determine trend direction
        if emotion_slope > 0.02:
            trend = "improving"
        elif emotion_slope < -0.02:
            trend = "declining"
        else:
            trend = "stable"
    elif len(emotion_scores) > 0:
        emotion_slope = 0.0
        avg_emotion = sum(emotion_scores) / len(emotion_scores)
        volatility = _calculate_std(emotion_scores) if len(emotion_scores) > 1 else 0.0
        trend = "insufficient_data"
    else:
        emotion_slope = 0.0
        avg_emotion = None
        volatility = 0.0
        trend = "insufficient_data"

    # Activity frequency (checkins per week)
    weeks_in_period = max(days / 7, 1)
    checkin_frequency = round(checkin_count / weeks_in_period, 2)

    # Weekly averages
    weekly_averages = _compute_weekly_averages(data_points, days)

    # Detect disengagement (activity dropped in recent week vs overall average)
    disengagement_flag = False
    if len(weekly_averages) >= 2:
        recent_week = weekly_averages[-1]
        older_weeks = weekly_averages[:-1]
        avg_older_count = sum(w["record_count"] for w in older_weeks) / len(older_weeks)
        if avg_older_count > 0 and recent_week["record_count"] < avg_older_count * 0.4:
            disengagement_flag = True

    # Declining flag
    declining_flag = trend == "declining" and emotion_slope < -0.03

    result = TrendResult(
        student_id=student_id,
        period_days=days,
        total_records=total_records,
        data_points=len(emotion_scores),
        emotion_trend=trend,
        emotion_slope=round(emotion_slope, 6),
        avg_emotion_score=round(avg_emotion, 4) if avg_emotion is not None else None,
        emotion_volatility=round(volatility, 4),
        checkin_count=checkin_count,
        journal_count=journal_count,
        checkin_frequency=checkin_frequency,
        weekly_averages=weekly_averages,
        declining_flag=declining_flag,
        disengagement_flag=disengagement_flag,
    )

    logger.info(
        f"Trend analysis: student_id={student_id}, trend={trend}, "
        f"slope={emotion_slope:.6f}, records={total_records}"
    )

    return result
