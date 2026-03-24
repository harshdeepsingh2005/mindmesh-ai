"""MindMesh AI — Behavioral Trend Analysis Module (Unsupervised).

Analyzes temporal patterns in student behavioral data using
statistical methods:

  • Linear regression slope for trend direction
  • Z-score anomaly detection on time-series
  • Change point detection via rolling statistics
  • Disengagement detection through activity frequency

All methods are unsupervised — they detect statistical deviations
from the student's own baseline, NOT from labelled data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
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
class AnomalyPoint:
    """A detected anomaly in the time series."""
    timestamp: datetime
    value: float
    z_score: float
    is_anomaly: bool


@dataclass
class TrendResult:
    """Result of behavioral trend analysis."""
    student_id: str
    period_days: int
    total_records: int
    data_points: int

    # Emotion trend
    emotion_trend: str        # improving, declining, stable, insufficient_data
    emotion_slope: float      # positive = improving, negative = declining
    avg_emotion_score: Optional[float]
    emotion_volatility: float  # standard deviation of scores

    # Activity trends
    checkin_count: int
    journal_count: int
    checkin_frequency: float  # avg checkins per week

    # Weekly aggregates
    weekly_averages: List[Dict]

    # Time-series anomalies
    anomaly_points: List[AnomalyPoint]
    anomaly_count: int

    # Change point detection
    change_point_detected: bool
    change_point_index: Optional[int]

    # Flags
    declining_flag: bool         # True if clear downward trend
    disengagement_flag: bool     # True if activity dropped
    volatility_flag: bool        # True if high emotional instability


async def fetch_student_records(
    db: AsyncSession,
    student_id: str,
    days: int = 30,
) -> List[TrendDataPoint]:
    """Fetch behavioral records for trend analysis."""
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
    """Calculate linear regression slope (least squares)."""
    n = len(values)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def _z_score_anomalies(
    values: List[float],
    threshold: float = 2.0,
) -> List[Tuple[int, float, float, bool]]:
    """Detect anomalies in a sequence using z-scores.

    A point is anomalous if |z-score| > threshold.

    Args:
        values: Time-series values.
        threshold: Z-score threshold for anomaly.

    Returns:
        List of (index, value, z_score, is_anomaly) tuples.
    """
    if len(values) < 3:
        return [(i, v, 0.0, False) for i, v in enumerate(values)]

    arr = np.array(values, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return [(i, v, 0.0, False) for i, v in enumerate(values)]

    results = []
    for i, v in enumerate(values):
        z = (v - mean) / std
        results.append((i, v, float(z), abs(z) > threshold))

    return results


def _detect_change_point(
    values: List[float],
    window: int = 5,
) -> Optional[int]:
    """Detect change points using CUSUM-inspired rolling mean shift.

    Compares rolling mean of first half vs second half.  If the
    difference exceeds a threshold, a change point is detected.

    Args:
        values: Time-series values.
        window: Minimum window size for comparison.

    Returns:
        Index of detected change point, or None.
    """
    if len(values) < window * 2:
        return None

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    max_diff = 0.0
    best_point = None
    overall_std = arr.std()

    if overall_std == 0:
        return None

    for i in range(window, n - window):
        left_mean = arr[:i].mean()
        right_mean = arr[i:].mean()
        diff = abs(right_mean - left_mean) / overall_std

        if diff > max_diff and diff > 1.5:  # threshold
            max_diff = diff
            best_point = i

    return best_point


def _compute_weekly_averages(
    data_points: List[TrendDataPoint], days: int
) -> List[Dict]:
    """Compute weekly average emotion scores."""
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
        scores_arr = np.array(scores)
        result.append({
            "weeks_ago": week_num,
            "avg_emotion_score": round(float(scores_arr.mean()), 4),
            "record_count": len(scores),
            "min_score": round(float(scores_arr.min()), 4),
            "max_score": round(float(scores_arr.max()), 4),
            "std_score": round(float(scores_arr.std()), 4),
        })

    return result


async def analyze_trends(
    db: AsyncSession,
    student_id: str,
    days: int = 30,
) -> TrendResult:
    """Analyze behavioral trends with unsupervised anomaly detection.

    Examines emotion scores, activity frequency, z-score anomalies,
    and change points to determine if the student's wellbeing is
    changing.
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
    anomaly_points: List[AnomalyPoint] = []
    change_point = None
    volatility_flag = False

    if len(emotion_scores) >= 3:
        emotion_slope = _calculate_slope(emotion_scores)
        avg_emotion = float(np.mean(emotion_scores))
        volatility = float(np.std(emotion_scores))

        # Trend direction
        if emotion_slope > 0.02:
            trend = "improving"
        elif emotion_slope < -0.02:
            trend = "declining"
        else:
            trend = "stable"

        # Z-score anomaly detection
        z_results = _z_score_anomalies(emotion_scores, threshold=2.0)
        for i, val, z, is_anom in z_results:
            if is_anom and i < len(data_points):
                anomaly_points.append(AnomalyPoint(
                    timestamp=data_points[i].timestamp,
                    value=val,
                    z_score=round(z, 4),
                    is_anomaly=True,
                ))

        # Change point detection
        change_point = _detect_change_point(emotion_scores)

        # Volatility flag
        volatility_flag = volatility > 0.3

    elif len(emotion_scores) > 0:
        emotion_slope = 0.0
        avg_emotion = float(np.mean(emotion_scores))
        volatility = float(np.std(emotion_scores)) if len(emotion_scores) > 1 else 0.0
        trend = "insufficient_data"
    else:
        emotion_slope = 0.0
        avg_emotion = None
        volatility = 0.0
        trend = "insufficient_data"

    # Activity frequency
    weeks_in_period = max(days / 7, 1)
    checkin_frequency = round(checkin_count / weeks_in_period, 2)

    # Weekly averages
    weekly_averages = _compute_weekly_averages(data_points, days)

    # Disengagement detection
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
        anomaly_points=anomaly_points,
        anomaly_count=len(anomaly_points),
        change_point_detected=change_point is not None,
        change_point_index=change_point,
        declining_flag=declining_flag,
        disengagement_flag=disengagement_flag,
        volatility_flag=volatility_flag,
    )

    logger.info(
        f"Trend analysis: student_id={student_id}, trend={trend}, "
        f"slope={emotion_slope:.6f}, records={total_records}, "
        f"anomalies={len(anomaly_points)}, "
        f"change_point={'yes' if change_point else 'no'}"
    )

    return result
