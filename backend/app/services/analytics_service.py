"""MindMesh AI — Analytics Service.

Provides aggregated analytics queries for the counselor/admin dashboard:
  • School-wide and per-student emotional trend summaries
  • Risk distribution statistics (heatmap data)
  • Alert summaries and response-time metrics
  • Engagement metrics and activity breakdowns
  • Time-series data for charting (emotion trends, risk trajectories)

All queries are read-only and respect role-based access control
enforced at the route layer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.student import Student
from ..models.behavioral_record import BehavioralRecord
from ..models.emotion_analysis import EmotionAnalysis
from ..models.risk_score import RiskScore
from ..models.alert import Alert
from ..logging_config import logger


# ─── Data Classes ────────────────────────────────────────────────


@dataclass
class OverviewStats:
    """Top-level dashboard overview numbers."""
    total_students: int = 0
    total_records: int = 0
    total_checkins: int = 0
    total_journals: int = 0
    total_risk_assessments: int = 0
    open_alerts: int = 0
    acknowledged_alerts: int = 0
    high_risk_students: int = 0
    medium_risk_students: int = 0
    low_risk_students: int = 0
    unassessed_students: int = 0
    avg_risk_score: Optional[float] = None
    avg_emotion_score: Optional[float] = None
    avg_sentiment_score: Optional[float] = None
    period_days: int = 30


@dataclass
class EmotionDistribution:
    """Distribution of detected emotions."""
    emotion: str
    count: int
    percentage: float


@dataclass
class RiskBucket:
    """Risk distribution bucket for heatmap / histogram."""
    risk_level: str
    count: int
    percentage: float
    student_ids: List[str] = field(default_factory=list)


@dataclass
class TimeSeriesPoint:
    """Single point in a time-series chart."""
    date: str  # ISO date string (YYYY-MM-DD)
    value: float
    count: int = 0


@dataclass
class StudentSummary:
    """Compact per-student summary for dashboard table."""
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


# ─── Overview Stats ──────────────────────────────────────────────


async def get_overview_stats(
    db: AsyncSession,
    days: int = 30,
) -> OverviewStats:
    """Compute top-level dashboard overview statistics.

    Args:
        db: Async database session.
        days: Look-back period for activity counts.

    Returns:
        OverviewStats with all summary numbers.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stats = OverviewStats(period_days=days)

    # Total students
    result = await db.execute(select(func.count(Student.id)))
    stats.total_students = result.scalar() or 0

    # Records in period
    result = await db.execute(
        select(func.count(BehavioralRecord.id)).where(
            BehavioralRecord.timestamp >= since
        )
    )
    stats.total_records = result.scalar() or 0

    # Checkins in period
    result = await db.execute(
        select(func.count(BehavioralRecord.id)).where(
            and_(
                BehavioralRecord.timestamp >= since,
                BehavioralRecord.activity_type == "checkin",
            )
        )
    )
    stats.total_checkins = result.scalar() or 0

    # Journals in period
    result = await db.execute(
        select(func.count(BehavioralRecord.id)).where(
            and_(
                BehavioralRecord.timestamp >= since,
                BehavioralRecord.activity_type == "journal",
            )
        )
    )
    stats.total_journals = result.scalar() or 0

    # Risk assessments in period
    result = await db.execute(
        select(func.count(RiskScore.id)).where(
            RiskScore.calculated_at >= since
        )
    )
    stats.total_risk_assessments = result.scalar() or 0

    # Open alerts
    result = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == "open")
    )
    stats.open_alerts = result.scalar() or 0

    # Acknowledged alerts
    result = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == "acknowledged")
    )
    stats.acknowledged_alerts = result.scalar() or 0

    # Risk distribution — latest risk score per student
    latest_risk_subq = (
        select(
            RiskScore.student_id,
            func.max(RiskScore.calculated_at).label("max_date"),
        )
        .group_by(RiskScore.student_id)
        .subquery()
    )

    latest_risks = await db.execute(
        select(RiskScore.risk_level, func.count(RiskScore.id)).join(
            latest_risk_subq,
            and_(
                RiskScore.student_id == latest_risk_subq.c.student_id,
                RiskScore.calculated_at == latest_risk_subq.c.max_date,
            ),
        ).group_by(RiskScore.risk_level)
    )

    assessed_students = 0
    for level, cnt in latest_risks.all():
        assessed_students += cnt
        if level == "high":
            stats.high_risk_students = cnt
        elif level == "medium":
            stats.medium_risk_students = cnt
        elif level == "low":
            stats.low_risk_students = cnt

    stats.unassessed_students = stats.total_students - assessed_students

    # Average risk score (latest per student)
    result = await db.execute(
        select(func.avg(RiskScore.risk_score)).join(
            latest_risk_subq,
            and_(
                RiskScore.student_id == latest_risk_subq.c.student_id,
                RiskScore.calculated_at == latest_risk_subq.c.max_date,
            ),
        )
    )
    avg_risk = result.scalar()
    stats.avg_risk_score = round(float(avg_risk), 2) if avg_risk is not None else None

    # Average emotion score in period
    result = await db.execute(
        select(func.avg(BehavioralRecord.emotion_score)).where(
            and_(
                BehavioralRecord.timestamp >= since,
                BehavioralRecord.emotion_score.isnot(None),
            )
        )
    )
    avg_emo = result.scalar()
    stats.avg_emotion_score = round(float(avg_emo), 3) if avg_emo is not None else None

    # Average sentiment score in period
    result = await db.execute(
        select(func.avg(BehavioralRecord.sentiment_score)).where(
            and_(
                BehavioralRecord.timestamp >= since,
                BehavioralRecord.sentiment_score.isnot(None),
            )
        )
    )
    avg_sent = result.scalar()
    stats.avg_sentiment_score = (
        round(float(avg_sent), 3) if avg_sent is not None else None
    )

    logger.info(
        f"Overview stats computed: {stats.total_students} students, "
        f"{stats.total_records} records, {stats.open_alerts} open alerts, "
        f"period={days}d"
    )

    return stats


# ─── Emotion Distribution ───────────────────────────────────────


async def get_emotion_distribution(
    db: AsyncSession,
    days: int = 30,
    student_id: Optional[str] = None,
) -> List[EmotionDistribution]:
    """Get the distribution of detected emotions.

    Args:
        db: Async database session.
        days: Look-back period.
        student_id: Optional filter for a specific student.

    Returns:
        List of EmotionDistribution sorted by count descending.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    conditions = [BehavioralRecord.timestamp >= since]
    if student_id:
        conditions.append(BehavioralRecord.student_id == student_id)

    result = await db.execute(
        select(
            EmotionAnalysis.predicted_emotion,
            func.count(EmotionAnalysis.id).label("cnt"),
        )
        .join(
            BehavioralRecord,
            EmotionAnalysis.record_id == BehavioralRecord.id,
        )
        .where(and_(*conditions))
        .group_by(EmotionAnalysis.predicted_emotion)
        .order_by(func.count(EmotionAnalysis.id).desc())
    )

    rows = result.all()
    total = sum(r.cnt for r in rows) if rows else 0

    return [
        EmotionDistribution(
            emotion=r.predicted_emotion,
            count=r.cnt,
            percentage=round((r.cnt / total) * 100, 1) if total > 0 else 0.0,
        )
        for r in rows
    ]


# ─── Risk Distribution (Heatmap Data) ───────────────────────────


async def get_risk_distribution(
    db: AsyncSession,
) -> List[RiskBucket]:
    """Get the risk-level distribution across all students (latest score).

    Used for risk heatmap / pie chart on the dashboard.

    Returns:
        List of RiskBucket with counts and student IDs.
    """
    # Subquery: latest risk score per student
    latest_risk_subq = (
        select(
            RiskScore.student_id,
            func.max(RiskScore.calculated_at).label("max_date"),
        )
        .group_by(RiskScore.student_id)
        .subquery()
    )

    result = await db.execute(
        select(RiskScore.student_id, RiskScore.risk_level, RiskScore.risk_score)
        .join(
            latest_risk_subq,
            and_(
                RiskScore.student_id == latest_risk_subq.c.student_id,
                RiskScore.calculated_at == latest_risk_subq.c.max_date,
            ),
        )
    )

    rows = result.all()
    total = len(rows)

    buckets: Dict[str, List[str]] = {"low": [], "medium": [], "high": []}
    for row in rows:
        level = row.risk_level
        if level in buckets:
            buckets[level].append(row.student_id)

    return [
        RiskBucket(
            risk_level=level,
            count=len(ids),
            percentage=round((len(ids) / total) * 100, 1) if total > 0 else 0.0,
            student_ids=ids,
        )
        for level, ids in buckets.items()
    ]


async def get_risk_score_histogram(
    db: AsyncSession,
    bucket_size: int = 10,
) -> List[Dict[str, Any]]:
    """Get risk score histogram (buckets of N points).

    Returns data like: [{"range": "0-9", "count": 5}, ...]
    """
    latest_risk_subq = (
        select(
            RiskScore.student_id,
            func.max(RiskScore.calculated_at).label("max_date"),
        )
        .group_by(RiskScore.student_id)
        .subquery()
    )

    result = await db.execute(
        select(RiskScore.risk_score).join(
            latest_risk_subq,
            and_(
                RiskScore.student_id == latest_risk_subq.c.student_id,
                RiskScore.calculated_at == latest_risk_subq.c.max_date,
            ),
        )
    )

    scores = [row[0] for row in result.all()]

    # Build histogram buckets
    num_buckets = math.ceil(101 / bucket_size)
    histogram = []
    for i in range(num_buckets):
        low = i * bucket_size
        high = min(low + bucket_size - 1, 100)
        count = sum(1 for s in scores if low <= s <= high)
        histogram.append({
            "range": f"{low}-{high}",
            "low": low,
            "high": high,
            "count": count,
        })

    return histogram


# ─── Time-Series: Emotion Trends ────────────────────────────────


async def get_emotion_trend_timeseries(
    db: AsyncSession,
    days: int = 30,
    student_id: Optional[str] = None,
) -> List[TimeSeriesPoint]:
    """Get daily average emotion scores over time.

    Args:
        db: Async database session.
        days: Look-back period.
        student_id: Optional filter.

    Returns:
        List of TimeSeriesPoint with daily averages.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    conditions = [
        BehavioralRecord.timestamp >= since,
        BehavioralRecord.emotion_score.isnot(None),
    ]
    if student_id:
        conditions.append(BehavioralRecord.student_id == student_id)

    result = await db.execute(
        select(
            func.date(BehavioralRecord.timestamp).label("day"),
            func.avg(BehavioralRecord.emotion_score).label("avg_score"),
            func.count(BehavioralRecord.id).label("cnt"),
        )
        .where(and_(*conditions))
        .group_by(func.date(BehavioralRecord.timestamp))
        .order_by(func.date(BehavioralRecord.timestamp))
    )

    return [
        TimeSeriesPoint(
            date=str(row.day),
            value=round(float(row.avg_score), 3),
            count=row.cnt,
        )
        for row in result.all()
    ]


async def get_sentiment_trend_timeseries(
    db: AsyncSession,
    days: int = 30,
    student_id: Optional[str] = None,
) -> List[TimeSeriesPoint]:
    """Get daily average sentiment scores over time."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    conditions = [
        BehavioralRecord.timestamp >= since,
        BehavioralRecord.sentiment_score.isnot(None),
    ]
    if student_id:
        conditions.append(BehavioralRecord.student_id == student_id)

    result = await db.execute(
        select(
            func.date(BehavioralRecord.timestamp).label("day"),
            func.avg(BehavioralRecord.sentiment_score).label("avg_score"),
            func.count(BehavioralRecord.id).label("cnt"),
        )
        .where(and_(*conditions))
        .group_by(func.date(BehavioralRecord.timestamp))
        .order_by(func.date(BehavioralRecord.timestamp))
    )

    return [
        TimeSeriesPoint(
            date=str(row.day),
            value=round(float(row.avg_score), 3),
            count=row.cnt,
        )
        for row in result.all()
    ]


async def get_risk_trend_timeseries(
    db: AsyncSession,
    days: int = 30,
    student_id: Optional[str] = None,
) -> List[TimeSeriesPoint]:
    """Get daily average risk scores over time."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    conditions = [RiskScore.calculated_at >= since]
    if student_id:
        conditions.append(RiskScore.student_id == student_id)

    result = await db.execute(
        select(
            func.date(RiskScore.calculated_at).label("day"),
            func.avg(RiskScore.risk_score).label("avg_score"),
            func.count(RiskScore.id).label("cnt"),
        )
        .where(and_(*conditions))
        .group_by(func.date(RiskScore.calculated_at))
        .order_by(func.date(RiskScore.calculated_at))
    )

    return [
        TimeSeriesPoint(
            date=str(row.day),
            value=round(float(row.avg_score), 1),
            count=row.cnt,
        )
        for row in result.all()
    ]


# ─── Activity Breakdown ─────────────────────────────────────────


async def get_activity_breakdown(
    db: AsyncSession,
    days: int = 30,
) -> List[Dict[str, Any]]:
    """Get activity type breakdown with daily counts.

    Returns data for a stacked bar chart.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date(BehavioralRecord.timestamp).label("day"),
            BehavioralRecord.activity_type,
            func.count(BehavioralRecord.id).label("cnt"),
        )
        .where(BehavioralRecord.timestamp >= since)
        .group_by(
            func.date(BehavioralRecord.timestamp),
            BehavioralRecord.activity_type,
        )
        .order_by(func.date(BehavioralRecord.timestamp))
    )

    # Reshape into [{date, checkin, journal, survey}, ...]
    day_data: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"checkin": 0, "journal": 0, "survey": 0}
    )
    for row in result.all():
        day_data[str(row.day)][row.activity_type] = row.cnt

    return [
        {"date": day, **counts}
        for day, counts in sorted(day_data.items())
    ]


# ─── Alert Summary ──────────────────────────────────────────────


async def get_alert_summary(
    db: AsyncSession,
    days: int = 30,
) -> Dict[str, Any]:
    """Get alert statistics for the dashboard.

    Returns counts by status, by type, and a daily creation trend.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # By status
    result = await db.execute(
        select(
            Alert.status,
            func.count(Alert.id).label("cnt"),
        )
        .where(Alert.created_at >= since)
        .group_by(Alert.status)
    )
    by_status = {row.status: row.cnt for row in result.all()}

    # By type
    result = await db.execute(
        select(
            Alert.alert_type,
            func.count(Alert.id).label("cnt"),
        )
        .where(Alert.created_at >= since)
        .group_by(Alert.alert_type)
    )
    by_type = {row.alert_type: row.cnt for row in result.all()}

    # Daily creation trend
    result = await db.execute(
        select(
            func.date(Alert.created_at).label("day"),
            func.count(Alert.id).label("cnt"),
        )
        .where(Alert.created_at >= since)
        .group_by(func.date(Alert.created_at))
        .order_by(func.date(Alert.created_at))
    )
    daily_trend = [
        {"date": str(row.day), "count": row.cnt}
        for row in result.all()
    ]

    total = sum(by_status.values())

    return {
        "period_days": days,
        "total_alerts": total,
        "by_status": by_status,
        "by_type": by_type,
        "daily_trend": daily_trend,
    }


# ─── Student Summaries (Dashboard Table) ────────────────────────


async def get_student_summaries(
    db: AsyncSession,
    days: int = 30,
    risk_level_filter: Optional[str] = None,
    school_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[List[StudentSummary], int]:
    """Get compact summaries for all students for the dashboard table.

    Includes latest risk score, average emotion, open alert count,
    and last activity date.

    Args:
        db: Async database session.
        days: Look-back period for averages.
        risk_level_filter: Optional filter by latest risk level.
        school_filter: Optional filter by school.
        skip: Pagination offset.
        limit: Pagination limit.

    Returns:
        Tuple of (list of StudentSummary, total count).
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Base student query with filters
    student_conditions = []
    if school_filter:
        student_conditions.append(Student.school == school_filter)

    where_clause = and_(*student_conditions) if student_conditions else True

    # Total count
    count_result = await db.execute(
        select(func.count(Student.id)).where(where_clause)
    )
    total = count_result.scalar() or 0

    # Fetch students
    result = await db.execute(
        select(Student)
        .where(where_clause)
        .order_by(Student.student_identifier)
        .offset(skip)
        .limit(limit)
    )
    students = list(result.scalars().all())

    summaries = []
    for student in students:
        summary = StudentSummary(
            student_id=student.id,
            student_identifier=student.student_identifier,
            school=student.school,
            grade=student.grade,
        )

        # Latest risk score
        risk_result = await db.execute(
            select(RiskScore)
            .where(RiskScore.student_id == student.id)
            .order_by(RiskScore.calculated_at.desc())
            .limit(1)
        )
        latest_risk = risk_result.scalar_one_or_none()
        if latest_risk:
            summary.latest_risk_score = latest_risk.risk_score
            summary.latest_risk_level = latest_risk.risk_level

        # Filter by risk level if requested
        if risk_level_filter:
            if summary.latest_risk_level != risk_level_filter:
                continue

        # Avg emotion score in period
        emo_result = await db.execute(
            select(func.avg(BehavioralRecord.emotion_score)).where(
                and_(
                    BehavioralRecord.student_id == student.id,
                    BehavioralRecord.timestamp >= since,
                    BehavioralRecord.emotion_score.isnot(None),
                )
            )
        )
        avg_emo = emo_result.scalar()
        summary.avg_emotion_score = (
            round(float(avg_emo), 3) if avg_emo is not None else None
        )

        # Avg sentiment score in period
        sent_result = await db.execute(
            select(func.avg(BehavioralRecord.sentiment_score)).where(
                and_(
                    BehavioralRecord.student_id == student.id,
                    BehavioralRecord.timestamp >= since,
                    BehavioralRecord.sentiment_score.isnot(None),
                )
            )
        )
        avg_sent = sent_result.scalar()
        summary.avg_sentiment_score = (
            round(float(avg_sent), 3) if avg_sent is not None else None
        )

        # Record count in period
        rec_result = await db.execute(
            select(func.count(BehavioralRecord.id)).where(
                and_(
                    BehavioralRecord.student_id == student.id,
                    BehavioralRecord.timestamp >= since,
                )
            )
        )
        summary.record_count = rec_result.scalar() or 0

        # Open alert count
        alert_result = await db.execute(
            select(func.count(Alert.id)).where(
                and_(
                    Alert.student_id == student.id,
                    Alert.status.in_(["open", "acknowledged"]),
                )
            )
        )
        summary.open_alert_count = alert_result.scalar() or 0

        # Last activity
        last_result = await db.execute(
            select(func.max(BehavioralRecord.timestamp)).where(
                BehavioralRecord.student_id == student.id
            )
        )
        last_ts = last_result.scalar()
        summary.last_activity = last_ts.isoformat() if last_ts else None

        summaries.append(summary)

    return summaries, total


# ─── School-Level Aggregation ────────────────────────────────────


async def get_school_stats(
    db: AsyncSession,
    days: int = 30,
) -> List[Dict[str, Any]]:
    """Get per-school aggregated statistics.

    Returns:
        List of dicts with school name, student count, avg risk, etc.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Students per school
    result = await db.execute(
        select(
            Student.school,
            func.count(Student.id).label("student_count"),
        )
        .group_by(Student.school)
        .order_by(Student.school)
    )
    schools = {
        row.school: {"school": row.school, "student_count": row.student_count}
        for row in result.all()
    }

    # Avg emotion per school
    result = await db.execute(
        select(
            Student.school,
            func.avg(BehavioralRecord.emotion_score).label("avg_emotion"),
            func.count(BehavioralRecord.id).label("record_count"),
        )
        .join(Student, BehavioralRecord.student_id == Student.id)
        .where(
            and_(
                BehavioralRecord.timestamp >= since,
                BehavioralRecord.emotion_score.isnot(None),
            )
        )
        .group_by(Student.school)
    )
    for row in result.all():
        if row.school in schools:
            schools[row.school]["avg_emotion_score"] = (
                round(float(row.avg_emotion), 3) if row.avg_emotion else None
            )
            schools[row.school]["record_count"] = row.record_count

    # Open alerts per school
    result = await db.execute(
        select(
            Student.school,
            func.count(Alert.id).label("alert_count"),
        )
        .join(Student, Alert.student_id == Student.id)
        .where(Alert.status.in_(["open", "acknowledged"]))
        .group_by(Student.school)
    )
    for row in result.all():
        if row.school in schools:
            schools[row.school]["open_alert_count"] = row.alert_count

    # Fill defaults
    for school_data in schools.values():
        school_data.setdefault("avg_emotion_score", None)
        school_data.setdefault("record_count", 0)
        school_data.setdefault("open_alert_count", 0)

    return list(schools.values())
