"""MindMesh AI — Analytics & Dashboard API Routes.

Provides read-only endpoints for the counselor / admin dashboard:
  • Overview stats & compound dashboard endpoint
  • Emotion distribution & trend time-series
  • Risk distribution, histogram & trend time-series
  • Sentiment trend time-series
  • Activity breakdown (stacked bar chart)
  • Alert summary & daily trend
  • Student summary table (paginated)
  • School-level aggregated stats
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..database.analytics_schemas import (
    OverviewStatsResponse,
    EmotionDistributionItem,
    EmotionDistributionResponse,
    RiskBucketResponse,
    RiskDistributionResponse,
    RiskHistogramBucket,
    RiskHistogramResponse,
    TimeSeriesPointResponse,
    TimeSeriesResponse,
    ActivityDayResponse,
    ActivityBreakdownResponse,
    AlertDailyTrend,
    AlertSummaryResponse,
    StudentSummaryResponse,
    StudentSummaryListResponse,
    SchoolStatsItem,
    SchoolStatsResponse,
    DashboardResponse,
)
from ..dependencies import get_current_user, require_roles
from ..services import analytics_service
from ..logging_config import logger

router = APIRouter()


# ─── Compound Dashboard Endpoint ────────────────────────────────


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get full dashboard payload",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_dashboard(
    days: int = Query(30, ge=1, le=365, description="Look-back period in days"),
    db: AsyncSession = Depends(get_db),
):
    """All-in-one dashboard endpoint — reduces frontend API calls.

    Combines: overview, risk distribution, emotion distribution,
    emotion trend, alert summary, and top at-risk students.
    """
    overview = await analytics_service.get_overview_stats(db, days=days)
    risk_dist = await analytics_service.get_risk_distribution(db)
    emotion_dist = await analytics_service.get_emotion_distribution(db, days=days)
    emotion_trend = await analytics_service.get_emotion_trend_timeseries(db, days=days)
    alert_summary = await analytics_service.get_alert_summary(db, days=days)
    at_risk, _ = await analytics_service.get_student_summaries(
        db, days=days, risk_level_filter="high", limit=10
    )

    overview_resp = OverviewStatsResponse(**overview.__dict__)

    risk_buckets = [RiskBucketResponse(**b.__dict__) for b in risk_dist]
    risk_total = sum(b.count for b in risk_dist)
    risk_resp = RiskDistributionResponse(buckets=risk_buckets, total_assessed=risk_total)

    emo_items = [EmotionDistributionItem(**e.__dict__) for e in emotion_dist]
    emo_resp = EmotionDistributionResponse(
        period_days=days,
        distribution=emo_items,
        total_analyses=sum(e.count for e in emotion_dist),
    )

    trend_points = [TimeSeriesPointResponse(**p.__dict__) for p in emotion_trend]
    trend_resp = TimeSeriesResponse(
        metric="emotion_score",
        period_days=days,
        data_points=trend_points,
        total_points=len(trend_points),
    )

    alert_resp = AlertSummaryResponse(
        period_days=days,
        total_alerts=alert_summary["total_alerts"],
        by_status=alert_summary["by_status"],
        by_type=alert_summary["by_type"],
        daily_trend=[AlertDailyTrend(**d) for d in alert_summary["daily_trend"]],
    )

    at_risk_resp = [StudentSummaryResponse(**s.__dict__) for s in at_risk]

    return DashboardResponse(
        overview=overview_resp,
        risk_distribution=risk_resp,
        emotion_distribution=emo_resp,
        emotion_trend=trend_resp,
        alert_summary=alert_resp,
        at_risk_students=at_risk_resp,
    )


# ─── Overview Stats ──────────────────────────────────────────────


@router.get(
    "/overview",
    response_model=OverviewStatsResponse,
    summary="Get dashboard overview statistics",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_overview(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Top-level numerical summary for the dashboard header cards."""
    stats = await analytics_service.get_overview_stats(db, days=days)
    return OverviewStatsResponse(**stats.__dict__)


# ─── Emotion Distribution ───────────────────────────────────────


@router.get(
    "/emotions/distribution",
    response_model=EmotionDistributionResponse,
    summary="Get emotion distribution",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_emotion_distribution(
    days: int = Query(30, ge=1, le=365),
    student_id: Optional[str] = Query(None, description="Filter by student"),
    db: AsyncSession = Depends(get_db),
):
    """Distribution of detected emotions (pie / donut chart)."""
    dist = await analytics_service.get_emotion_distribution(
        db, days=days, student_id=student_id
    )
    items = [EmotionDistributionItem(**d.__dict__) for d in dist]
    return EmotionDistributionResponse(
        period_days=days,
        student_id=student_id,
        distribution=items,
        total_analyses=sum(d.count for d in dist),
    )


# ─── Risk Distribution (Heatmap) ────────────────────────────────


@router.get(
    "/risk/distribution",
    response_model=RiskDistributionResponse,
    summary="Get risk distribution across students",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_risk_distribution(
    db: AsyncSession = Depends(get_db),
):
    """Risk-level distribution (heatmap / pie chart data)."""
    buckets = await analytics_service.get_risk_distribution(db)
    bucket_resp = [RiskBucketResponse(**b.__dict__) for b in buckets]
    total = sum(b.count for b in buckets)
    return RiskDistributionResponse(buckets=bucket_resp, total_assessed=total)


@router.get(
    "/risk/histogram",
    response_model=RiskHistogramResponse,
    summary="Get risk score histogram",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_risk_histogram(
    bucket_size: int = Query(10, ge=5, le=25),
    db: AsyncSession = Depends(get_db),
):
    """Risk score histogram for bar chart visualization."""
    buckets = await analytics_service.get_risk_score_histogram(db, bucket_size=bucket_size)
    bucket_resp = [RiskHistogramBucket(**b) for b in buckets]
    total = sum(b.count for b in bucket_resp)
    return RiskHistogramResponse(
        bucket_size=bucket_size,
        buckets=bucket_resp,
        total_assessed=total,
    )


# ─── Time-Series Trends ─────────────────────────────────────────


@router.get(
    "/trends/emotion",
    response_model=TimeSeriesResponse,
    summary="Get emotion score trend (time-series)",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_emotion_trend(
    days: int = Query(30, ge=1, le=365),
    student_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Daily average emotion scores for line chart."""
    points = await analytics_service.get_emotion_trend_timeseries(
        db, days=days, student_id=student_id
    )
    return TimeSeriesResponse(
        metric="emotion_score",
        period_days=days,
        student_id=student_id,
        data_points=[TimeSeriesPointResponse(**p.__dict__) for p in points],
        total_points=len(points),
    )


@router.get(
    "/trends/sentiment",
    response_model=TimeSeriesResponse,
    summary="Get sentiment score trend (time-series)",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_sentiment_trend(
    days: int = Query(30, ge=1, le=365),
    student_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Daily average sentiment scores for line chart."""
    points = await analytics_service.get_sentiment_trend_timeseries(
        db, days=days, student_id=student_id
    )
    return TimeSeriesResponse(
        metric="sentiment_score",
        period_days=days,
        student_id=student_id,
        data_points=[TimeSeriesPointResponse(**p.__dict__) for p in points],
        total_points=len(points),
    )


@router.get(
    "/trends/risk",
    response_model=TimeSeriesResponse,
    summary="Get risk score trend (time-series)",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_risk_trend(
    days: int = Query(30, ge=1, le=365),
    student_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Daily average risk scores for line chart."""
    points = await analytics_service.get_risk_trend_timeseries(
        db, days=days, student_id=student_id
    )
    return TimeSeriesResponse(
        metric="risk_score",
        period_days=days,
        student_id=student_id,
        data_points=[TimeSeriesPointResponse(**p.__dict__) for p in points],
        total_points=len(points),
    )


# ─── Activity Breakdown ─────────────────────────────────────────


@router.get(
    "/activity/breakdown",
    response_model=ActivityBreakdownResponse,
    summary="Get activity breakdown by type (stacked bar chart)",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_activity_breakdown(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Daily activity counts by type for stacked bar chart."""
    daily = await analytics_service.get_activity_breakdown(db, days=days)
    return ActivityBreakdownResponse(
        period_days=days,
        daily=[ActivityDayResponse(**d) for d in daily],
        total_days=len(daily),
    )


# ─── Alert Summary ──────────────────────────────────────────────


@router.get(
    "/alerts/summary",
    response_model=AlertSummaryResponse,
    summary="Get alert statistics and daily trend",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_alert_summary(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Alert statistics with counts by status, type, and daily trend."""
    summary = await analytics_service.get_alert_summary(db, days=days)
    return AlertSummaryResponse(
        period_days=days,
        total_alerts=summary["total_alerts"],
        by_status=summary["by_status"],
        by_type=summary["by_type"],
        daily_trend=[AlertDailyTrend(**d) for d in summary["daily_trend"]],
    )


# ─── Student Summaries (Dashboard Table) ────────────────────────


@router.get(
    "/students",
    response_model=StudentSummaryListResponse,
    summary="Get paginated student summary table",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_student_summaries(
    days: int = Query(30, ge=1, le=365),
    risk_level: Optional[str] = Query(None, description="Filter: low, medium, high"),
    school: Optional[str] = Query(None, description="Filter by school name"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Paginated student summaries with risk, emotion, and alert data."""
    students, total = await analytics_service.get_student_summaries(
        db,
        days=days,
        risk_level_filter=risk_level,
        school_filter=school,
        skip=skip,
        limit=limit,
    )
    return StudentSummaryListResponse(
        students=[StudentSummaryResponse(**s.__dict__) for s in students],
        total=total,
        period_days=days,
    )


# ─── School Stats ───────────────────────────────────────────────


@router.get(
    "/schools",
    response_model=SchoolStatsResponse,
    summary="Get per-school aggregated statistics",
    dependencies=[Depends(require_roles(["admin", "teacher"]))],
)
async def get_school_stats(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """School-level aggregated stats for cross-school comparison."""
    schools = await analytics_service.get_school_stats(db, days=days)
    return SchoolStatsResponse(
        period_days=days,
        schools=[SchoolStatsItem(**s) for s in schools],
    )
