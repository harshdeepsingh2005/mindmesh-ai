"""MindMesh AI — Counselor / Risk-Assessment Routes.

Endpoints for:
  • Triggering risk assessments (single + batch)
  • Retrieving risk scores & history
  • Viewing mental-health trajectory
  • Fetching intervention recommendations
  • Querying risk-level thresholds
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db
from ..database.schemas import (
    RiskScoreResponse,
    RiskScoreListResponse,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    BatchRiskAssessmentRequest,
    BatchRiskAssessmentResponse,
    RiskThresholdsResponse,
)
from ..dependencies import get_current_user, require_roles
from ..models.user import User
from ..services.risk_scoring import (
    assess_student_risk,
    get_latest_risk_score,
    get_risk_history,
    batch_assess_students,
    RISK_THRESHOLDS,
    FACTOR_WEIGHTS,
)
from ..services.trend_analysis import analyze_trend
from ..logging_config import logger

router = APIRouter()


# ── 1. Trigger a single risk assessment ─────────────────────────

@router.post(
    "/risk/assess",
    response_model=RiskAssessmentResponse,
    summary="Assess a student's risk score",
)
async def assess_risk(
    body: RiskAssessmentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Run the multi-factor risk-scoring pipeline for one student."""
    assessment = await assess_student_risk(
        db,
        body.student_id,
        lookback_days=body.lookback_days or 30,
    )
    return RiskAssessmentResponse(
        student_id=assessment.student_id,
        composite_score=assessment.composite_score,
        risk_level=assessment.risk_level,
        contributing_factors=assessment.factors.to_dict(),
        assessed_at=assessment.assessed_at,
    )


# ── 2. Get latest risk score for a student ──────────────────────

@router.get(
    "/risk/{student_id}",
    response_model=RiskScoreResponse,
    summary="Get latest risk score",
)
async def get_risk_score(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Return the most recent risk score for the given student."""
    score = await get_latest_risk_score(db, student_id)
    if not score:
        raise HTTPException(
            status_code=404,
            detail=f"No risk score found for student {student_id}",
        )
    return score


# ── 3. Risk-score history ───────────────────────────────────────

@router.get(
    "/risk/{student_id}/history",
    response_model=RiskScoreListResponse,
    summary="Get risk score history",
)
async def get_risk_score_history(
    student_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Return past risk scores for a student, newest first."""
    scores = await get_risk_history(db, student_id, limit=limit)
    return RiskScoreListResponse(
        student_id=student_id,
        scores=[RiskScoreResponse.model_validate(s) for s in scores],
        total=len(scores),
    )


# ── 4. Batch risk assessment ────────────────────────────────────

@router.post(
    "/risk/batch",
    response_model=BatchRiskAssessmentResponse,
    summary="Batch-assess multiple students",
)
async def batch_risk_assessment(
    body: BatchRiskAssessmentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Run risk assessments for a list of student IDs."""
    assessments = await batch_assess_students(
        db,
        body.student_ids,
        lookback_days=body.lookback_days or 30,
    )
    results = [
        RiskAssessmentResponse(
            student_id=a.student_id,
            composite_score=a.composite_score,
            risk_level=a.risk_level,
            contributing_factors=a.factors.to_dict(),
            assessed_at=a.assessed_at,
        )
        for a in assessments
    ]
    return BatchRiskAssessmentResponse(
        assessments=results,
        total=len(results),
    )


# ── 5. Risk-level thresholds (config) ───────────────────────────

@router.get(
    "/risk/config/thresholds",
    response_model=RiskThresholdsResponse,
    summary="Get risk threshold configuration",
)
async def get_risk_thresholds(
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Return the current risk-level thresholds and factor weights."""
    return RiskThresholdsResponse(
        thresholds={
            level: {"min": lo, "max": hi}
            for level, (lo, hi) in RISK_THRESHOLDS.items()
        },
        factor_weights=FACTOR_WEIGHTS,
    )


# ── 6. Mental-health trajectory ─────────────────────────────────

@router.get(
    "/trajectory/{student_id}",
    summary="Get mental-health trajectory",
)
async def get_trajectory(
    student_id: str,
    days: int = Query(default=30, ge=7, le=180),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Return mood-trend and risk-score trajectory over the window."""
    trend = await analyze_trend(db, student_id, days=days)
    risk_scores = await get_risk_history(db, student_id, limit=days)

    return {
        "student_id": student_id,
        "days": days,
        "trend_analysis": trend,
        "risk_history": [
            {
                "score": rs.risk_score,
                "level": rs.risk_level,
                "date": rs.calculated_at.isoformat(),
            }
            for rs in risk_scores
        ],
    }


# ── 7. Intervention recommendations ─────────────────────────────

@router.get(
    "/recommendations/{student_id}",
    summary="Get intervention recommendations",
)
async def get_recommendations(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("counselor", "admin")),
):
    """Generate rule-based intervention recommendations."""
    score_row = await get_latest_risk_score(db, student_id)
    if not score_row:
        raise HTTPException(
            status_code=404,
            detail=f"No risk data for student {student_id}",
        )

    recommendations: List[dict] = []
    factors = score_row.contributing_factors or {}

    # Rule-based recommendation engine
    if score_row.risk_level == "high":
        recommendations.append({
            "priority": "urgent",
            "action": "Schedule immediate counseling session",
            "reason": f"Risk score {score_row.risk_score}/100 — HIGH level",
        })

    if factors.get("high_risk_keywords", 0) > 50:
        recommendations.append({
            "priority": "urgent",
            "action": "Review flagged journal entries for crisis language",
            "reason": "High-risk keyword frequency exceeds threshold",
        })

    if factors.get("sentiment_score", 0) > 60:
        recommendations.append({
            "priority": "high",
            "action": "Conduct one-on-one emotional check-in",
            "reason": "Consistently negative sentiment detected",
        })

    if factors.get("emotion_intensity", 0) > 60:
        recommendations.append({
            "priority": "high",
            "action": "Assess for anxiety or depression indicators",
            "reason": "Strong negative emotional intensity",
        })

    if factors.get("trend_direction", 0) > 50:
        recommendations.append({
            "priority": "medium",
            "action": "Monitor trend — consider peer-support referral",
            "reason": "Worsening behavioral trend",
        })

    if factors.get("mood_variability", 0) > 50:
        recommendations.append({
            "priority": "medium",
            "action": "Introduce mood-regulation exercises",
            "reason": "High mood variability observed",
        })

    if not recommendations:
        recommendations.append({
            "priority": "low",
            "action": "Continue routine monitoring",
            "reason": "All factors within normal range",
        })

    return {
        "student_id": student_id,
        "current_risk_score": score_row.risk_score,
        "current_risk_level": score_row.risk_level,
        "recommendations": recommendations,
        "generated_at": score_row.calculated_at.isoformat(),
    }
