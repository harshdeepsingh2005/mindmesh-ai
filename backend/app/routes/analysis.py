"""MindMesh AI — AI Analysis Routes.

Provides endpoints for emotion analysis, sentiment analysis,
and behavioral trend analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..database.schemas import (
    EmotionAnalyzeRequest,
    EmotionAnalyzeResponse,
    EmotionAnalysisResponse,
    TrendAnalysisResponse,
)
from ..models.user import User
from ..models.behavioral_record import BehavioralRecord
from ..models.emotion_analysis import EmotionAnalysis
from ..dependencies import get_current_user, require_roles
from ..services.ai_analysis import analyze_record, analyze_text_standalone
from ..services.trend_analysis import analyze_trends
from ..services.student_service import StudentService
from ..logging_config import logger

router = APIRouter()


@router.post(
    "/analyze",
    response_model=EmotionAnalyzeResponse,
    summary="Analyze text for emotion and sentiment",
)
async def analyze_text(
    payload: EmotionAnalyzeRequest,
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
) -> EmotionAnalyzeResponse:
    """Analyze text input for emotion and sentiment.

    Does not store results in database — use for real-time analysis.
    For stored analysis, submit via /student/checkin or /student/journal.

    Args:
        payload: Text input and optional student_id.
        current_user: Authenticated user.

    Returns:
        Emotion and sentiment analysis results.
    """
    result = await analyze_text_standalone(payload.text_input)
    logger.info(
        f"Standalone analysis by user={current_user.id}: "
        f"emotion={result['emotion']['cluster_label']}"
    )
    return EmotionAnalyzeResponse(**result)


@router.post(
    "/analyze-record/{record_id}",
    response_model=EmotionAnalysisResponse,
    summary="Run AI analysis on a behavioral record",
)
async def analyze_behavioral_record(
    record_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> EmotionAnalysisResponse:
    """Run the full AI analysis pipeline on a specific behavioral record.

    Creates an EmotionAnalysis entry and updates the record's scores.
    Teacher/admin only.

    Args:
        record_id: The behavioral record ID to analyze.
        db: Database session.
        current_user: Authenticated teacher/admin.

    Returns:
        The created EmotionAnalysis result.

    Raises:
        HTTPException 404: If record not found.
        HTTPException 409: If record already has analysis.
    """
    # Fetch record
    result = await db.execute(
        select(BehavioralRecord).where(BehavioralRecord.id == record_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Behavioral record with id={record_id} not found.",
        )

    # Check if already analyzed
    existing = await db.execute(
        select(EmotionAnalysis).where(EmotionAnalysis.record_id == record_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Record {record_id} already has an emotion analysis.",
        )

    analysis = await analyze_record(db, record)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Analysis could not be performed on this record.",
        )

    return EmotionAnalysisResponse.model_validate(analysis)


@router.get(
    "/analysis/{record_id}",
    response_model=EmotionAnalysisResponse,
    summary="Get emotion analysis for a record",
)
async def get_emotion_analysis(
    record_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
) -> EmotionAnalysisResponse:
    """Retrieve the stored emotion analysis for a behavioral record.

    Args:
        record_id: The behavioral record ID.
        db: Database session.
        current_user: Authenticated user.

    Returns:
        The stored EmotionAnalysis result.

    Raises:
        HTTPException 404: If no analysis found.
    """
    result = await db.execute(
        select(EmotionAnalysis).where(EmotionAnalysis.record_id == record_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No emotion analysis found for record_id={record_id}.",
        )

    return EmotionAnalysisResponse.model_validate(analysis)


@router.get(
    "/trends/{student_id}",
    response_model=TrendAnalysisResponse,
    summary="Analyze behavioral trends for a student",
)
async def get_behavioral_trends(
    student_id: str,
    days: int = Query(30, ge=7, le=365, description="Analysis period in days"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "teacher", "student"])),
) -> TrendAnalysisResponse:
    """Analyze behavioral trends for a student over a given period.

    Students can only view their own trends. Teachers/admins can view any student.

    Args:
        student_id: The student's ID.
        days: Number of days to analyze.
        db: Database session.
        current_user: Authenticated user.

    Returns:
        Comprehensive trend analysis results.

    Raises:
        HTTPException 404: If student not found.
        HTTPException 403: If student tries to view another's trends.
    """
    # Verify student exists and access permissions
    svc = StudentService(db)
    student = await svc._get_student_or_404(student_id)

    if current_user.role == "student":
        if not student.user_id or student.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own trends.",
            )

    trend_result = await analyze_trends(db, student_id, days=days)

    return TrendAnalysisResponse(
        student_id=trend_result.student_id,
        period_days=trend_result.period_days,
        total_records=trend_result.total_records,
        data_points=trend_result.data_points,
        emotion_trend=trend_result.emotion_trend,
        emotion_slope=trend_result.emotion_slope,
        avg_emotion_score=trend_result.avg_emotion_score,
        emotion_volatility=trend_result.emotion_volatility,
        checkin_count=trend_result.checkin_count,
        journal_count=trend_result.journal_count,
        checkin_frequency=trend_result.checkin_frequency,
        weekly_averages=trend_result.weekly_averages,
        declining_flag=trend_result.declining_flag,
        disengagement_flag=trend_result.disengagement_flag,
    )
