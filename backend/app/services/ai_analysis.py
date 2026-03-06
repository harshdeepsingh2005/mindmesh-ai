"""MindMesh AI — AI Analysis Orchestrator.

Coordinates emotion detection, sentiment analysis, and trend analysis.
Stores all model predictions in the database via EmotionAnalysis records.
After each analysis, triggers a risk re-assessment for the student.
If the assessment produces a high-risk score, generates an alert and
notifies teachers.

This is the central service that routes invoke to analyze behavioral data.
"""

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.behavioral_record import BehavioralRecord
from ..models.emotion_analysis import EmotionAnalysis
from ..logging_config import logger

from .emotion_detection import detect_emotion, EmotionResult
from .sentiment_analysis import analyze_sentiment, SentimentResult


async def analyze_record(
    db: AsyncSession,
    record: BehavioralRecord,
) -> Optional[EmotionAnalysis]:
    """Run full AI analysis pipeline on a behavioral record.

    Steps:
    1. Detect emotion from text input
    2. Analyze sentiment polarity
    3. Update record with scores
    4. Store EmotionAnalysis prediction in database
    5. Trigger risk re-assessment for the student
    6. Generate alert + notify teachers if high risk

    Args:
        db: Async database session.
        record: The BehavioralRecord to analyze.

    Returns:
        EmotionAnalysis ORM object if analysis was performed, else None.
    """
    text = record.text_input

    # Run emotion detection
    emotion_result: EmotionResult = detect_emotion(text)

    # Run sentiment analysis
    sentiment_result: SentimentResult = analyze_sentiment(text)

    # Update the behavioral record with computed scores
    if record.emotion_score is None and emotion_result.predicted_emotion != "neutral":
        # Map emotion to a 0-1 score (positive emotions = high, negative = low)
        emotion_valence = {
            "happy": 0.85,
            "neutral": 0.50,
            "sad": 0.20,
            "anxious": 0.25,
            "angry": 0.20,
        }
        record.emotion_score = emotion_valence.get(
            emotion_result.predicted_emotion, 0.5
        )

    if record.sentiment_score is None:
        record.sentiment_score = sentiment_result.sentiment_score

    # Create EmotionAnalysis record
    analysis = EmotionAnalysis(
        id=str(uuid.uuid4()),
        record_id=record.id,
        predicted_emotion=emotion_result.predicted_emotion,
        confidence_score=emotion_result.confidence_score,
        model_version=emotion_result.model_version,
    )

    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    logger.info(
        f"AI analysis complete: record_id={record.id}, "
        f"emotion={emotion_result.predicted_emotion} "
        f"(conf={emotion_result.confidence_score}), "
        f"sentiment={sentiment_result.sentiment_label} "
        f"(score={sentiment_result.sentiment_score}), "
        f"high_risk={sentiment_result.high_risk_flag}"
    )

    # ── Trigger risk re-assessment + alert generation ────────────
    try:
        from .risk_scoring import assess_student_risk

        assessment = await assess_student_risk(
            db, record.student_id, lookback_days=30
        )
        logger.info(
            f"Risk re-assessment after analysis: "
            f"student_id={record.student_id}, "
            f"score={assessment.composite_score}, "
            f"level={assessment.risk_level}"
        )

        # Generate alert and notify teachers if high risk
        from .alert_service import generate_and_notify

        alert_result = await generate_and_notify(
            db,
            student_id=record.student_id,
            risk_score=assessment.composite_score,
            risk_level=assessment.risk_level,
        )
        if alert_result:
            logger.warning(
                f"Alert generated after analysis: "
                f"alert_id={alert_result['alert_id']}, "
                f"notifications={alert_result['notifications_sent']}"
            )

    except Exception as exc:
        # Non-fatal — log but don't block the analysis response
        logger.error(
            f"Post-analysis pipeline failed for "
            f"student_id={record.student_id}: {exc}",
            exc_info=True,
        )

    # ── Immediate alert on high-risk content detection ───────────
    if sentiment_result.high_risk_flag:
        try:
            from .alert_service import generate_alert, notify_teachers

            alert = await generate_alert(
                db,
                student_id=record.student_id,
                risk_score=100,
                alert_type="high_risk",
                message=(
                    f"HIGH RISK CONTENT DETECTED in {record.activity_type}. "
                    f"Keywords found: {', '.join(sentiment_result.high_risk_keywords_found)}. "
                    "Immediate counselor intervention required."
                ),
                deduplicate_minutes=30,
            )
            if alert:
                await notify_teachers(db, alert)
                logger.warning(
                    f"IMMEDIATE ALERT for high-risk content: "
                    f"student_id={record.student_id}, "
                    f"keywords={sentiment_result.high_risk_keywords_found}"
                )
        except Exception as exc:
            logger.error(
                f"High-risk alert generation failed: {exc}",
                exc_info=True,
            )

    return analysis


async def analyze_text_standalone(
    text: str,
) -> dict:
    """Run AI analysis on arbitrary text without storing results.

    Useful for real-time preview or the /emotion/analyze API endpoint.

    Args:
        text: Input text to analyze.

    Returns:
        Dict with emotion and sentiment analysis results.
    """
    emotion_result = detect_emotion(text)
    sentiment_result = analyze_sentiment(text)

    return {
        "emotion": {
            "predicted_emotion": emotion_result.predicted_emotion,
            "confidence_score": emotion_result.confidence_score,
            "emotion_scores": emotion_result.emotion_scores,
            "model_version": emotion_result.model_version,
        },
        "sentiment": {
            "sentiment_label": sentiment_result.sentiment_label,
            "sentiment_score": sentiment_result.sentiment_score,
            "positive_score": sentiment_result.positive_score,
            "negative_score": sentiment_result.negative_score,
            "high_risk_flag": sentiment_result.high_risk_flag,
            "high_risk_keywords_found": sentiment_result.high_risk_keywords_found,
            "model_version": sentiment_result.model_version,
        },
    }
