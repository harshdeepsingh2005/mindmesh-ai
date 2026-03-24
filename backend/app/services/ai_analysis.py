"""MindMesh AI — AI Analysis Orchestrator (Unsupervised).

Coordinates the unsupervised analysis pipeline:
  1. VADER sentiment analysis
  2. K-Means emotion cluster assignment
  3. NMF topic discovery
  4. Risk re-assessment (Isolation Forest anomaly detection)
  5. Alert generation + teacher notification

This is the central service that routes invoke to analyze
behavioral data through the unsupervised ML pipeline.
"""

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.behavioral_record import BehavioralRecord
from ..models.emotion_analysis import EmotionAnalysis
from ..logging_config import logger

from .emotion_detection import detect_emotion, EmotionResult
from .sentiment_analysis import analyze_sentiment, SentimentResult
from .topic_discovery import get_topic_engine


async def analyze_record(
    db: AsyncSession,
    record: BehavioralRecord,
) -> Optional[EmotionAnalysis]:
    """Run full unsupervised AI analysis pipeline on a behavioral record.

    Steps:
    1. Analyze sentiment with VADER
    2. Assign emotion cluster via K-Means
    3. Discover topic via NMF
    4. Update record with scores
    5. Store EmotionAnalysis prediction in database
    6. Trigger risk re-assessment (Isolation Forest)
    7. Generate alert + notify teachers if high risk

    Args:
        db: Async database session.
        record: The BehavioralRecord to analyze.

    Returns:
        EmotionAnalysis ORM object if analysis was performed, else None.
    """
    text = record.text_input

    # 1. VADER sentiment analysis
    sentiment_result: SentimentResult = analyze_sentiment(text)

    # 2. K-Means emotion cluster assignment
    emotion_result: EmotionResult = detect_emotion(text)

    # 3. NMF topic discovery (if engine is fitted)
    topic_engine = get_topic_engine()
    topic_label = "unknown"
    if topic_engine.is_fitted and text:
        topic_result = topic_engine.predict(text)
        topic_label = topic_result.topic_label

    # 4. Update the behavioral record with computed scores
    if record.emotion_score is None:
        # Map sentiment compound score to 0-1 (positive = high)
        record.emotion_score = (sentiment_result.sentiment_score + 1) / 2

    if record.sentiment_score is None:
        record.sentiment_score = sentiment_result.sentiment_score

    # 5. Create EmotionAnalysis record
    analysis = EmotionAnalysis(
        id=str(uuid.uuid4()),
        record_id=record.id,
        predicted_emotion=emotion_result.cluster_label,
        confidence_score=emotion_result.confidence_score,
        model_version=emotion_result.model_version,
    )

    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    logger.info(
        f"AI analysis complete: record_id={record.id}, "
        f"emotion_cluster={emotion_result.cluster_label} "
        f"(conf={emotion_result.confidence_score}), "
        f"sentiment={sentiment_result.sentiment_label} "
        f"(compound={sentiment_result.sentiment_score}), "
        f"topic={topic_label}, "
        f"high_risk={sentiment_result.high_risk_flag}"
    )

    # ── 6. Trigger risk re-assessment ────────────────────────
    try:
        from .risk_scoring import assess_student_risk

        assessment = await assess_student_risk(
            db, record.student_id, lookback_days=30
        )
        logger.info(
            f"Risk re-assessment: "
            f"student_id={record.student_id}, "
            f"score={assessment.composite_score}, "
            f"level={assessment.risk_level}, "
            f"anomaly_factor={assessment.factors.anomaly_score:.2f}"
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
                f"Alert generated: "
                f"alert_id={alert_result['alert_id']}, "
                f"notifications={alert_result['notifications_sent']}"
            )

    except Exception as exc:
        logger.error(
            f"Post-analysis pipeline failed for "
            f"student_id={record.student_id}: {exc}",
            exc_info=True,
        )

    # ── Immediate alert on high-risk content ─────────────────
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
        Dict with emotion, sentiment, and topic analysis results.
    """
    emotion_result = detect_emotion(text)
    sentiment_result = analyze_sentiment(text)

    # Topic analysis (if fitted)
    topic_info = {}
    topic_engine = get_topic_engine()
    if topic_engine.is_fitted:
        topic_result = topic_engine.predict(text)
        topic_info = {
            "dominant_topic": topic_result.dominant_topic,
            "topic_label": topic_result.topic_label,
            "topic_distribution": topic_result.topic_distribution,
            "confidence": topic_result.confidence,
        }

    return {
        "emotion": {
            "predicted_cluster": emotion_result.predicted_cluster,
            "cluster_label": emotion_result.cluster_label,
            "confidence_score": emotion_result.confidence_score,
            "cluster_distances": emotion_result.cluster_distances,
            "top_terms": emotion_result.top_terms,
            "model_version": emotion_result.model_version,
        },
        "sentiment": {
            "sentiment_label": sentiment_result.sentiment_label,
            "sentiment_score": sentiment_result.sentiment_score,
            "positive_score": sentiment_result.positive_score,
            "negative_score": sentiment_result.negative_score,
            "neutral_score": sentiment_result.neutral_score,
            "high_risk_flag": sentiment_result.high_risk_flag,
            "high_risk_keywords_found": sentiment_result.high_risk_keywords_found,
            "model_version": sentiment_result.model_version,
        },
        "topic": topic_info,
    }
