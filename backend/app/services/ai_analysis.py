"""MindMesh AI — AI Analysis Orchestrator.

Coordinates emotion detection, sentiment analysis, and trend analysis.
Stores all model predictions in the database via EmotionAnalysis records.

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
