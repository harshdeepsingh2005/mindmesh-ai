"""Tests for the AI analysis pipeline — emotion clustering & sentiment analysis.

Updated for the unsupervised ML pipeline:
  • Emotion detection now uses K-Means clustering (EmotionResult has cluster_label)
  • Sentiment analysis now uses VADER (SentimentResult has compound score)
"""

import pytest

from app.services.emotion_detection import detect_emotion, EmotionResult
from app.services.sentiment_analysis import (
    analyze_sentiment,
    SentimentResult,
    HIGH_RISK_KEYWORDS,
)


# ─── Emotion Detection (K-Means Clustering) ─────────────────────


class TestEmotionDetection:
    """Unit tests for detect_emotion() — K-Means cluster assignment."""

    def test_detects_positive_text(self):
        result = detect_emotion("I am so happy and excited today!")
        assert isinstance(result, EmotionResult)
        assert result.cluster_label != ""
        assert result.confidence_score > 0

    def test_detects_negative_text(self):
        result = detect_emotion("I feel really sad and lonely")
        assert isinstance(result, EmotionResult)
        assert result.confidence_score > 0

    def test_detects_anxious_text(self):
        result = detect_emotion("I'm very anxious and worried about exams")
        assert isinstance(result, EmotionResult)
        assert result.confidence_score > 0

    def test_detects_angry_text(self):
        result = detect_emotion("I am furious and really angry right now")
        assert isinstance(result, EmotionResult)
        assert result.confidence_score > 0

    def test_neutral_text(self):
        result = detect_emotion("Today was an ordinary day at school")
        assert isinstance(result, EmotionResult)

    def test_empty_string(self):
        result = detect_emotion("")
        assert result.cluster_label == "neutral"
        assert result.confidence_score == 0.5

    def test_none_input(self):
        result = detect_emotion(None)
        assert result.cluster_label == "neutral"
        assert result.confidence_score == 0.5

    def test_model_version_present(self):
        result = detect_emotion("test text")
        assert result.model_version != ""
        assert isinstance(result.model_version, str)

    def test_returns_cluster_distances(self):
        result = detect_emotion("I feel happy but also a bit anxious")
        # Cluster distances are only populated when the engine is fitted
        assert isinstance(result.cluster_distances, dict)

    def test_returns_top_terms(self):
        result = detect_emotion("I am very happy and excited")
        assert isinstance(result.top_terms, list)

    def test_predicted_cluster_is_int(self):
        result = detect_emotion("Today was a great day")
        assert isinstance(result.predicted_cluster, int)


# ─── Sentiment Analysis (VADER) ─────────────────────────────────


class TestSentimentAnalysis:
    """Unit tests for analyze_sentiment() — VADER sentiment analyser."""

    def test_positive_text(self):
        result = analyze_sentiment("I love this school, it's great and wonderful!")
        assert result.sentiment_label == "positive"
        assert result.sentiment_score > 0

    def test_negative_text(self):
        result = analyze_sentiment("I feel terrible and miserable today")
        assert result.sentiment_label == "negative"
        assert result.sentiment_score < 0

    def test_neutral_text(self):
        result = analyze_sentiment("The weather is okay")
        # Should be neutral or close to neutral
        assert abs(result.sentiment_score) <= 0.5

    def test_empty_string(self):
        result = analyze_sentiment("")
        assert result.sentiment_label == "neutral"
        assert result.sentiment_score == 0.0
        assert result.high_risk_flag is False

    def test_none_input(self):
        result = analyze_sentiment(None)
        assert result.sentiment_label == "neutral"
        assert result.sentiment_score == 0.0

    def test_high_risk_detection(self):
        result = analyze_sentiment("I want to end my life")
        assert result.high_risk_flag is True
        assert len(result.high_risk_keywords_found) > 0
        assert result.sentiment_label == "negative"

    def test_no_risk_in_positive(self):
        result = analyze_sentiment("I am happy and grateful for my life")
        assert result.high_risk_flag is False
        assert result.high_risk_keywords_found == []

    def test_score_range(self):
        texts = [
            "absolutely amazing wonderful day",
            "terrible horrible miserable day",
            "ordinary normal average day",
        ]
        for text in texts:
            result = analyze_sentiment(text)
            assert -1.0 <= result.sentiment_score <= 1.0
            assert 0.0 <= result.positive_score <= 1.0
            assert 0.0 <= result.negative_score <= 1.0

    def test_model_version_present(self):
        result = analyze_sentiment("test text")
        assert result.model_version != ""
        assert isinstance(result.model_version, str)

    def test_neutral_score_present(self):
        result = analyze_sentiment("This is some text")
        assert 0.0 <= result.neutral_score <= 1.0

    def test_vader_raw_scores(self):
        result = analyze_sentiment("I love this!")
        assert isinstance(result.vader_raw, dict)
        assert "compound" in result.vader_raw
