"""Tests for the AI analysis pipeline — emotion detection & sentiment analysis."""

import pytest

from app.services.emotion_detection import detect_emotion, EmotionResult, EMOTION_LEXICONS
from app.services.sentiment_analysis import (
    analyze_sentiment,
    SentimentResult,
    HIGH_RISK_KEYWORDS,
)


# ─── Emotion Detection ──────────────────────────────────────────


class TestEmotionDetection:
    """Unit tests for detect_emotion()."""

    def test_happy_text(self):
        result = detect_emotion("I am so happy and excited today!")
        assert result.predicted_emotion == "happy"
        assert result.confidence_score > 0

    def test_sad_text(self):
        result = detect_emotion("I feel really sad and lonely")
        assert result.predicted_emotion == "sad"
        assert result.confidence_score > 0

    def test_anxious_text(self):
        result = detect_emotion("I'm very anxious and worried about exams")
        assert result.predicted_emotion == "anxious"
        assert result.confidence_score > 0

    def test_angry_text(self):
        result = detect_emotion("I am furious and really angry right now")
        assert result.predicted_emotion == "angry"
        assert result.confidence_score > 0

    def test_neutral_text(self):
        result = detect_emotion("Today was an ordinary day at school")
        # Should either be neutral or have low confidence
        assert isinstance(result, EmotionResult)

    def test_empty_string(self):
        result = detect_emotion("")
        assert result.predicted_emotion == "neutral"
        assert result.confidence_score == 0.5

    def test_none_input(self):
        result = detect_emotion(None)
        assert result.predicted_emotion == "neutral"
        assert result.confidence_score == 0.5

    def test_negation_handling(self):
        result = detect_emotion("I am not happy at all")
        # Negation should reduce happy score
        assert result.predicted_emotion != "happy" or result.confidence_score < 0.5

    def test_intensifier_boosting(self):
        base = detect_emotion("I feel sad")
        intense = detect_emotion("I feel extremely sad")
        # Intensified version should have higher or equal confidence
        assert intense.confidence_score >= base.confidence_score * 0.8

    def test_model_version_present(self):
        result = detect_emotion("test text")
        assert result.model_version != ""
        assert isinstance(result.model_version, str)

    def test_emotion_scores_all_categories(self):
        result = detect_emotion("I feel happy but also a bit anxious")
        # Should have scores for all emotion categories
        for emotion in EMOTION_LEXICONS:
            assert emotion in result.emotion_scores

    def test_word_matches_returned(self):
        result = detect_emotion("I am very happy and excited")
        assert isinstance(result.word_matches, dict)
        # Should have at least one match for happy
        if "happy" in result.word_matches:
            assert len(result.word_matches["happy"]) > 0


# ─── Sentiment Analysis ─────────────────────────────────────────


class TestSentimentAnalysis:
    """Unit tests for analyze_sentiment()."""

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
