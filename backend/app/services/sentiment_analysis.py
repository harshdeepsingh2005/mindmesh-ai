"""MindMesh AI — Sentiment Analysis Module (Unsupervised).

Analyzes text sentiment polarity using NLTK's VADER sentiment
analyser — a rule-based model specifically attuned to social media
and informal text (well-suited for student journal entries).

VADER is unsupervised — it requires NO training data.  It uses a
curated lexicon with grammatical and syntactical heuristics
(capitalisation, punctuation, intensifiers, negation, conjunctions)
to produce a compound sentiment score.

Additionally, this module maintains high-risk keyword detection for
immediate counselor escalation — this is a safety-critical feature
that operates independently of the ML pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from ..logging_config import logger

# ── NLTK Data Download (one-time) ────────────────────────────────

import os
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("sentiment/vader_lexicon.zip", paths=[nltk_data_dir])
except LookupError:
    nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=True)


# ── Constants ────────────────────────────────────────────────────

_MODEL_VERSION = "sentiment-v3.0.0-vader"

# High-risk keywords that MUST trigger immediate counselor escalation.
# This is a safety feature — NOT part of the ML pipeline.
HIGH_RISK_KEYWORDS = {
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "selfharm", "self-harm", "self harm", "cutting", "overdose",
    "no reason to live", "better off dead", "can't go on",
    "don't want to be here", "wish i was dead", "hurt myself",
}


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment_label: str        # positive, negative, neutral
    sentiment_score: float      # compound score: -1.0 to 1.0
    positive_score: float       # positive proportion (0-1)
    negative_score: float       # negative proportion (0-1)
    neutral_score: float        # neutral proportion (0-1)
    high_risk_flag: bool
    high_risk_keywords_found: List[str]
    model_version: str = ""
    vader_raw: Dict[str, float] = field(default_factory=dict)


# ── VADER Engine ─────────────────────────────────────────────────

_vader: Optional[SentimentIntensityAnalyzer] = None


def _get_vader() -> SentimentIntensityAnalyzer:
    """Get or initialise the VADER analyser (lazy singleton)."""
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
        logger.info("VADER SentimentIntensityAnalyzer initialised")
    return _vader


# ── High-Risk Detection ──────────────────────────────────────────


def _check_high_risk(text: str) -> List[str]:
    """Check for high-risk keywords in text.

    These indicate potential self-harm or suicidal ideation
    and MUST be escalated to a human counselor immediately.

    Args:
        text: Raw text input.

    Returns:
        List of matched high-risk keywords.
    """
    text_lower = text.lower()
    found = []
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return found


# ── Public API ───────────────────────────────────────────────────


def analyze_sentiment(text: Optional[str]) -> SentimentResult:
    """Analyze sentiment polarity and intensity of text using VADER.

    VADER produces four scores:
      - pos:      proportion of positive sentiment
      - neg:      proportion of negative sentiment
      - neu:      proportion of neutral sentiment
      - compound: normalised composite (-1 to 1)

    Classification thresholds (standard VADER):
      compound >=  0.05  → positive
      compound <= -0.05  → negative
      otherwise          → neutral

    Args:
        text: Input text to analyze.

    Returns:
        SentimentResult with polarity, scores, and risk flags.
    """
    if not text or not text.strip():
        return SentimentResult(
            sentiment_label="neutral",
            sentiment_score=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            high_risk_flag=False,
            high_risk_keywords_found=[],
            model_version=_MODEL_VERSION,
        )

    # High-risk check is independent of sentiment model
    high_risk_found = _check_high_risk(text)
    if high_risk_found:
        logger.warning(
            f"HIGH RISK content detected. Keywords: {high_risk_found}. "
            "Flagging for counselor escalation."
        )

    # VADER sentiment analysis
    vader = _get_vader()
    scores = vader.polarity_scores(text)

    compound = scores["compound"]
    pos = scores["pos"]
    neg = scores["neg"]
    neu = scores["neu"]

    # Classify
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Override on high-risk content — force negative
    if high_risk_found:
        label = "negative"
        compound = min(compound, -0.5)
        neg = max(neg, 0.8)

    result = SentimentResult(
        sentiment_label=label,
        sentiment_score=round(compound, 4),
        positive_score=round(pos, 4),
        negative_score=round(neg, 4),
        neutral_score=round(neu, 4),
        high_risk_flag=bool(high_risk_found),
        high_risk_keywords_found=high_risk_found,
        model_version=_MODEL_VERSION,
        vader_raw=scores,
    )

    logger.debug(
        f"Sentiment: {label} (compound={compound:.4f}), "
        f"pos={pos:.4f}, neg={neg:.4f}, neu={neu:.4f}, "
        f"high_risk={result.high_risk_flag}"
    )

    return result
