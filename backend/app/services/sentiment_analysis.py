"""MindMesh AI — Sentiment Analysis Module.

Analyzes text sentiment polarity (positive/negative/neutral) and
intensity. Uses a lexicon-based approach with VADER-inspired scoring.

Upgradeable to transformer-based models in later phases.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..logging_config import logger

MODEL_VERSION = "sentiment-v0.2.0-lexicon"

# Positive and negative word lexicons with weights
POSITIVE_WORDS: Dict[str, float] = {
    "good": 0.6, "great": 0.8, "excellent": 0.9, "wonderful": 0.9,
    "amazing": 0.9, "fantastic": 0.9, "awesome": 0.8, "love": 0.8,
    "happy": 0.8, "joy": 0.9, "excited": 0.8, "grateful": 0.8,
    "thankful": 0.7, "blessed": 0.7, "proud": 0.7, "confident": 0.7,
    "hopeful": 0.7, "positive": 0.6, "better": 0.6, "best": 0.8,
    "beautiful": 0.7, "kind": 0.6, "friendly": 0.6, "helpful": 0.6,
    "enjoy": 0.7, "fun": 0.7, "nice": 0.5, "pleasant": 0.6,
    "comfortable": 0.6, "peaceful": 0.7, "calm": 0.6, "relaxed": 0.6,
    "safe": 0.6, "supported": 0.7, "understood": 0.7, "accepted": 0.7,
    "improve": 0.6, "progress": 0.6, "achieve": 0.7, "success": 0.7,
    "smile": 0.6, "laugh": 0.7, "celebrate": 0.7, "win": 0.6,
}

NEGATIVE_WORDS: Dict[str, float] = {
    "bad": 0.6, "terrible": 0.9, "horrible": 0.9, "awful": 0.9,
    "hate": 0.8, "sad": 0.8, "depressed": 1.0, "anxious": 0.8,
    "angry": 0.8, "frustrated": 0.7, "stressed": 0.8, "worried": 0.7,
    "scared": 0.8, "lonely": 0.8, "alone": 0.7, "hopeless": 1.0,
    "worthless": 1.0, "helpless": 0.9, "miserable": 0.9, "hurt": 0.7,
    "pain": 0.7, "suffer": 0.8, "cry": 0.7, "crying": 0.8,
    "tears": 0.7, "broken": 0.8, "lost": 0.6, "confused": 0.6,
    "overwhelmed": 0.8, "exhausted": 0.7, "tired": 0.5, "sick": 0.6,
    "ugly": 0.7, "stupid": 0.7, "dumb": 0.6, "fail": 0.7,
    "failure": 0.8, "worst": 0.9, "worse": 0.7, "trouble": 0.6,
    "problem": 0.5, "difficult": 0.5, "hard": 0.4, "struggle": 0.6,
    "bully": 0.8, "bullied": 0.9, "abuse": 0.9, "neglect": 0.8,
    "panic": 0.9, "fear": 0.8, "nightmare": 0.8, "death": 0.9,
    "die": 0.9, "kill": 1.0, "suicide": 1.0, "selfharm": 1.0,
}

# High-risk keywords that require immediate counselor escalation
HIGH_RISK_KEYWORDS = {
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "selfharm", "self-harm", "self harm", "cutting", "overdose",
    "no reason to live", "better off dead", "can't go on",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "can't", "cannot", "hardly",
    "barely", "scarcely", "isn't", "aren't", "wasn't", "weren't",
}

INTENSIFIERS = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "so": 1.2,
    "incredibly": 1.4, "absolutely": 1.4, "totally": 1.3,
    "completely": 1.4, "deeply": 1.3, "severely": 1.4,
}


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment_label: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 (most negative) to 1.0 (most positive)
    positive_score: float
    negative_score: float
    high_risk_flag: bool
    high_risk_keywords_found: List[str]
    model_version: str = MODEL_VERSION


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    text = text.lower()
    return re.sub(r"[^a-z\s']", " ", text).split()


def _check_high_risk(text: str) -> List[str]:
    """Check for high-risk keywords in text.

    These indicate potential self-harm or suicidal ideation
    and MUST be escalated to a human counselor immediately.
    """
    text_lower = text.lower()
    found = []
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return found


def analyze_sentiment(text: Optional[str]) -> SentimentResult:
    """Analyze sentiment polarity and intensity of text.

    Scores text on a scale from -1.0 (most negative) to 1.0 (most positive).
    Also flags high-risk content for counselor escalation.

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
            high_risk_flag=False,
            high_risk_keywords_found=[],
        )

    # Check for high-risk content first
    high_risk_found = _check_high_risk(text)
    if high_risk_found:
        logger.warning(
            f"HIGH RISK content detected. Keywords: {high_risk_found}. "
            "Flagging for counselor escalation."
        )

    tokens = _tokenize(text)
    if not tokens:
        return SentimentResult(
            sentiment_label="neutral",
            sentiment_score=0.0,
            positive_score=0.0,
            negative_score=0.0,
            high_risk_flag=bool(high_risk_found),
            high_risk_keywords_found=high_risk_found,
        )

    pos_score = 0.0
    neg_score = 0.0

    for i, token in enumerate(tokens):
        # Check for intensifier
        intensifier = 1.0
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            intensifier = INTENSIFIERS[tokens[i - 1]]

        # Check for negation
        is_negated = False
        start = max(0, i - 3)
        for j in range(start, i):
            if tokens[j] in NEGATION_WORDS:
                is_negated = True
                break

        if token in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[token] * intensifier
            if is_negated:
                neg_score += weight * 0.5
            else:
                pos_score += weight

        if token in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[token] * intensifier
            if is_negated:
                pos_score += weight * 0.3
            else:
                neg_score += weight

    # Normalize by token count
    token_count = max(len(tokens), 1)
    pos_normalized = min(1.0, pos_score / (token_count * 0.3))
    neg_normalized = min(1.0, neg_score / (token_count * 0.3))

    # Calculate composite score (-1 to 1)
    composite = pos_normalized - neg_normalized
    composite = max(-1.0, min(1.0, composite))

    # Determine label
    if composite > 0.1:
        label = "positive"
    elif composite < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # If high-risk content found, force negative bias
    if high_risk_found:
        label = "negative"
        composite = min(composite, -0.5)
        neg_normalized = max(neg_normalized, 0.8)

    result = SentimentResult(
        sentiment_label=label,
        sentiment_score=round(composite, 4),
        positive_score=round(pos_normalized, 4),
        negative_score=round(neg_normalized, 4),
        high_risk_flag=bool(high_risk_found),
        high_risk_keywords_found=high_risk_found,
    )

    logger.debug(
        f"Sentiment: {label} (score={composite:.4f}), "
        f"pos={pos_normalized:.4f}, neg={neg_normalized:.4f}, "
        f"high_risk={result.high_risk_flag}"
    )

    return result
