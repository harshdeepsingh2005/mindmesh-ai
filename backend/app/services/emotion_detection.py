"""MindMesh AI — Emotion Detection Pipeline.

Detects emotional state from text input using keyword-based
heuristics and scoring. Designed as a pluggable pipeline that
can be upgraded to transformer-based models in later phases.

NOTE: This module provides risk INDICATORS only. It does NOT
generate clinical diagnoses. All high-risk outputs must be
escalated to human counselors.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..logging_config import logger

# Fallback model version identifier
_FALLBACK_VERSION = "emotion-v0.2.0-keyword"


def _get_model_version() -> str:
    """Get the active emotion detection model version from the registry."""
    try:
        from .model_registry import registry
        return registry.get_active_version_string("emotion_detection")
    except Exception:
        return _FALLBACK_VERSION

# Emotion lexicons — weighted keyword lists for each emotion category
EMOTION_LEXICONS: Dict[str, Dict[str, float]] = {
    "happy": {
        "happy": 1.0, "joy": 1.0, "excited": 0.9, "great": 0.8,
        "wonderful": 0.9, "amazing": 0.9, "good": 0.6, "love": 0.8,
        "grateful": 0.8, "thankful": 0.7, "cheerful": 0.9, "glad": 0.7,
        "delighted": 0.9, "fantastic": 0.9, "awesome": 0.8, "pleased": 0.7,
        "smile": 0.7, "laugh": 0.8, "fun": 0.7, "enjoy": 0.7,
        "positive": 0.6, "hope": 0.6, "proud": 0.7, "confident": 0.7,
    },
    "sad": {
        "sad": 1.0, "unhappy": 0.9, "depressed": 1.0, "down": 0.7,
        "miserable": 0.9, "crying": 0.9, "cry": 0.8, "tears": 0.8,
        "lonely": 0.8, "alone": 0.7, "hopeless": 1.0, "empty": 0.8,
        "numb": 0.8, "worthless": 1.0, "helpless": 0.9, "grief": 0.9,
        "loss": 0.7, "heartbroken": 0.9, "gloomy": 0.8, "despair": 1.0,
        "hurt": 0.7, "pain": 0.7, "suffer": 0.8, "broken": 0.8,
    },
    "anxious": {
        "anxious": 1.0, "anxiety": 1.0, "worried": 0.9, "nervous": 0.9,
        "scared": 0.8, "fear": 0.9, "panic": 1.0, "stressed": 0.9,
        "stress": 0.8, "overwhelmed": 0.9, "tense": 0.8, "restless": 0.7,
        "uneasy": 0.7, "dread": 0.9, "terrified": 1.0, "frightened": 0.9,
        "phobia": 0.9, "obsess": 0.8, "overthink": 0.8, "paranoid": 0.8,
        "worry": 0.8, "trembling": 0.7, "shaking": 0.7, "sweating": 0.6,
    },
    "angry": {
        "angry": 1.0, "anger": 1.0, "furious": 1.0, "mad": 0.8,
        "frustrated": 0.8, "irritated": 0.7, "annoyed": 0.7, "rage": 1.0,
        "hate": 0.9, "hostile": 0.9, "aggressive": 0.8, "bitter": 0.7,
        "resentful": 0.8, "outraged": 0.9, "livid": 1.0, "enraged": 1.0,
        "violent": 0.9, "fight": 0.6, "yell": 0.7, "scream": 0.7,
    },
    "neutral": {
        "okay": 0.6, "fine": 0.5, "alright": 0.5, "normal": 0.6,
        "average": 0.5, "nothing": 0.4, "regular": 0.5, "usual": 0.5,
        "moderate": 0.5, "calm": 0.6, "stable": 0.6, "balanced": 0.6,
    },
}

# Negation words that flip sentiment
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "isn't", "aren't", "wasn't", "weren't", "hardly", "barely",
    "scarcely", "seldom", "rarely",
}

# Intensifier words that boost scores
INTENSIFIERS = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "so": 1.2,
    "incredibly": 1.4, "absolutely": 1.4, "totally": 1.3,
    "completely": 1.4, "deeply": 1.3, "severely": 1.4,
    "terribly": 1.3, "awfully": 1.3, "immensely": 1.4,
}


@dataclass
class EmotionResult:
    """Result of emotion detection analysis."""
    predicted_emotion: str
    confidence_score: float
    emotion_scores: Dict[str, float]
    model_version: str = ""
    word_matches: Dict[str, List[str]] = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return text.split()


def _check_negation_window(tokens: List[str], index: int, window: int = 3) -> bool:
    """Check if a negation word appears within a window before the given index."""
    start = max(0, index - window)
    for i in range(start, index):
        if tokens[i] in NEGATION_WORDS:
            return True
    return False


def _get_intensifier(tokens: List[str], index: int) -> float:
    """Check if an intensifier word precedes the given index."""
    if index > 0 and tokens[index - 1] in INTENSIFIERS:
        return INTENSIFIERS[tokens[index - 1]]
    return 1.0


def detect_emotion(text: Optional[str]) -> EmotionResult:
    """Detect the primary emotion from text input.

    Uses keyword-matching with negation handling and intensifiers
    to score text against emotion lexicons.

    Args:
        text: Input text to analyze. Can be journal entry, check-in notes, etc.

    Returns:
        EmotionResult with predicted emotion, confidence, and per-emotion scores.
    """
    version = _get_model_version()

    if not text or not text.strip():
        return EmotionResult(
            predicted_emotion="neutral",
            confidence_score=0.5,
            emotion_scores={"neutral": 0.5},
            model_version=version,
        )

    tokens = _tokenize(text)
    if not tokens:
        return EmotionResult(
            predicted_emotion="neutral",
            confidence_score=0.5,
            emotion_scores={"neutral": 0.5},
            model_version=version,
        )

    emotion_scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTION_LEXICONS}
    word_matches: Dict[str, List[str]] = {emotion: [] for emotion in EMOTION_LEXICONS}
    total_matches = 0

    for i, token in enumerate(tokens):
        for emotion, lexicon in EMOTION_LEXICONS.items():
            if token in lexicon:
                base_weight = lexicon[token]
                intensifier = _get_intensifier(tokens, i)
                is_negated = _check_negation_window(tokens, i)

                if is_negated:
                    # Negation: reduce this emotion, slightly boost opposite
                    score = -base_weight * 0.5 * intensifier
                else:
                    score = base_weight * intensifier

                emotion_scores[emotion] += score
                word_matches[emotion].append(token)
                total_matches += 1

    # If no matches, return neutral
    if total_matches == 0:
        return EmotionResult(
            predicted_emotion="neutral",
            confidence_score=0.4,
            emotion_scores={"neutral": 0.4},
            model_version=version,
        )

    # Normalize scores to 0-1 range
    max_raw = max(abs(v) for v in emotion_scores.values()) if emotion_scores else 1.0
    if max_raw > 0:
        normalized = {
            k: max(0.0, min(1.0, (v / max_raw + 1) / 2))
            for k, v in emotion_scores.items()
        }
    else:
        normalized = {k: 0.5 for k in emotion_scores}

    # Find predicted emotion
    predicted = max(normalized, key=normalized.get)
    confidence = normalized[predicted]

    # Adjust confidence based on match density
    match_density = total_matches / max(len(tokens), 1)
    confidence = min(1.0, confidence * (0.5 + match_density))
    confidence = round(max(0.1, confidence), 4)

    logger.debug(
        f"Emotion detected: {predicted} (conf={confidence}), "
        f"matches={total_matches}/{len(tokens)} tokens"
    )

    return EmotionResult(
        predicted_emotion=predicted,
        confidence_score=confidence,
        emotion_scores={k: round(v, 4) for k, v in normalized.items()},
        model_version=version,
        word_matches={k: v for k, v in word_matches.items() if v},
    )
