"""MindMesh AI — Prediction Quality Metrics.

Computes a complete metrics bundle for standalone text analysis:
  • Confidence Score
  • Agent Agreement / Consensus
  • Bias Detection Score
  • Fact Reliability Score
  • Reasoning Completeness (structural)
  • Latency
  • Cost efficiency (implicit)

These are lightweight runtime heuristics designed for observability
and triage, not clinical or legal decision-making.
"""

from __future__ import annotations

import re
from typing import Any, Dict


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _compute_consensus(emotion: Dict[str, Any], sentiment: Dict[str, Any], topic: Dict[str, Any]) -> Dict[str, Any]:
    """Compute agreement score across emotion/sentiment/topic agents."""
    emotion_label = (emotion.get("cluster_label") or "").lower()
    sentiment_label = (sentiment.get("sentiment_label") or "").lower()
    topic_label = (topic.get("topic_label") or "").lower()

    emotion_risky = any(t in emotion_label for t in ["distress", "anxiety", "anger"])
    sentiment_risky = sentiment_label == "negative" or bool(sentiment.get("high_risk_flag", False))
    topic_risky = any(t in topic_label for t in [
        "anxiety", "stress", "depress", "die", "suicide", "hurt", "help",
    ])

    votes = {
        "emotion": int(emotion_risky),
        "sentiment": int(sentiment_risky),
        "topic": int(topic_risky),
    }
    risky_votes = sum(votes.values())
    safe_votes = 3 - risky_votes
    agreement = max(risky_votes, safe_votes) / 3.0

    if risky_votes >= 2:
        label = "risk_consensus"
    elif safe_votes >= 2:
        label = "stable_consensus"
    else:
        label = "mixed"

    return {
        "score": round(agreement, 4),
        "label": label,
        "votes": votes,
        "risky_votes": risky_votes,
    }


def _compute_bias_detection_score(text: str, sentiment: Dict[str, Any]) -> float:
    """Heuristic bias-detection quality score (higher is better)."""
    protected_terms = {
        "he", "she", "him", "her", "man", "woman", "boy", "girl",
        "black", "white", "asian", "muslim", "hindu", "christian",
        "gay", "lesbian", "trans", "disabled", "autistic",
    }

    words = re.findall(r"[a-zA-Z']+", text.lower())
    protected_hits = sum(1 for w in words if w in protected_terms)

    sentiment_magnitude = abs(float(sentiment.get("sentiment_score", 0.0)))
    high_risk = 1.0 if sentiment.get("high_risk_flag", False) else 0.0

    bias_risk = _clamp((protected_hits * 0.18) + (sentiment_magnitude * 0.35) + (high_risk * 0.2))
    return round(1.0 - bias_risk, 4)


def _compute_fact_reliability_score(text: str) -> float:
    """Estimate factual reliability from linguistic cues (higher is better)."""
    t = text.lower()

    uncertainty_markers = [
        "maybe", "perhaps", "probably", "not sure", "i think", "i guess", "might",
    ]
    grounding_markers = [
        "because", "since", "today", "yesterday", "when", "after", "before", "at",
    ]

    uncertainty = sum(1 for m in uncertainty_markers if m in t)
    grounding = sum(1 for m in grounding_markers if m in t)
    has_number = bool(re.search(r"\d", t))

    score = 0.55 + (grounding * 0.08) - (uncertainty * 0.12) + (0.05 if has_number else 0.0)
    return round(_clamp(score), 4)


def _compute_reasoning_completeness(text: str) -> Dict[str, Any]:
    """Check structural completeness across common reasoning components."""
    t = text.lower()

    components = {
        "context": any(k in t for k in ["today", "at school", "in class", "this week", "recently"]),
        "cause": any(k in t for k in ["because", "since", "due to", "after", "when"]),
        "feeling": any(k in t for k in ["feel", "feeling", "anxious", "sad", "angry", "stressed"]),
        "impact": any(k in t for k in ["can't", "cannot", "unable", "affect", "hard", "difficult"]),
        "next_step": any(k in t for k in ["need help", "talk", "plan", "will", "going to", "support"]),
    }

    present = sum(1 for v in components.values() if v)
    score = present / len(components)
    return {
        "score": round(score, 4),
        "components": components,
        "present_components": present,
        "total_components": len(components),
    }


def compute_prediction_metrics(
    text: str,
    emotion: Dict[str, Any],
    sentiment: Dict[str, Any],
    topic: Dict[str, Any],
    latency_seconds: float,
) -> Dict[str, Any]:
    """Return a complete metrics payload for standalone prediction."""
    confidence = float(emotion.get("confidence_score", 0.0) or 0.0)
    consensus = _compute_consensus(emotion, sentiment, topic)
    bias_score = _compute_bias_detection_score(text, sentiment)
    fact_reliability = _compute_fact_reliability_score(text)
    reasoning = _compute_reasoning_completeness(text)

    # Local model inference with no per-token API calls.
    latency_norm = _clamp(1.0 - (latency_seconds / 0.5))
    cost_efficiency_score = round((0.7 * latency_norm) + 0.3, 4)

    return {
        "confidence_score": round(_clamp(confidence), 4),
        "agent_agreement_consensus": consensus,
        "bias_detection_score": bias_score,
        "fact_reliability_score": fact_reliability,
        "reasoning_completeness_structural": reasoning,
        "latency_seconds": round(float(latency_seconds), 6),
        "latency_ms": round(float(latency_seconds) * 1000.0, 3),
        "cost_efficiency": {
            "score": cost_efficiency_score,
            "label": "high" if cost_efficiency_score >= 0.8 else ("medium" if cost_efficiency_score >= 0.5 else "low"),
            "mode": "local_inference_no_token_billing",
        },
    }
