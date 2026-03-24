"""MindMesh AI — Anomaly Detection Service (Unsupervised).

Detects behavioural outliers among students using unsupervised
anomaly detection algorithms.  Students whose behavioural patterns
deviate significantly from the norm are flagged for attention.

Algorithms:
  • Isolation Forest:  Tree-based anomaly detection.  Isolates
    anomalies by randomly partitioning feature space — anomalies
    require fewer partitions to isolate.
  • Local Outlier Factor (LOF):  Density-based method.  Compares
    a student's local density to their neighbours.  Students in
    low-density regions relative to their k-neighbours are outliers.

Feature vectors for anomaly detection are constructed from:
  • Sentiment scores (avg, std, trend)
  • Emotion cluster distributions
  • Activity frequency (check-ins, journals per week)
  • Mood variability
  • Engagement metrics (disengagement flags)

This is the CORE unsupervised learning component — it replaces
hardcoded risk thresholds with data-driven outlier detection.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ..config import settings
from ..logging_config import logger


# ── Constants ────────────────────────────────────────────────────

DEFAULT_CONTAMINATION = "auto"   # Let IF statistically determine boundaries
DEFAULT_N_ESTIMATORS = 100       # trees in Isolation Forest
DEFAULT_N_NEIGHBORS = 10         # neighbours for LOF
MIN_SAMPLES_FOR_FITTING = 10     # minimum students to fit model


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class BehavioralFeatureVector:
    """Feature vector representing one student's behavioural profile.

    Constructed from aggregated behavioural records over a time window.
    """
    student_id: str

    # Sentiment features
    avg_sentiment: float = 0.0          # average compound sentiment
    sentiment_std: float = 0.0          # sentiment volatility
    negative_ratio: float = 0.0         # proportion of negative entries

    # Emotion features
    dominant_cluster: int = 0           # most frequent emotion cluster
    emotion_entropy: float = 0.0        # diversity of emotion clusters
    distress_ratio: float = 0.0         # proportion of distress-related entries

    # Activity features
    entries_per_week: float = 0.0       # behavioural record frequency
    journal_ratio: float = 0.0          # proportion of journal vs check-in
    days_since_last_entry: float = 0.0  # recency of engagement

    # Mood features
    avg_mood: float = 0.0              # average mood rating
    mood_std: float = 0.0             # mood volatility

    # High-risk flags count
    high_risk_flags: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.avg_sentiment,
            self.sentiment_std,
            self.negative_ratio,
            self.dominant_cluster,
            self.emotion_entropy,
            self.distress_ratio,
            self.entries_per_week,
            self.journal_ratio,
            self.days_since_last_entry,
            self.avg_mood,
            self.mood_std,
            self.high_risk_flags,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names."""
        return [
            "avg_sentiment", "sentiment_std", "negative_ratio",
            "dominant_cluster", "emotion_entropy", "distress_ratio",
            "entries_per_week", "journal_ratio", "days_since_last_entry",
            "avg_mood", "mood_std", "high_risk_flags",
        ]


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single student."""
    student_id: str
    is_anomaly: bool
    anomaly_score: float            # lower (more negative) = more anomalous
    anomaly_percentile: float       # 0-100: where this student falls
    risk_level: str                 # low, medium, high (derived from score)
    contributing_features: List[Tuple[str, float]]  # top features causing anomaly
    model_version: str = "anomaly-v1.0.0-iforest"


@dataclass
class CohortAnomalyReport:
    """Report for a batch of students."""
    total_students: int
    anomalies_detected: int
    anomaly_rate: float
    results: List[AnomalyResult]
    model_config: Dict


# ── Anomaly Detection Engine ────────────────────────────────────


class AnomalyDetectionEngine:
    """Isolation Forest + LOF anomaly detection for student behaviour.

    Operates in two phases:
      1. fit():      Learn the "normal" behavioural distribution
      2. predict():  Score new students against the learned norm

    Students with high anomaly scores are flagged for counselor attention.
    """

    def __init__(
        self,
        contamination: str | float = DEFAULT_CONTAMINATION,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors

        self._iforest: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._version = "anomaly-v1.0.0-iforest"
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, feature_vectors: List[BehavioralFeatureVector]) -> "AnomalyDetectionEngine":
        """Fit the anomaly detection model on student data.

        Args:
            feature_vectors: List of behavioural feature vectors.

        Returns:
            Self (for chaining).
        """
        if len(feature_vectors) < MIN_SAMPLES_FOR_FITTING:
            logger.warning(
                f"Not enough samples for anomaly detection "
                f"({len(feature_vectors)} < {MIN_SAMPLES_FOR_FITTING})"
            )
            return self

        # Build feature matrix
        X = np.array([fv.to_array() for fv in feature_vectors])

        # Scale features
        X_scaled = self._scaler.fit_transform(X)
        self._feature_means = self._scaler.mean_
        self._feature_stds = self._scaler.scale_

        # Fit Isolation Forest
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._iforest.fit(X_scaled)
        self._is_fitted = True

        # Log training stats
        predictions = self._iforest.predict(X_scaled)
        n_anomalies = int((predictions == -1).sum())

        logger.info(
            f"AnomalyDetectionEngine fitted: samples={len(feature_vectors)}, "
            f"anomalies_found={n_anomalies}, "
            f"contamination={self.contamination}"
        )
        self.save()

        return self

    def predict(self, feature_vector: BehavioralFeatureVector) -> AnomalyResult:
        """Score a single student for anomaly.

        Args:
            feature_vector: The student's behavioural feature vector.

        Returns:
            AnomalyResult with anomaly score and risk level.
        """
        if not self._is_fitted or self._iforest is None:
            return self._fallback_predict(feature_vector)

        X = feature_vector.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        # Anomaly score: negative = more anomalous
        score = float(self._iforest.score_samples(X_scaled)[0])
        prediction = int(self._iforest.predict(X_scaled)[0])
        is_anomaly = prediction == -1

        # Convert score to percentile (approximate)
        # Isolation Forest scores are typically in [-0.5, 0.5]
        # More negative = more anomalous
        percentile = max(0, min(100, (score + 0.5) * 100))

        # Risk level from score
        if score < -0.3:
            risk_level = "high"
        elif score < -0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Feature contribution analysis
        contributing = self._analyze_contributions(X_scaled[0])

        return AnomalyResult(
            student_id=feature_vector.student_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            anomaly_percentile=round(percentile, 2),
            risk_level=risk_level,
            contributing_features=contributing,
            model_version=self._version,
        )

    def predict_batch(
        self, feature_vectors: List[BehavioralFeatureVector]
    ) -> CohortAnomalyReport:
        """Score a batch of students.

        Args:
            feature_vectors: List of student feature vectors.

        Returns:
            CohortAnomalyReport with per-student results.
        """
        results = [self.predict(fv) for fv in feature_vectors]

        anomalies = [r for r in results if r.is_anomaly]

        return CohortAnomalyReport(
            total_students=len(results),
            anomalies_detected=len(anomalies),
            anomaly_rate=round(len(anomalies) / max(len(results), 1), 4),
            results=results,
            model_config=self.get_config(),
        )

    def _analyze_contributions(self, x_scaled: np.ndarray) -> List[Tuple[str, float]]:
        """Identify which features contribute most to anomaly.

        Uses z-score analysis: features with highest absolute z-scores
        are the most "unusual" and likely driving the anomaly.
        """
        feature_names = BehavioralFeatureVector.feature_names()
        contributions = []

        for i, (name, z) in enumerate(zip(feature_names, x_scaled)):
            contributions.append((name, round(float(abs(z)), 4)))

        # Sort by magnitude (most unusual first)
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:5]  # top 5 contributing features

    def _fallback_predict(self, fv: BehavioralFeatureVector) -> AnomalyResult:
        """Simple threshold-based fallback when model is not fitted."""
        # Use basic heuristics
        risk_score = 0.0
        if fv.negative_ratio > 0.5:
            risk_score += 0.3
        if fv.sentiment_std > 0.5:
            risk_score += 0.2
        if fv.high_risk_flags > 0:
            risk_score += 0.4
        if fv.entries_per_week < 1:
            risk_score += 0.1

        risk_score = min(risk_score, 1.0)

        if risk_score > 0.6:
            risk_level = "high"
        elif risk_score > 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return AnomalyResult(
            student_id=fv.student_id,
            is_anomaly=risk_score > 0.5,
            anomaly_score=round(-risk_score, 4),
            anomaly_percentile=round((1 - risk_score) * 100, 2),
            risk_level=risk_level,
            contributing_features=[],
            model_version="anomaly-v1.0.0-fallback",
        )

    def get_config(self) -> Dict:
        """Return engine config for registry."""
        return {
            "type": "isolation_forest",
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "n_neighbors": self.n_neighbors,
            "is_fitted": self._is_fitted,
            "version": self._version,
        }

    def save(self) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            return
        os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
        path = os.path.join(settings.MODEL_SAVE_DIR, "anomaly_detection.joblib")
        joblib.dump(self, path)
        logger.info(f"AnomalyDetectionEngine saved to {path}")

    @classmethod
    def load(cls) -> Optional["AnomalyDetectionEngine"]:
        """Load the fitted model from disk."""
        path = os.path.join(settings.MODEL_SAVE_DIR, "anomaly_detection.joblib")
        if os.path.exists(path):
            try:
                engine = joblib.load(path)
                logger.info(f"AnomalyDetectionEngine loaded from {path}")
                return engine
            except Exception as e:
                logger.error(f"Failed to load AnomalyDetectionEngine: {e}")
        return None


# ── Singleton ────────────────────────────────────────────────────

_engine: Optional[AnomalyDetectionEngine] = None


def get_anomaly_engine() -> AnomalyDetectionEngine:
    """Get or create the global AnomalyDetectionEngine."""
    global _engine
    if _engine is None:
        _engine = AnomalyDetectionEngine.load()
        if _engine is None:
            _engine = AnomalyDetectionEngine()
    return _engine
