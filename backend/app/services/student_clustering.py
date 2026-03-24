"""MindMesh AI — Student Clustering Service (Unsupervised).

Groups students into behavioural profiles using K-Means and
Gaussian Mixture Models (GMM) on their aggregated behavioural
feature vectors.

Use cases:
  • Group students with similar behavioural patterns for cohort analysis
  • Identify at-risk clusters that share common distress signals
  • Track cluster drift over time (students moving between groups)
  • Provide context to counselors: "This student is in the
    'disengaged-anxious' cluster, along with 12 other students"

Unlike static risk categories (low/medium/high), clustering
discovers natural groupings in the data — the number and nature
of groups emerge from the students themselves.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..config import settings
from ..logging_config import logger
from .anomaly_detection import BehavioralFeatureVector


# ── Constants ────────────────────────────────────────────────────

DEFAULT_N_CLUSTERS = 4
MIN_STUDENTS_FOR_CLUSTERING = 8
MAX_CLUSTERS_TO_TRY = 10


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class StudentCluster:
    """Description of a discovered student behavioural cluster."""
    cluster_id: int
    label: str                           # auto-generated label
    size: int                            # number of students
    centroid_features: Dict[str, float]  # average feature values
    risk_profile: str                    # low_risk, moderate, elevated, high_risk
    key_characteristics: List[str]       # human-readable descriptions


@dataclass
class StudentClusterAssignment:
    """Cluster assignment for one student."""
    student_id: str
    cluster_id: int
    cluster_label: str
    membership_probabilities: Dict[int, float]  # GMM soft assignments
    confidence: float
    cluster_risk_profile: str


@dataclass
class ClusteringReport:
    """Full clustering analysis report."""
    n_clusters: int
    clusters: List[StudentCluster]
    silhouette_score: float
    calinski_harabasz_score: float
    total_students: int
    assignments: List[StudentClusterAssignment]
    pca_2d: Optional[List[Dict]] = None  # 2D PCA coords for visualisation
    model_version: str = "clustering-v1.0.0-gmm"


# ── Student Clustering Engine ───────────────────────────────────


class StudentClusterEngine:
    """GMM-based student behavioural clustering.

    Uses Gaussian Mixture Models for soft clustering — each student
    gets a probability distribution over clusters, not a hard
    assignment.  This is more nuanced than K-Means and better
    reflects the reality that students can exhibit mixed behaviours.
    """

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS) -> None:
        self.n_clusters = n_clusters
        self._gmm: Optional[GaussianMixture] = None
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=2)
        self._clusters: List[StudentCluster] = []
        self._is_fitted = False
        self._version = "clustering-v1.0.0-gmm"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        feature_vectors: List[BehavioralFeatureVector],
        auto_select_k: bool = True,
    ) -> "StudentClusterEngine":
        """Fit clustering model on student behavioural data.

        Args:
            feature_vectors: List of student feature vectors.
            auto_select_k: If True, automatically select optimal K
                          using BIC (Bayesian Information Criterion).

        Returns:
            Self (for chaining).
        """
        if len(feature_vectors) < MIN_STUDENTS_FOR_CLUSTERING:
            logger.warning(
                f"Not enough students for clustering "
                f"({len(feature_vectors)} < {MIN_STUDENTS_FOR_CLUSTERING})"
            )
            return self

        # Build feature matrix
        X = np.array([fv.to_array() for fv in feature_vectors])
        X_scaled = self._scaler.fit_transform(X)

        # Auto-select K using BIC
        if auto_select_k:
            actual_k = self._select_optimal_k(X_scaled)
        else:
            actual_k = min(self.n_clusters, len(feature_vectors) // 2)
            actual_k = max(2, actual_k)

        # Fit GMM
        self._gmm = GaussianMixture(
            n_components=actual_k,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        self._gmm.fit(X_scaled)

        # Fit PCA for 2D visualisation
        if X_scaled.shape[1] > 2:
            self._pca.fit(X_scaled)

        # Compute cluster descriptions
        labels = self._gmm.predict(X_scaled)
        self._clusters = self._describe_clusters(
            X_scaled, labels, feature_vectors
        )

        self._is_fitted = True

        # Compute quality metrics
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
        else:
            sil = 0.0
            ch = 0.0

        logger.info(
            f"StudentClusterEngine fitted: k={actual_k}, "
            f"silhouette={sil:.4f}, students={len(feature_vectors)}"
        )
        self.save()

        return self

    def predict(
        self, feature_vector: BehavioralFeatureVector
    ) -> StudentClusterAssignment:
        """Assign a student to a cluster with probabilities.

        Args:
            feature_vector: Student's behavioural feature vector.

        Returns:
            StudentClusterAssignment with soft membership.
        """
        if not self._is_fitted or self._gmm is None:
            return StudentClusterAssignment(
                student_id=feature_vector.student_id,
                cluster_id=0,
                cluster_label="unclassified",
                membership_probabilities={},
                confidence=0.0,
                cluster_risk_profile="unknown",
            )

        X = feature_vector.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        # GMM soft assignment
        probas = self._gmm.predict_proba(X_scaled)[0]
        cluster_id = int(np.argmax(probas))
        confidence = float(probas[cluster_id])

        membership = {i: round(float(p), 4) for i, p in enumerate(probas)}

        cluster_info = (
            self._clusters[cluster_id]
            if cluster_id < len(self._clusters)
            else None
        )
        label = cluster_info.label if cluster_info else f"cluster_{cluster_id}"
        risk_profile = cluster_info.risk_profile if cluster_info else "unknown"

        return StudentClusterAssignment(
            student_id=feature_vector.student_id,
            cluster_id=cluster_id,
            cluster_label=label,
            membership_probabilities=membership,
            confidence=round(confidence, 4),
            cluster_risk_profile=risk_profile,
        )

    def generate_report(
        self,
        feature_vectors: List[BehavioralFeatureVector],
    ) -> ClusteringReport:
        """Generate a full clustering report for a cohort.

        Args:
            feature_vectors: List of student feature vectors.

        Returns:
            ClusteringReport with all assignments and metrics.
        """
        if not self._is_fitted:
            self.fit(feature_vectors)

        assignments = [self.predict(fv) for fv in feature_vectors]

        # Compute metrics
        X = np.array([fv.to_array() for fv in feature_vectors])
        X_scaled = self._scaler.transform(X)
        labels = [a.cluster_id for a in assignments]

        if len(set(labels)) > 1 and len(labels) > 2:
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
        else:
            sil = 0.0
            ch = 0.0

        # 2D PCA for visualisation
        pca_2d = None
        if X_scaled.shape[1] > 2:
            coords = self._pca.transform(X_scaled)
            pca_2d = [
                {
                    "student_id": fv.student_id,
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                    "cluster": assignments[i].cluster_id,
                }
                for i, fv in enumerate(feature_vectors)
            ]

        return ClusteringReport(
            n_clusters=len(self._clusters),
            clusters=self._clusters,
            silhouette_score=round(sil, 4),
            calinski_harabasz_score=round(ch, 4),
            total_students=len(feature_vectors),
            assignments=assignments,
            pca_2d=pca_2d,
            model_version=self._version,
        )

    def _select_optimal_k(self, X: np.ndarray) -> int:
        """Select optimal number of clusters using BIC.

        Lower BIC = better model.  Tests K from 2 to max_k.
        """
        max_k = min(MAX_CLUSTERS_TO_TRY, len(X) // 3)
        max_k = max(2, max_k)

        best_k = 2
        best_bic = float("inf")

        for k in range(2, max_k + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=3,
                random_state=42,
            )
            gmm.fit(X)
            bic = gmm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_k = k

        logger.info(f"Optimal K selected: {best_k} (BIC={best_bic:.2f})")
        return best_k

    def _describe_clusters(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        feature_vectors: List[BehavioralFeatureVector],
    ) -> List[StudentCluster]:
        """Generate human-readable cluster descriptions."""
        feature_names = BehavioralFeatureVector.feature_names()
        clusters = []

        for cluster_id in range(self._gmm.n_components):
            mask = labels == cluster_id
            cluster_data = X_scaled[mask]
            size = int(mask.sum())

            if size == 0:
                continue

            # Centroid (mean of cluster members)
            centroid = cluster_data.mean(axis=0)
            centroid_dict = {
                name: round(float(val), 4)
                for name, val in zip(feature_names, centroid)
            }

            # Determine risk profile from features
            risk_profile = self._assess_cluster_risk(centroid_dict)

            # Generate label and characteristics
            label, characteristics = self._generate_label(centroid_dict)

            clusters.append(StudentCluster(
                cluster_id=cluster_id,
                label=label,
                size=size,
                centroid_features=centroid_dict,
                risk_profile=risk_profile,
                key_characteristics=characteristics,
            ))

        return clusters

    def _assess_cluster_risk(self, centroid: Dict[str, float]) -> str:
        """Determine risk profile of a cluster from its centroid."""
        risk_signals = 0

        if centroid.get("avg_sentiment", 0) < -0.5:
            risk_signals += 2
        elif centroid.get("avg_sentiment", 0) < 0:
            risk_signals += 1

        if centroid.get("negative_ratio", 0) > 0.5:
            risk_signals += 2
        elif centroid.get("negative_ratio", 0) > 0.3:
            risk_signals += 1

        if centroid.get("distress_ratio", 0) > 0.3:
            risk_signals += 2

        if centroid.get("high_risk_flags", 0) > 0.5:
            risk_signals += 3

        if centroid.get("entries_per_week", 0) < -0.5:
            risk_signals += 1  # disengagement

        if risk_signals >= 5:
            return "high_risk"
        elif risk_signals >= 3:
            return "elevated"
        elif risk_signals >= 1:
            return "moderate"
        return "low_risk"

    def _generate_label(self, centroid: Dict[str, float]) -> Tuple[str, List[str]]:
        """Generate a human-readable label for a cluster."""
        characteristics = []
        label_parts = []

        # Sentiment
        avg_sent = centroid.get("avg_sentiment", 0)
        if avg_sent > 0.5:
            label_parts.append("positive")
            characteristics.append("Generally positive sentiment")
        elif avg_sent < -0.5:
            label_parts.append("distressed")
            characteristics.append("Consistently negative sentiment")
        elif avg_sent < 0:
            label_parts.append("mixed")
            characteristics.append("Mixed/slightly negative sentiment")
        else:
            label_parts.append("balanced")

        # Activity
        activity = centroid.get("entries_per_week", 0)
        if activity > 0.5:
            label_parts.append("active")
            characteristics.append("High engagement with platform")
        elif activity < -0.5:
            label_parts.append("disengaged")
            characteristics.append("Low platform engagement")

        # Volatility
        mood_std = centroid.get("mood_std", 0)
        if mood_std > 0.5:
            characteristics.append("High mood variability")
        elif mood_std < -0.3:
            characteristics.append("Stable mood patterns")

        # High risk
        if centroid.get("high_risk_flags", 0) > 0:
            characteristics.append("Contains high-risk keyword flags")

        label = "_".join(label_parts) if label_parts else "general"
        return label, characteristics

    def get_config(self) -> Dict:
        return {
            "type": "gmm_clustering",
            "n_clusters": self.n_clusters,
            "n_clusters_actual": len(self._clusters),
            "is_fitted": self._is_fitted,
            "version": self._version,
        }

    def save(self) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            return
        os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
        path = os.path.join(settings.MODEL_SAVE_DIR, "student_clustering.joblib")
        joblib.dump(self, path)
        logger.info(f"StudentClusterEngine saved to {path}")

    @classmethod
    def load(cls) -> Optional["StudentClusterEngine"]:
        """Load the fitted model from disk."""
        path = os.path.join(settings.MODEL_SAVE_DIR, "student_clustering.joblib")
        if os.path.exists(path):
            try:
                engine = joblib.load(path)
                logger.info(f"StudentClusterEngine loaded from {path}")
                return engine
            except Exception as e:
                logger.error(f"Failed to load StudentClusterEngine: {e}")
        return None


# ── Singleton ────────────────────────────────────────────────────

_engine: Optional[StudentClusterEngine] = None


def get_cluster_engine() -> StudentClusterEngine:
    """Get or create the global StudentClusterEngine."""
    global _engine
    if _engine is None:
        _engine = StudentClusterEngine.load()
        if _engine is None:
            _engine = StudentClusterEngine()
    return _engine
