"""MindMesh AI — Model Evaluation Metrics (Unsupervised).

Provides evaluation metrics for unsupervised models:
  • Clustering: silhouette score, Calinski-Harabasz index,
    Davies-Bouldin index, cluster distribution
  • Anomaly detection: anomaly rate, score distribution,
    consistency metrics
  • Topic models: reconstruction error, topic coherence,
    topic prevalence distribution
  • Time-series: trend stability, drift detection

All metrics are designed for models that operate WITHOUT
labelled data — no accuracy, F1, or confusion matrices.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..logging_config import logger


# ── Data Classes ────────────────────────────────────────────────


@dataclass
class ClusteringMetrics:
    """Metrics for evaluating clustering quality (unsupervised)."""

    model_name: str
    model_version: str
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_samples: int = 0
    n_clusters: int = 0

    # Internal validation indices
    silhouette_score: float = 0.0       # [-1, 1] higher = better
    calinski_harabasz: float = 0.0      # higher = better
    davies_bouldin: float = 0.0         # lower = better

    # Cluster distribution
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    cluster_balance: float = 0.0        # entropy of cluster sizes (higher = more balanced)

    # Inertia (K-Means only)
    inertia: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": round(self.silhouette_score, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 4),
            "davies_bouldin": round(self.davies_bouldin, 4),
            "n_clusters": self.n_clusters,
            "total_samples": self.total_samples,
            "cluster_sizes": self.cluster_sizes,
            "cluster_balance": round(self.cluster_balance, 4),
            "inertia": round(self.inertia, 4),
        }


@dataclass
class AnomalyMetrics:
    """Metrics for evaluating anomaly detection (unsupervised)."""

    model_name: str
    model_version: str
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_samples: int = 0

    # Anomaly statistics
    anomaly_count: int = 0
    anomaly_rate: float = 0.0
    score_mean: float = 0.0
    score_std: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0

    # Score distribution (histogram)
    score_distribution: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "score_mean": round(self.score_mean, 4),
            "score_std": round(self.score_std, 4),
            "score_min": round(self.score_min, 4),
            "score_max": round(self.score_max, 4),
            "total_samples": self.total_samples,
        }


@dataclass
class TopicMetrics:
    """Metrics for evaluating topic models (unsupervised)."""

    model_name: str
    model_version: str
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_documents: int = 0
    n_topics: int = 0

    reconstruction_error: float = 0.0   # NMF reconstruction error
    topic_prevalence: Dict[int, float] = field(default_factory=dict)
    avg_topic_coherence: float = 0.0     # average pairwise term similarity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconstruction_error": round(self.reconstruction_error, 4),
            "n_topics": self.n_topics,
            "total_documents": self.total_documents,
            "topic_prevalence": {
                k: round(v, 4) for k, v in self.topic_prevalence.items()
            },
            "avg_topic_coherence": round(self.avg_topic_coherence, 4),
        }


@dataclass
class ModelComparison:
    """Side-by-side comparison of two model versions."""

    model_name: str
    version_a: str
    version_b: str
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    winner: str = ""
    compared_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── Clustering Evaluation ───────────────────────────────────────


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str = "clustering",
    model_version: str = "1.0.0",
    inertia: float = 0.0,
) -> ClusteringMetrics:
    """Evaluate clustering quality using internal validation indices.

    Args:
        X: Feature matrix (samples × features).
        labels: Cluster assignments.
        model_name: Name of the model.
        model_version: Version string.
        inertia: K-Means inertia (optional).

    Returns:
        ClusteringMetrics with all computed values.
    """
    n = len(labels)
    unique_labels = set(labels)
    n_clusters = len(unique_labels)

    if n < 2 or n_clusters < 2:
        return ClusteringMetrics(
            model_name=model_name,
            model_version=model_version,
            total_samples=n,
            n_clusters=n_clusters,
        )

    # Compute metrics
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    # Cluster sizes and balance
    cluster_counts = Counter(labels)
    cluster_sizes = {int(k): int(v) for k, v in cluster_counts.items()}

    # Balance (normalised entropy)
    probs = [c / n for c in cluster_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(n_clusters) if n_clusters > 1 else 1.0
    balance = entropy / max_entropy if max_entropy > 0 else 0.0

    metrics = ClusteringMetrics(
        model_name=model_name,
        model_version=model_version,
        total_samples=n,
        n_clusters=n_clusters,
        silhouette_score=round(sil, 4),
        calinski_harabasz=round(ch, 4),
        davies_bouldin=round(db, 4),
        cluster_sizes=cluster_sizes,
        cluster_balance=round(balance, 4),
        inertia=inertia,
    )

    logger.info(
        f"Clustering eval: {model_name} v{model_version}, "
        f"n={n}, k={n_clusters}, silhouette={sil:.4f}, "
        f"CH={ch:.4f}, DB={db:.4f}"
    )

    return metrics


# ── Anomaly Evaluation ──────────────────────────────────────────


def evaluate_anomaly_detection(
    scores: List[float],
    predictions: List[int],
    model_name: str = "anomaly_detection",
    model_version: str = "1.0.0",
    num_buckets: int = 10,
) -> AnomalyMetrics:
    """Evaluate anomaly detection model.

    Args:
        scores: Anomaly scores for each sample.
        predictions: Binary predictions (-1 = anomaly, 1 = normal).
        model_name: Name of the model.
        model_version: Version string.
        num_buckets: Number of histogram buckets.

    Returns:
        AnomalyMetrics with score distribution.
    """
    n = len(scores)

    if n == 0:
        return AnomalyMetrics(
            model_name=model_name,
            model_version=model_version,
        )

    scores_arr = np.array(scores)
    anomaly_count = int(sum(1 for p in predictions if p == -1))

    # Score distribution histogram
    hist, bin_edges = np.histogram(scores_arr, bins=num_buckets)
    distribution = [
        {
            "bucket": i,
            "range_low": round(float(bin_edges[i]), 4),
            "range_high": round(float(bin_edges[i + 1]), 4),
            "count": int(hist[i]),
        }
        for i in range(len(hist))
    ]

    metrics = AnomalyMetrics(
        model_name=model_name,
        model_version=model_version,
        total_samples=n,
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / n, 4),
        score_mean=round(float(scores_arr.mean()), 4),
        score_std=round(float(scores_arr.std()), 4),
        score_min=round(float(scores_arr.min()), 4),
        score_max=round(float(scores_arr.max()), 4),
        score_distribution=distribution,
    )

    logger.info(
        f"Anomaly eval: {model_name} v{model_version}, "
        f"n={n}, anomalies={anomaly_count}, "
        f"rate={anomaly_count/n:.4f}, "
        f"score μ={scores_arr.mean():.4f}"
    )

    return metrics


# ── Topic Model Evaluation ──────────────────────────────────────


def evaluate_topic_model(
    reconstruction_error: float,
    topic_prevalences: Dict[int, float],
    model_name: str = "topic_discovery",
    model_version: str = "1.0.0",
    n_documents: int = 0,
) -> TopicMetrics:
    """Evaluate topic model quality.

    Args:
        reconstruction_error: NMF reconstruction error.
        topic_prevalences: Topic → proportion mapping.
        model_name: Name of the model.
        model_version: Version string.
        n_documents: Total documents in corpus.

    Returns:
        TopicMetrics with reconstruction error and prevalence.
    """
    metrics = TopicMetrics(
        model_name=model_name,
        model_version=model_version,
        total_documents=n_documents,
        n_topics=len(topic_prevalences),
        reconstruction_error=round(reconstruction_error, 4),
        topic_prevalence=topic_prevalences,
    )

    logger.info(
        f"Topic eval: {model_name} v{model_version}, "
        f"topics={len(topic_prevalences)}, "
        f"recon_error={reconstruction_error:.4f}"
    )

    return metrics


# ── Model Comparison ────────────────────────────────────────────


def compare_clustering_models(
    metrics_a: ClusteringMetrics,
    metrics_b: ClusteringMetrics,
) -> ModelComparison:
    """Compare two clustering model evaluations.

    Higher silhouette + higher CH + lower DB = better.
    """
    deltas = {
        "silhouette_score": round(
            metrics_b.silhouette_score - metrics_a.silhouette_score, 4
        ),
        "calinski_harabasz": round(
            metrics_b.calinski_harabasz - metrics_a.calinski_harabasz, 4
        ),
        "davies_bouldin": round(
            metrics_a.davies_bouldin - metrics_b.davies_bouldin, 4
        ),  # reversed: lower is better
    }

    # Winner: primarily based on silhouette score
    if deltas["silhouette_score"] > 0.01:
        winner = metrics_b.model_version
    elif deltas["silhouette_score"] < -0.01:
        winner = metrics_a.model_version
    else:
        winner = "tie"

    return ModelComparison(
        model_name=metrics_a.model_name,
        version_a=metrics_a.model_version,
        version_b=metrics_b.model_version,
        metric_deltas=deltas,
        winner=winner,
    )


# ── Helper: Elbow Method ────────────────────────────────────────


def compute_elbow_scores(
    X: np.ndarray,
    max_k: int = 10,
) -> List[Dict[str, float]]:
    """Compute inertia for K from 2..max_k for the elbow method.

    Args:
        X: Feature matrix.
        max_k: Maximum number of clusters to try.

    Returns:
        List of {k, inertia, silhouette} dicts.
    """
    from sklearn.cluster import KMeans

    results = []
    max_k = min(max_k, len(X) - 1)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_

        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0

        results.append({
            "k": k,
            "inertia": round(float(kmeans.inertia_), 4),
            "silhouette_score": round(sil, 4),
        })

    return results
