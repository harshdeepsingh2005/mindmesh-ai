"""MindMesh AI — Model Registry & Versioning (Unsupervised).

Manages model metadata, version tracking, and deployment state
for all unsupervised ML models:

  • text_embeddings     — TF-IDF vectoriser
  • emotion_detection   — K-Means emotion clustering
  • sentiment_analysis  — VADER sentiment analyser
  • anomaly_detection   — Isolation Forest
  • student_clustering  — Gaussian Mixture Model
  • topic_discovery     — NMF topic modelling
  • risk_scoring        — Composite (anomaly + sentiment)

The registry is stored in-memory for the prototype but designed
to be backed by a database table in production.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..logging_config import logger


@dataclass
class ModelVersion:
    """A single registered model version."""

    model_name: str
    version: str
    status: str = "candidate"  # candidate | active | retired
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    activated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self.status,
            "config": self.config,
            "metrics": self.metrics,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
        }


class ModelRegistry:
    """In-memory model registry.

    Thread-safe for single-process FastAPI (async single-thread).
    For multi-worker deployments, back with Redis or database.
    """

    def __init__(self) -> None:
        # {model_name: {version: ModelVersion}}
        self._registry: Dict[str, Dict[str, ModelVersion]] = {}
        self._active: Dict[str, str] = {}  # {model_name: active_version}
        self._initialise_defaults()

    def _initialise_defaults(self) -> None:
        """Register the built-in unsupervised model versions."""

        # Text Embeddings (TF-IDF)
        self.register(
            ModelVersion(
                model_name="text_embeddings",
                version="3.0.0",
                status="active",
                config={
                    "type": "tfidf",
                    "max_features": 5000,
                    "ngram_range": [1, 2],
                    "sublinear_tf": True,
                },
                metrics={"vocabulary_size": 0},
                description="TF-IDF text embedding engine v3 (unfitted, awaiting data)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Emotion Detection (K-Means Clustering)
        self.register(
            ModelVersion(
                model_name="emotion_detection",
                version="3.0.0",
                status="active",
                config={
                    "type": "kmeans_clustering",
                    "n_clusters": 5,
                    "init": "k-means++",
                },
                metrics={
                    "silhouette_score": 0.0,
                    "n_clusters": 5,
                },
                description="K-Means emotion cluster discovery v3 (unfitted, awaiting data)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Sentiment Analysis (VADER)
        self.register(
            ModelVersion(
                model_name="sentiment_analysis",
                version="3.0.0",
                status="active",
                config={
                    "type": "vader",
                    "library": "nltk",
                    "compound_threshold_pos": 0.05,
                    "compound_threshold_neg": -0.05,
                },
                metrics={
                    "type": "rule_based_unsupervised",
                    "description": "VADER requires no training",
                },
                description="VADER sentiment analyser v3 (no training required)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Anomaly Detection (Isolation Forest)
        self.register(
            ModelVersion(
                model_name="anomaly_detection",
                version="1.0.0",
                status="active",
                config={
                    "type": "isolation_forest",
                    "contamination": 0.1,
                    "n_estimators": 100,
                },
                metrics={
                    "anomaly_rate": 0.0,
                },
                description="Isolation Forest anomaly detection v1 (unfitted, awaiting data)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Student Clustering (GMM)
        self.register(
            ModelVersion(
                model_name="student_clustering",
                version="1.0.0",
                status="active",
                config={
                    "type": "gmm",
                    "covariance_type": "full",
                    "auto_select_k": True,
                },
                metrics={
                    "silhouette_score": 0.0,
                    "n_clusters": 0,
                },
                description="GMM student clustering v1 (unfitted, awaiting data)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Topic Discovery (NMF)
        self.register(
            ModelVersion(
                model_name="topic_discovery",
                version="1.0.0",
                status="active",
                config={
                    "type": "nmf",
                    "n_topics": 8,
                    "init": "nndsvd",
                },
                metrics={
                    "reconstruction_error": 0.0,
                    "n_topics": 0,
                },
                description="NMF topic discovery v1 (unfitted, awaiting data)",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Risk Scoring (Composite: Anomaly + Sentiment)
        self.register(
            ModelVersion(
                model_name="risk_scoring",
                version="2.0.0",
                status="active",
                config={
                    "type": "anomaly_composite",
                    "primary_signal": "isolation_forest",
                    "secondary_signals": ["vader_sentiment", "emotion_clusters"],
                    "threshold_low": 40,
                    "threshold_high": 70,
                    "lookback_days": 30,
                },
                metrics={
                    "type": "unsupervised_composite",
                    "anomaly_weight": 0.30,
                    "sentiment_weight": 0.20,
                },
                description="Anomaly-based composite risk scorer v2",
                activated_at=datetime.now(timezone.utc),
            )
        )

        logger.info(
            f"Model registry initialised: "
            f"{sum(len(v) for v in self._registry.values())} versions across "
            f"{len(self._registry)} models"
        )

    # ─── Registration ────────────────────────────────────────

    def register(self, model: ModelVersion) -> ModelVersion:
        """Register a new model version."""
        if model.model_name not in self._registry:
            self._registry[model.model_name] = {}

        self._registry[model.model_name][model.version] = model

        if model.status == "active":
            self._active[model.model_name] = model.version

        logger.info(
            f"Model registered: {model.model_name} v{model.version} "
            f"(status={model.status})"
        )

        return model

    # ─── Activation / Retirement ─────────────────────────────

    def activate(self, model_name: str, version: str) -> ModelVersion:
        """Promote a model version to active and retire the previous one."""
        model = self.get(model_name, version)
        if model is None:
            raise ValueError(
                f"Model {model_name} v{version} not found in registry"
            )

        # Retire current active
        current_active = self._active.get(model_name)
        if current_active and current_active != version:
            old = self._registry[model_name][current_active]
            old.status = "retired"
            old.retired_at = datetime.now(timezone.utc)
            logger.info(f"Model retired: {model_name} v{current_active}")

        model.status = "active"
        model.activated_at = datetime.now(timezone.utc)
        self._active[model_name] = version

        logger.info(f"Model activated: {model_name} v{version}")
        return model

    def retire(self, model_name: str, version: str) -> ModelVersion:
        """Retire a model version."""
        model = self.get(model_name, version)
        if model is None:
            raise ValueError(
                f"Model {model_name} v{version} not found in registry"
            )

        model.status = "retired"
        model.retired_at = datetime.now(timezone.utc)

        if self._active.get(model_name) == version:
            del self._active[model_name]

        logger.info(f"Model retired: {model_name} v{version}")
        return model

    # ─── Queries ─────────────────────────────────────────────

    def get(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._registry.get(model_name, {}).get(version)

    def get_active(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version of a model."""
        active_version = self._active.get(model_name)
        if active_version:
            return self._registry[model_name][active_version]
        return None

    def get_active_version_string(self, model_name: str) -> str:
        """Get the active version string, or 'unknown'."""
        return self._active.get(model_name, "unknown")

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._registry.keys())

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions for a model, newest first."""
        versions = list(self._registry.get(model_name, {}).values())
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def list_all(self) -> List[ModelVersion]:
        """List all model versions across all models."""
        all_versions = []
        for versions in self._registry.values():
            all_versions.extend(versions.values())
        return sorted(all_versions, key=lambda v: v.created_at, reverse=True)

    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, Any],
    ) -> ModelVersion:
        """Update stored metrics for a model version."""
        model = self.get(model_name, version)
        if model is None:
            raise ValueError(
                f"Model {model_name} v{version} not found"
            )
        model.metrics.update(metrics)
        logger.info(
            f"Metrics updated: {model_name} v{version} → {metrics}"
        )
        return model


# ── Singleton instance ───────────────────────────────────────────

registry = ModelRegistry()
