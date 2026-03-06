"""MindMesh AI — Model Registry & Versioning.

Manages model metadata, version tracking, and deployment state.
Each model (emotion detection, sentiment analysis, risk scoring)
has a registry entry with:
  • Version string (semver)
  • Configuration / hyperparameters
  • Evaluation metrics from the last benchmark
  • Status (active, candidate, retired)
  • Timestamps

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
    metrics: Dict[str, float] = field(default_factory=dict)
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
        """Register the built-in model versions."""
        # Emotion detection
        self.register(
            ModelVersion(
                model_name="emotion_detection",
                version="1.0.0",
                status="active",
                config={
                    "type": "keyword_rule_based",
                    "emotions": ["happy", "sad", "anxious", "angry", "neutral"],
                    "keyword_lists": 5,
                    "fallback": "neutral",
                },
                metrics={
                    "accuracy": 0.62,
                    "macro_f1": 0.58,
                    "weighted_f1": 0.60,
                },
                description="Rule-based keyword emotion classifier v1",
                activated_at=datetime.now(timezone.utc),
            )
        )

        self.register(
            ModelVersion(
                model_name="emotion_detection",
                version="2.0.0",
                status="candidate",
                config={
                    "type": "tfidf_svm",
                    "max_features": 5000,
                    "kernel": "linear",
                    "emotions": ["happy", "sad", "anxious", "angry", "neutral"],
                },
                metrics={},
                description="TF-IDF + SVM emotion classifier (candidate)",
            )
        )

        # Sentiment analysis
        self.register(
            ModelVersion(
                model_name="sentiment_analysis",
                version="1.0.0",
                status="active",
                config={
                    "type": "lexicon_based",
                    "positive_words": 45,
                    "negative_words": 55,
                    "high_risk_keywords": 20,
                },
                metrics={
                    "mae": 0.18,
                    "pearson_correlation": 0.71,
                },
                description="Lexicon-based sentiment analyser v1",
                activated_at=datetime.now(timezone.utc),
            )
        )

        # Risk scoring
        self.register(
            ModelVersion(
                model_name="risk_scoring",
                version="1.0.0",
                status="active",
                config={
                    "type": "weighted_factor_composite",
                    "num_factors": 7,
                    "threshold_low": 40,
                    "threshold_high": 70,
                    "lookback_days": 30,
                },
                metrics={
                    "level_accuracy": 0.75,
                    "score_mae": 8.5,
                    "separation_score": 2.1,
                },
                description="7-factor weighted composite risk scorer v1",
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
        """Register a new model version.

        If the version already exists, it is overwritten.
        """
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
        metrics: Dict[str, float],
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
