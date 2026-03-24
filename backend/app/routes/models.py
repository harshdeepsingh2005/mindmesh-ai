"""MindMesh AI — Model Management API Routes (Unsupervised).

Endpoints for:
  • Viewing model registry (all models, versions, active status)
  • Triggering unsupervised training pipelines
  • Viewing unsupervised model metrics
  • Promoting / retiring model versions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..models.user import User
from ..dependencies import require_roles
from ..services.model_registry import registry, ModelVersion
from ..services.training_pipeline import (
    run_full_training_pipeline,
    train_text_embeddings,
    train_emotion_clusters,
    train_anomaly_detection,
    train_student_clustering,
    train_topic_discovery,
    generate_synthetic_corpus,
    generate_synthetic_features,
    TrainingResult,
)
from ..logging_config import logger

router = APIRouter()


# ─── Schemas ─────────────────────────────────────────────────────


class ModelVersionResponse(BaseModel):
    model_name: str
    version: str
    status: str
    config: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    description: str = ""
    created_at: str
    activated_at: Optional[str] = None
    retired_at: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelVersionResponse]
    total: int


class TrainingRequest(BaseModel):
    model_name: Optional[str] = Field(
        None,
        description="Model to train: text_embeddings, emotion_detection, "
                    "anomaly_detection, student_clustering, topic_discovery, or all",
    )
    corpus_size: int = Field(500, ge=50, le=5000, description="Synthetic corpus size")
    feature_size: int = Field(100, ge=20, le=1000, description="Synthetic feature count")


class TrainingResultResponse(BaseModel):
    model_name: str
    new_version: str
    training_samples: int
    metrics: Dict[str, Any]
    promoted: bool
    duration_seconds: float


class TrainingPipelineResponse(BaseModel):
    results: List[TrainingResultResponse]
    total_models: int


class PromoteRequest(BaseModel):
    model_name: str
    version: str


class ComparisonResponse(BaseModel):
    model_name: str
    version_a: str
    version_b: str
    metric_deltas: Dict[str, float]
    winner: str


# ─── Model Registry Endpoints ────────────────────────────────────


@router.get(
    "/",
    response_model=ModelListResponse,
    summary="List all registered models",
)
async def list_models(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    current_user: User = Depends(require_roles(["admin"])),
) -> ModelListResponse:
    """List all registered model versions with status and metrics."""
    if model_name:
        versions = registry.list_versions(model_name)
    else:
        versions = registry.list_all()

    return ModelListResponse(
        models=[
            ModelVersionResponse(
                model_name=v.model_name,
                version=v.version,
                status=v.status,
                config=v.config,
                metrics=v.metrics,
                description=v.description,
                created_at=v.created_at.isoformat(),
                activated_at=v.activated_at.isoformat() if v.activated_at else None,
                retired_at=v.retired_at.isoformat() if v.retired_at else None,
            )
            for v in versions
        ],
        total=len(versions),
    )


@router.get(
    "/active",
    summary="List active model versions",
)
async def list_active_models(
    current_user: User = Depends(require_roles(["admin", "teacher"])),
) -> Dict[str, Any]:
    """Get the currently active version of each model."""
    model_names = registry.list_models()
    active = {}
    for name in model_names:
        model = registry.get_active(name)
        if model:
            active[name] = model.to_dict()
        else:
            active[name] = None

    return {"active_models": active}


@router.get(
    "/{model_name}/{version}",
    response_model=ModelVersionResponse,
    summary="Get a specific model version",
)
async def get_model_version(
    model_name: str,
    version: str,
    current_user: User = Depends(require_roles(["admin"])),
) -> ModelVersionResponse:
    """Get details of a specific model version."""
    model = registry.get(model_name, version)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} v{version} not found",
        )

    return ModelVersionResponse(
        model_name=model.model_name,
        version=model.version,
        status=model.status,
        config=model.config,
        metrics=model.metrics,
        description=model.description,
        created_at=model.created_at.isoformat(),
        activated_at=model.activated_at.isoformat() if model.activated_at else None,
        retired_at=model.retired_at.isoformat() if model.retired_at else None,
    )


# ─── Training Endpoints ─────────────────────────────────────────


@router.post(
    "/train",
    response_model=TrainingPipelineResponse,
    summary="Trigger unsupervised model training pipeline",
)
async def trigger_training(
    body: TrainingRequest,
    current_user: User = Depends(require_roles(["admin"])),
) -> TrainingPipelineResponse:
    """Trigger training for one or all unsupervised models.

    Generates synthetic data (if no real data available), fits the
    model, and registers it in the model registry.

    Admin only.
    """
    results: List[TrainingResult] = []

    valid_models = {
        "text_embeddings", "emotion_detection", "anomaly_detection",
        "student_clustering", "topic_discovery",
    }

    if body.model_name is None or body.model_name == "all":
        corpus = generate_synthetic_corpus(n=body.corpus_size)
        features = generate_synthetic_features(n=body.feature_size)
        results = await run_full_training_pipeline(
            corpus=corpus, features=features
        )
    elif body.model_name == "text_embeddings":
        corpus = generate_synthetic_corpus(n=body.corpus_size)
        result = await train_text_embeddings(corpus=corpus)
        results.append(result)
    elif body.model_name == "emotion_detection":
        corpus = generate_synthetic_corpus(n=body.corpus_size)
        result = await train_emotion_clusters(corpus=corpus)
        results.append(result)
    elif body.model_name == "anomaly_detection":
        features = generate_synthetic_features(n=body.feature_size)
        result = await train_anomaly_detection(features=features)
        results.append(result)
    elif body.model_name == "student_clustering":
        features = generate_synthetic_features(n=body.feature_size)
        result = await train_student_clustering(features=features)
        results.append(result)
    elif body.model_name == "topic_discovery":
        corpus = generate_synthetic_corpus(n=body.corpus_size)
        result = await train_topic_discovery(corpus=corpus)
        results.append(result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {body.model_name}. "
                   f"Valid options: {', '.join(valid_models)} or 'all'.",
        )

    logger.info(
        f"Training triggered by user={current_user.id}: "
        f"{len(results)} models trained"
    )

    return TrainingPipelineResponse(
        results=[
            TrainingResultResponse(
                model_name=r.model_name,
                new_version=r.new_version,
                training_samples=r.training_samples,
                metrics=r.metrics,
                promoted=r.promoted,
                duration_seconds=r.duration_seconds,
            )
            for r in results
        ],
        total_models=len(results),
    )


# ─── Comparison Endpoint ────────────────────────────────────────


@router.get(
    "/compare/{model_name}",
    response_model=ComparisonResponse,
    summary="Compare two model versions",
)
async def compare_model_versions(
    model_name: str,
    version_a: str = Query(..., description="Baseline version"),
    version_b: str = Query(..., description="Candidate version"),
    current_user: User = Depends(require_roles(["admin"])),
) -> ComparisonResponse:
    """Compare metrics between two versions of the same model."""
    model_a = registry.get(model_name, version_a)
    model_b = registry.get(model_name, version_b)

    if model_a is None:
        raise HTTPException(404, detail=f"{model_name} v{version_a} not found")
    if model_b is None:
        raise HTTPException(404, detail=f"{model_name} v{version_b} not found")

    if not model_a.metrics or not model_b.metrics:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both models must have metrics. Run /train first.",
        )

    # Build deltas
    all_keys = set(model_a.metrics.keys()) | set(model_b.metrics.keys())
    # Only compare numeric values
    deltas = {}
    for key in all_keys:
        val_a = model_a.metrics.get(key, 0.0)
        val_b = model_b.metrics.get(key, 0.0)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            deltas[key] = round(val_b - val_a, 4)

    # Winner based on silhouette or anomaly rate
    if "silhouette_score" in deltas:
        delta = deltas["silhouette_score"]
        winner = version_b if delta > 0.01 else (version_a if delta < -0.01 else "tie")
    else:
        winner = "tie"

    return ComparisonResponse(
        model_name=model_name,
        version_a=version_a,
        version_b=version_b,
        metric_deltas=deltas,
        winner=winner,
    )


# ─── Promotion / Retirement ─────────────────────────────────────


@router.post(
    "/promote",
    response_model=ModelVersionResponse,
    summary="Promote a model version to active",
)
async def promote_model(
    body: PromoteRequest,
    current_user: User = Depends(require_roles(["admin"])),
) -> ModelVersionResponse:
    """Promote a candidate model version to active. Admin only."""
    try:
        model = registry.activate(body.model_name, body.version)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    logger.info(
        f"Model promoted by user={current_user.id}: "
        f"{body.model_name} v{body.version}"
    )

    return ModelVersionResponse(
        model_name=model.model_name,
        version=model.version,
        status=model.status,
        config=model.config,
        metrics=model.metrics,
        description=model.description,
        created_at=model.created_at.isoformat(),
        activated_at=model.activated_at.isoformat() if model.activated_at else None,
        retired_at=model.retired_at.isoformat() if model.retired_at else None,
    )


@router.post(
    "/retire",
    response_model=ModelVersionResponse,
    summary="Retire a model version",
)
async def retire_model(
    body: PromoteRequest,
    current_user: User = Depends(require_roles(["admin"])),
) -> ModelVersionResponse:
    """Retire a model version. Admin only."""
    try:
        model = registry.retire(body.model_name, body.version)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    logger.info(
        f"Model retired by user={current_user.id}: "
        f"{body.model_name} v{body.version}"
    )

    return ModelVersionResponse(
        model_name=model.model_name,
        version=model.version,
        status=model.status,
        config=model.config,
        metrics=model.metrics,
        description=model.description,
        created_at=model.created_at.isoformat(),
        activated_at=model.activated_at.isoformat() if model.activated_at else None,
        retired_at=model.retired_at.isoformat() if model.retired_at else None,
    )
