"""MindMesh AI — Model Management API Routes.

Endpoints for:
  • Viewing model registry (all models, versions, active status)
  • Triggering training pipelines
  • Running model evaluations on datasets
  • Comparing model versions
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
from ..services.model_evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_risk_scoring,
    ClassificationMetrics,
    RegressionMetrics,
)
from ..services.training_pipeline import (
    run_full_training_pipeline,
    train_emotion_model,
    train_sentiment_model,
    generate_emotion_dataset,
    generate_sentiment_dataset,
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
    metrics: Dict[str, float] = {}
    description: str = ""
    created_at: str
    activated_at: Optional[str] = None
    retired_at: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelVersionResponse]
    total: int


class TrainingRequest(BaseModel):
    model_name: Optional[str] = Field(
        None, description="Train a specific model (emotion_detection, sentiment_analysis) or all"
    )
    version: str = Field("2.0.0", description="Version to assign to the new model")
    dataset_size: int = Field(500, ge=50, le=5000, description="Synthetic dataset size")
    auto_promote: bool = Field(False, description="Auto-promote if better than active")


class TrainingResultResponse(BaseModel):
    model_name: str
    new_version: str
    training_samples: int
    test_samples: int
    metrics: Dict[str, float]
    promoted: bool
    duration_seconds: float
    comparison: Optional[Dict[str, Any]] = None


class TrainingPipelineResponse(BaseModel):
    results: List[TrainingResultResponse]
    total_models: int


class EvaluationRequest(BaseModel):
    model_name: str = Field(..., description="Model to evaluate")
    model_version: str = Field(..., description="Version to evaluate")
    dataset_size: int = Field(200, ge=50, le=5000)


class EvaluationResponse(BaseModel):
    model_name: str
    model_version: str
    total_samples: int
    metrics: Dict[str, Any]


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
    summary="Trigger model training pipeline",
)
async def trigger_training(
    body: TrainingRequest,
    current_user: User = Depends(require_roles(["admin"])),
) -> TrainingPipelineResponse:
    """Trigger training for one or all models.

    Generates a synthetic dataset, trains the model, evaluates it,
    and optionally promotes it if it outperforms the active version.

    Admin only.
    """
    results: List[TrainingResult] = []

    if body.model_name is None or body.model_name == "all":
        results = await run_full_training_pipeline(
            auto_promote=body.auto_promote,
        )
    elif body.model_name == "emotion_detection":
        dataset = generate_emotion_dataset(n=body.dataset_size)
        result = await train_emotion_model(
            dataset=dataset,
            version=body.version,
            auto_promote=body.auto_promote,
        )
        results.append(result)
    elif body.model_name == "sentiment_analysis":
        dataset = generate_sentiment_dataset(n=body.dataset_size)
        result = await train_sentiment_model(
            dataset=dataset,
            version=body.version,
            auto_promote=body.auto_promote,
        )
        results.append(result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {body.model_name}. "
                   f"Use 'emotion_detection', 'sentiment_analysis', or 'all'.",
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
                test_samples=r.test_samples,
                metrics=r.metrics,
                promoted=r.promoted,
                duration_seconds=r.duration_seconds,
                comparison={
                    "version_a": r.comparison.version_a,
                    "version_b": r.comparison.version_b,
                    "metric_deltas": r.comparison.metric_deltas,
                    "winner": r.comparison.winner,
                } if r.comparison else None,
            )
            for r in results
        ],
        total_models=len(results),
    )


# ─── Evaluation Endpoints ───────────────────────────────────────


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate a model version on synthetic data",
)
async def evaluate_model(
    body: EvaluationRequest,
    current_user: User = Depends(require_roles(["admin"])),
) -> EvaluationResponse:
    """Run evaluation metrics on a model against a synthetic dataset.

    Admin only. Useful for benchmarking without training.
    """
    model = registry.get(body.model_name, body.model_version)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {body.model_name} v{body.model_version} not found",
        )

    if body.model_name == "emotion_detection":
        dataset = generate_emotion_dataset(n=body.dataset_size)
        _, _, test_ds = dataset.split()

        from ..services.emotion_detection import detect_emotion
        y_pred = [detect_emotion(t).predicted_emotion for t in test_ds.texts]

        metrics = evaluate_classification(
            test_ds.labels, y_pred,
            model_name=body.model_name,
            model_version=body.model_version,
        )

        # Update registry metrics
        registry.update_metrics(body.model_name, body.model_version, {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
        })

        return EvaluationResponse(
            model_name=body.model_name,
            model_version=body.model_version,
            total_samples=metrics.total_samples,
            metrics={
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "macro_precision": metrics.macro_precision,
                "macro_recall": metrics.macro_recall,
                "per_class": metrics.per_class,
            },
        )

    elif body.model_name == "sentiment_analysis":
        dataset = generate_sentiment_dataset(n=body.dataset_size)
        _, _, test_ds = dataset.split()

        from ..services.sentiment_analysis import analyze_sentiment
        y_pred = [analyze_sentiment(t).sentiment_score for t in test_ds.texts]

        metrics = evaluate_regression(
            test_ds.scores, y_pred,
            model_name=body.model_name,
            model_version=body.model_version,
        )

        registry.update_metrics(body.model_name, body.model_version, {
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "pearson_correlation": metrics.pearson_correlation,
        })

        return EvaluationResponse(
            model_name=body.model_name,
            model_version=body.model_version,
            total_samples=metrics.total_samples,
            metrics={
                "mae": metrics.mae,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r_squared": metrics.r_squared,
                "pearson_correlation": metrics.pearson_correlation,
            },
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation not supported for model: {body.model_name}",
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
            detail="Both models must have evaluation metrics. Run /evaluate first.",
        )

    # Build deltas
    all_keys = set(model_a.metrics.keys()) | set(model_b.metrics.keys())
    deltas = {}
    for key in all_keys:
        val_a = model_a.metrics.get(key, 0.0)
        val_b = model_b.metrics.get(key, 0.0)
        deltas[key] = round(val_b - val_a, 4)

    # Determine winner based on primary metric
    primary = "weighted_f1" if "weighted_f1" in all_keys else "mae"
    if primary == "mae":
        # Lower is better for MAE
        winner = version_a if model_a.metrics.get("mae", 1) < model_b.metrics.get("mae", 1) else version_b
    else:
        winner = version_b if deltas.get(primary, 0) > 0.005 else (
            version_a if deltas.get(primary, 0) < -0.005 else "tie"
        )

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
    """Promote a candidate model version to active.

    The currently active version will be retired.
    Admin only.
    """
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
