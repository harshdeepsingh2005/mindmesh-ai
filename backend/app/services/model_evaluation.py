"""MindMesh AI — Model Evaluation Metrics.

Provides standardised evaluation metrics for all AI models:
  • Emotion detection accuracy, precision, recall, F1
  • Sentiment analysis correlation and error metrics
  • Risk scoring calibration and discrimination metrics
  • Per-class breakdowns and confusion matrices
  • Model comparison utilities

All metrics follow scikit-learn conventions where possible.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import logger


# ─── Data Classes ────────────────────────────────────────────────


@dataclass
class ClassificationMetrics:
    """Metrics for a single classification model evaluation run."""

    model_name: str
    model_version: str
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_samples: int = 0

    # Overall metrics
    accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Confusion matrix: {true_label: {predicted_label: count}}
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Class labels in order
    labels: List[str] = field(default_factory=list)


@dataclass
class RegressionMetrics:
    """Metrics for regression / continuous-score evaluation."""

    model_name: str
    model_version: str
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_samples: int = 0

    mae: float = 0.0  # Mean Absolute Error
    mse: float = 0.0  # Mean Squared Error
    rmse: float = 0.0  # Root Mean Squared Error
    r_squared: float = 0.0  # Coefficient of determination
    pearson_correlation: float = 0.0

    # Calibration: predicted vs actual in buckets
    calibration_buckets: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class RiskCalibrationMetrics:
    """Specialised metrics for the risk scoring model."""

    model_name: str = "risk_scoring"
    model_version: str = "1.0.0"
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_samples: int = 0

    # Level classification
    level_accuracy: float = 0.0
    level_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    level_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Score regression
    score_mae: float = 0.0
    score_rmse: float = 0.0
    score_correlation: float = 0.0

    # Discrimination: can the model separate high from low risk?
    auc_high_vs_rest: Optional[float] = None
    separation_score: float = 0.0  # mean(high) - mean(low) / pooled_std


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


# ─── Classification Evaluation ───────────────────────────────────


def evaluate_classification(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "emotion_detection",
    model_version: str = "1.0.0",
) -> ClassificationMetrics:
    """Compute full classification metrics from predictions.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name of the model.
        model_version: Version string.

    Returns:
        ClassificationMetrics with all computed values.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must be same length"
    n = len(y_true)

    if n == 0:
        return ClassificationMetrics(
            model_name=model_name, model_version=model_version
        )

    labels = sorted(set(y_true) | set(y_pred))

    # Build confusion matrix
    confusion: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # Per-class precision, recall, F1
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in labels if t != label)
        fn = sum(confusion[label][p] for p in labels if p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = tp + fn

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro averages
    num_classes = len(labels)
    macro_precision = sum(pc["precision"] for pc in per_class.values()) / num_classes
    macro_recall = sum(pc["recall"] for pc in per_class.values()) / num_classes
    macro_f1 = sum(pc["f1"] for pc in per_class.values()) / num_classes

    # Weighted F1
    total_support = sum(pc["support"] for pc in per_class.values())
    weighted_f1 = (
        sum(pc["f1"] * pc["support"] for pc in per_class.values()) / total_support
        if total_support > 0
        else 0.0
    )

    metrics = ClassificationMetrics(
        model_name=model_name,
        model_version=model_version,
        total_samples=n,
        accuracy=round(accuracy, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        confusion_matrix=confusion,
        labels=labels,
    )

    logger.info(
        f"Classification eval: model={model_name} v{model_version}, "
        f"n={n}, accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}"
    )

    return metrics


# ─── Regression Evaluation ───────────────────────────────────────


def evaluate_regression(
    y_true: List[float],
    y_pred: List[float],
    model_name: str = "sentiment_analysis",
    model_version: str = "1.0.0",
    num_buckets: int = 10,
) -> RegressionMetrics:
    """Compute regression metrics for continuous predictions.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        model_name: Name of the model.
        model_version: Version string.
        num_buckets: Number of calibration buckets.

    Returns:
        RegressionMetrics with all computed values.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must be same length"
    n = len(y_true)

    if n == 0:
        return RegressionMetrics(
            model_name=model_name, model_version=model_version
        )

    # MAE, MSE, RMSE
    errors = [t - p for t, p in zip(y_true, y_pred)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e ** 2 for e in errors]

    mae = sum(abs_errors) / n
    mse = sum(sq_errors) / n
    rmse = math.sqrt(mse)

    # R²
    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pearson correlation
    pearson = _pearson_correlation(y_true, y_pred)

    # Calibration buckets
    calibration_buckets = _compute_calibration(y_true, y_pred, num_buckets)

    metrics = RegressionMetrics(
        model_name=model_name,
        model_version=model_version,
        total_samples=n,
        mae=round(mae, 4),
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        r_squared=round(r_squared, 4),
        pearson_correlation=round(pearson, 4),
        calibration_buckets=calibration_buckets,
    )

    logger.info(
        f"Regression eval: model={model_name} v{model_version}, "
        f"n={n}, MAE={mae:.4f}, R²={r_squared:.4f}"
    )

    return metrics


# ─── Risk Scoring Evaluation ────────────────────────────────────


def evaluate_risk_scoring(
    true_scores: List[int],
    pred_scores: List[int],
    true_levels: List[str],
    pred_levels: List[str],
    model_version: str = "1.0.0",
) -> RiskCalibrationMetrics:
    """Evaluate the risk scoring pipeline end-to-end.

    Combines classification metrics (risk level) with regression
    metrics (risk score) and discrimination analysis.

    Args:
        true_scores: Ground-truth risk scores (0-100).
        pred_scores: Predicted risk scores (0-100).
        true_levels: Ground-truth risk levels (low/medium/high).
        pred_levels: Predicted risk levels.
        model_version: Version string.

    Returns:
        RiskCalibrationMetrics.
    """
    n = len(true_scores)

    # Level classification
    level_eval = evaluate_classification(
        true_levels, pred_levels,
        model_name="risk_scoring_level",
        model_version=model_version,
    )

    # Score regression
    score_eval = evaluate_regression(
        [float(s) for s in true_scores],
        [float(s) for s in pred_scores],
        model_name="risk_scoring_score",
        model_version=model_version,
    )

    # Discrimination: separation between high and low risk
    high_scores = [s for s, l in zip(pred_scores, true_levels) if l == "high"]
    low_scores = [s for s, l in zip(pred_scores, true_levels) if l == "low"]

    separation = 0.0
    if high_scores and low_scores:
        mean_high = sum(high_scores) / len(high_scores)
        mean_low = sum(low_scores) / len(low_scores)
        std_high = _std(high_scores)
        std_low = _std(low_scores)
        pooled_std = math.sqrt(
            (std_high ** 2 + std_low ** 2) / 2
        ) if (std_high + std_low) > 0 else 1.0
        separation = (mean_high - mean_low) / pooled_std

    # AUC for high vs rest (binary)
    auc = _binary_auc(
        [1 if l == "high" else 0 for l in true_levels],
        pred_scores,
    )

    metrics = RiskCalibrationMetrics(
        model_version=model_version,
        total_samples=n,
        level_accuracy=level_eval.accuracy,
        level_metrics=level_eval.per_class,
        level_confusion=level_eval.confusion_matrix,
        score_mae=score_eval.mae,
        score_rmse=score_eval.rmse,
        score_correlation=score_eval.pearson_correlation,
        auc_high_vs_rest=round(auc, 4) if auc is not None else None,
        separation_score=round(separation, 4),
    )

    logger.info(
        f"Risk scoring eval: v{model_version}, n={n}, "
        f"level_acc={level_eval.accuracy:.4f}, "
        f"score_MAE={score_eval.mae:.4f}, AUC={auc}"
    )

    return metrics


# ─── Model Comparison ────────────────────────────────────────────


def compare_models(
    metrics_a: ClassificationMetrics,
    metrics_b: ClassificationMetrics,
) -> ModelComparison:
    """Compare two model evaluation results.

    Args:
        metrics_a: First model evaluation (baseline).
        metrics_b: Second model evaluation (candidate).

    Returns:
        ModelComparison with deltas and winner.
    """
    deltas = {
        "accuracy": round(metrics_b.accuracy - metrics_a.accuracy, 4),
        "macro_f1": round(metrics_b.macro_f1 - metrics_a.macro_f1, 4),
        "weighted_f1": round(metrics_b.weighted_f1 - metrics_a.weighted_f1, 4),
        "macro_precision": round(
            metrics_b.macro_precision - metrics_a.macro_precision, 4
        ),
        "macro_recall": round(
            metrics_b.macro_recall - metrics_a.macro_recall, 4
        ),
    }

    # Winner is whichever has higher weighted F1
    if deltas["weighted_f1"] > 0.005:
        winner = metrics_b.model_version
    elif deltas["weighted_f1"] < -0.005:
        winner = metrics_a.model_version
    else:
        winner = "tie"

    comparison = ModelComparison(
        model_name=metrics_a.model_name,
        version_a=metrics_a.model_version,
        version_b=metrics_b.model_version,
        metric_deltas=deltas,
        winner=winner,
    )

    logger.info(
        f"Model comparison: {metrics_a.model_version} vs {metrics_b.model_version}, "
        f"winner={winner}, Δf1={deltas['weighted_f1']}"
    )

    return comparison


# ─── Helper Functions ────────────────────────────────────────────


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def _std(values: List[float]) -> float:
    """Standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _compute_calibration(
    y_true: List[float],
    y_pred: List[float],
    num_buckets: int = 10,
) -> List[Dict[str, float]]:
    """Compute calibration buckets (predicted vs actual averages)."""
    if not y_pred:
        return []

    min_pred = min(y_pred)
    max_pred = max(y_pred)
    bucket_range = (max_pred - min_pred) / num_buckets if max_pred > min_pred else 1.0

    buckets: List[Dict[str, float]] = []
    for i in range(num_buckets):
        lo = min_pred + i * bucket_range
        hi = lo + bucket_range

        pairs = [
            (t, p)
            for t, p in zip(y_true, y_pred)
            if lo <= p < hi or (i == num_buckets - 1 and p == hi)
        ]

        if pairs:
            avg_true = sum(t for t, _ in pairs) / len(pairs)
            avg_pred = sum(p for _, p in pairs) / len(pairs)
            buckets.append({
                "bucket": i,
                "range_low": round(lo, 3),
                "range_high": round(hi, 3),
                "avg_predicted": round(avg_pred, 3),
                "avg_actual": round(avg_true, 3),
                "count": len(pairs),
            })

    return buckets


def _binary_auc(
    y_true_binary: List[int],
    y_scores: List[int],
) -> Optional[float]:
    """Compute AUC-ROC for binary classification using trapezoidal rule.

    Args:
        y_true_binary: Binary labels (0 or 1).
        y_scores: Continuous scores (higher = more positive).

    Returns:
        AUC value, or None if undefined.
    """
    n_pos = sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None

    # Sort by score descending
    pairs = sorted(
        zip(y_scores, y_true_binary), key=lambda x: -x[0]
    )

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg

        # Trapezoidal rule
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

        tpr_prev = tpr
        fpr_prev = fpr

    return auc
