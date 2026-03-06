"""Tests for model evaluation metrics module."""

import pytest

from app.services.model_evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_risk_scoring,
    compare_models,
    ClassificationMetrics,
    RegressionMetrics,
    RiskCalibrationMetrics,
    ModelComparison,
)


# ─── Classification Evaluation ───────────────────────────────────


class TestClassificationEvaluation:
    """evaluate_classification() tests."""

    def test_perfect_predictions(self):
        labels = ["happy", "sad", "angry", "happy", "sad"]
        metrics = evaluate_classification(labels, labels)
        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0

    def test_all_wrong(self):
        y_true = ["happy", "happy", "happy"]
        y_pred = ["sad", "sad", "sad"]
        metrics = evaluate_classification(y_true, y_pred)
        assert metrics.accuracy == 0.0

    def test_mixed_predictions(self):
        y_true = ["happy", "sad", "angry", "happy", "sad", "angry"]
        y_pred = ["happy", "sad", "happy", "happy", "angry", "angry"]
        metrics = evaluate_classification(y_true, y_pred)
        assert 0 < metrics.accuracy < 1
        assert 0 <= metrics.macro_f1 <= 1

    def test_empty_input(self):
        metrics = evaluate_classification([], [])
        assert metrics.total_samples == 0
        assert metrics.accuracy == 0.0

    def test_confusion_matrix_structure(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "a", "b", "b"]
        metrics = evaluate_classification(y_true, y_pred)
        assert "a" in metrics.confusion_matrix
        assert "b" in metrics.confusion_matrix
        # Diagonal should have 1 correct each
        assert metrics.confusion_matrix["a"]["a"] == 1
        assert metrics.confusion_matrix["b"]["b"] == 1

    def test_per_class_support(self):
        y_true = ["a", "a", "a", "b"]
        y_pred = ["a", "a", "a", "b"]
        metrics = evaluate_classification(y_true, y_pred)
        assert metrics.per_class["a"]["support"] == 3
        assert metrics.per_class["b"]["support"] == 1


# ─── Regression Evaluation ───────────────────────────────────────


class TestRegressionEvaluation:
    """evaluate_regression() tests."""

    def test_perfect_predictions(self):
        values = [0.1, 0.5, -0.3, 0.8, -0.7]
        metrics = evaluate_regression(values, values)
        assert metrics.mae == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r_squared == pytest.approx(1.0, abs=0.01)

    def test_imperfect_predictions(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.2, 2.8, 4.3, 4.7]
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics.mae > 0
        assert metrics.rmse >= metrics.mae  # RMSE >= MAE always
        assert 0 <= metrics.r_squared <= 1

    def test_empty_input(self):
        metrics = evaluate_regression([], [])
        assert metrics.total_samples == 0
        assert metrics.mae == 0.0

    def test_calibration_buckets(self):
        y_true = [float(i) for i in range(20)]
        y_pred = [float(i) + 0.5 for i in range(20)]
        metrics = evaluate_regression(y_true, y_pred, num_buckets=5)
        assert len(metrics.calibration_buckets) > 0

    def test_pearson_correlation(self):
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = evaluate_regression(y_true, y_pred)
        assert metrics.pearson_correlation == pytest.approx(1.0, abs=0.01)


# ─── Risk Scoring Evaluation ────────────────────────────────────


class TestRiskScoringEvaluation:
    """evaluate_risk_scoring() tests."""

    def test_basic_risk_eval(self):
        true_scores = [20, 50, 80, 30, 70]
        pred_scores = [25, 45, 85, 35, 65]
        true_levels = ["low", "medium", "high", "low", "high"]
        pred_levels = ["low", "medium", "high", "low", "medium"]
        metrics = evaluate_risk_scoring(
            true_scores, pred_scores, true_levels, pred_levels
        )
        assert isinstance(metrics, RiskCalibrationMetrics)
        assert metrics.total_samples == 5
        assert metrics.score_mae >= 0
        assert metrics.level_accuracy >= 0


# ─── Model Comparison ───────────────────────────────────────────


class TestModelComparison:
    """compare_models() tests."""

    def test_clear_winner(self):
        a = ClassificationMetrics(
            model_name="emotion",
            model_version="1.0",
            accuracy=0.6,
            macro_f1=0.55,
            weighted_f1=0.58,
            macro_precision=0.6,
            macro_recall=0.55,
        )
        b = ClassificationMetrics(
            model_name="emotion",
            model_version="2.0",
            accuracy=0.8,
            macro_f1=0.75,
            weighted_f1=0.78,
            macro_precision=0.8,
            macro_recall=0.75,
        )
        comparison = compare_models(a, b)
        assert comparison.winner == "2.0"
        assert comparison.metric_deltas["accuracy"] > 0

    def test_tie(self):
        m = ClassificationMetrics(
            model_name="emotion",
            model_version="1.0",
            accuracy=0.7,
            macro_f1=0.65,
            weighted_f1=0.68,
            macro_precision=0.7,
            macro_recall=0.65,
        )
        comparison = compare_models(m, m)
        assert comparison.winner == "tie"


# ─── Model Registry ─────────────────────────────────────────────


class TestModelRegistry:
    """Tests for the in-memory model registry."""

    def test_registry_has_defaults(self):
        from app.services.model_registry import registry

        models = registry.list_models()
        assert "emotion_detection" in models
        assert "sentiment_analysis" in models
        assert "risk_scoring" in models

    def test_get_active(self):
        from app.services.model_registry import registry

        active = registry.get_active("emotion_detection")
        assert active is not None
        assert active.status == "active"

    def test_list_versions(self):
        from app.services.model_registry import registry

        versions = registry.list_versions("emotion_detection")
        assert len(versions) >= 2  # v1.0.0 and v2.0.0

    def test_register_new_version(self):
        from app.services.model_registry import ModelVersion, ModelRegistry

        reg = ModelRegistry()
        v = ModelVersion(
            model_name="test_model",
            version="0.1.0",
            status="candidate",
        )
        result = reg.register(v)
        assert result.model_name == "test_model"
        assert reg.get("test_model", "0.1.0") is not None

    def test_activate_version(self):
        from app.services.model_registry import ModelVersion, ModelRegistry

        reg = ModelRegistry()
        v = ModelVersion(
            model_name="test_model",
            version="0.1.0",
            status="candidate",
        )
        reg.register(v)
        activated = reg.activate("test_model", "0.1.0")
        assert activated.status == "active"
        assert reg.get_active("test_model").version == "0.1.0"

    def test_activate_nonexistent_raises(self):
        from app.services.model_registry import ModelRegistry

        reg = ModelRegistry()
        with pytest.raises(ValueError):
            reg.activate("nonexistent", "1.0.0")

    def test_retire_version(self):
        from app.services.model_registry import ModelVersion, ModelRegistry

        reg = ModelRegistry()
        v = ModelVersion(
            model_name="test_retire",
            version="1.0.0",
            status="active",
        )
        reg.register(v)
        retired = reg.retire("test_retire", "1.0.0")
        assert retired.status == "retired"
        assert retired.retired_at is not None

    def test_update_metrics(self):
        from app.services.model_registry import ModelVersion, ModelRegistry

        reg = ModelRegistry()
        v = ModelVersion(model_name="test_m", version="1.0.0")
        reg.register(v)
        reg.update_metrics("test_m", "1.0.0", {"accuracy": 0.9})
        updated = reg.get("test_m", "1.0.0")
        assert updated.metrics["accuracy"] == 0.9
