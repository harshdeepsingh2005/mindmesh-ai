"""Tests for the training pipeline — dataset generation, TF-IDF, classifier."""

import pytest

from app.services.training_pipeline import (
    generate_emotion_dataset,
    generate_sentiment_dataset,
    SimpleTfidfVectorizer,
    NearestCentroidClassifier,
    TrainingDataset,
    train_emotion_model,
    train_sentiment_model,
)


# ─── Dataset Generation ─────────────────────────────────────────


class TestDatasetGeneration:
    """Synthetic dataset generators."""

    def test_emotion_dataset_size(self):
        ds = generate_emotion_dataset(n=100, seed=0)
        assert ds.size == 100
        assert len(ds.texts) == 100
        assert len(ds.labels) == 100

    def test_emotion_dataset_labels(self):
        ds = generate_emotion_dataset(n=200, seed=0)
        unique_labels = set(ds.labels)
        # Should have all 5 emotion categories
        assert unique_labels == {"happy", "sad", "anxious", "angry", "neutral"}

    def test_sentiment_dataset_scores(self):
        ds = generate_sentiment_dataset(n=100, seed=0)
        assert len(ds.scores) == 100
        for score in ds.scores:
            assert -1.0 <= score <= 1.0

    def test_dataset_split(self):
        ds = generate_emotion_dataset(n=100, seed=0)
        train, val, test = ds.split(train_ratio=0.7, val_ratio=0.15)
        assert train.size == 70
        assert val.size == 15
        assert test.size == 15

    def test_dataset_reproducibility(self):
        ds1 = generate_emotion_dataset(n=50, seed=42)
        ds2 = generate_emotion_dataset(n=50, seed=42)
        assert ds1.texts == ds2.texts
        assert ds1.labels == ds2.labels


# ─── TF-IDF Vectorizer ──────────────────────────────────────────


class TestTfidfVectorizer:
    """SimpleTfidfVectorizer tests."""

    def test_fit_transform(self):
        docs = [
            "I feel happy and great today",
            "Feeling sad and lonely",
            "Everything is okay and normal",
        ]
        vec = SimpleTfidfVectorizer(max_features=50)
        vectors = vec.fit_transform(docs)
        assert len(vectors) == 3
        assert len(vectors[0]) == len(vec.vocabulary_)
        assert len(vec.vocabulary_) <= 50

    def test_vectors_normalized(self):
        import math

        docs = ["hello world", "foo bar baz"]
        vec = SimpleTfidfVectorizer(max_features=100)
        vectors = vec.fit_transform(docs)
        for v in vectors:
            norm = math.sqrt(sum(x ** 2 for x in v))
            assert norm == pytest.approx(1.0, abs=0.01)

    def test_transform_unseen_words(self):
        docs = ["hello world"]
        vec = SimpleTfidfVectorizer(max_features=50)
        vec.fit(docs)
        new_vectors = vec.transform(["completely unseen words"])
        assert len(new_vectors) == 1
        # All zeros since no vocab overlap
        assert all(v == 0.0 for v in new_vectors[0])

    def test_max_features_limit(self):
        docs = [f"word{i} " * 3 for i in range(100)]
        vec = SimpleTfidfVectorizer(max_features=10)
        vec.fit(docs)
        assert len(vec.vocabulary_) <= 10


# ─── Nearest-Centroid Classifier ─────────────────────────────────


class TestNearestCentroidClassifier:
    """NearestCentroidClassifier tests."""

    def test_fit_predict(self):
        # Simple 2D data with clear clusters
        X = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        y = ["A", "A", "B", "B"]
        clf = NearestCentroidClassifier()
        clf.fit(X, y)
        preds = clf.predict([[0.8, 0.2], [0.2, 0.8]])
        assert preds[0] == "A"
        assert preds[1] == "B"

    def test_classes_stored(self):
        X = [[1.0], [0.0]]
        y = ["pos", "neg"]
        clf = NearestCentroidClassifier()
        clf.fit(X, y)
        assert set(clf.classes) == {"pos", "neg"}


# ─── End-to-End Training ────────────────────────────────────────


class TestTrainingPipeline:
    """Async training pipeline tests."""

    @pytest.mark.asyncio
    async def test_train_emotion_model(self):
        result = await train_emotion_model(
            dataset=generate_emotion_dataset(n=100, seed=0),
            version="test-0.1",
        )
        assert result.model_name == "emotion_detection"
        assert result.new_version == "test-0.1"
        assert result.training_samples > 0
        assert result.test_samples > 0
        assert "accuracy" in result.metrics

    @pytest.mark.asyncio
    async def test_train_sentiment_model(self):
        result = await train_sentiment_model(
            dataset=generate_sentiment_dataset(n=100, seed=0),
            version="test-0.1",
        )
        assert result.model_name == "sentiment_analysis"
        assert result.new_version == "test-0.1"
        assert "mae" in result.metrics
