"""Performance tests — latency benchmarks for AI pipelines and endpoints."""

import time

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


# ─── AI Pipeline Latency ────────────────────────────────────────


class TestAIPipelinePerformance:
    """Ensure AI analysis completes within acceptable time bounds."""

    def test_emotion_detection_latency(self):
        from app.services.emotion_detection import detect_emotion

        start = time.perf_counter()
        for _ in range(100):
            detect_emotion(
                "I am feeling very anxious and worried about my upcoming exams"
            )
        elapsed = (time.perf_counter() - start) / 100
        # Each call should take < 10ms
        assert elapsed < 0.01, f"Emotion detection too slow: {elapsed:.4f}s avg"

    def test_sentiment_analysis_latency(self):
        from app.services.sentiment_analysis import analyze_sentiment

        start = time.perf_counter()
        for _ in range(100):
            analyze_sentiment(
                "This is a wonderful and amazing experience at school today"
            )
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.01, f"Sentiment analysis too slow: {elapsed:.4f}s avg"

    def test_evaluation_metrics_latency(self):
        from app.services.model_evaluation import evaluate_classification

        y_true = ["happy", "sad", "angry", "neutral", "anxious"] * 200
        y_pred = ["happy", "sad", "happy", "neutral", "anxious"] * 200
        start = time.perf_counter()
        evaluate_classification(y_true, y_pred)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Classification eval too slow: {elapsed:.4f}s"

    def test_tfidf_vectorizer_latency(self):
        from app.services.training_pipeline import SimpleTfidfVectorizer

        docs = [f"document number {i} with some words in it" for i in range(500)]
        vec = SimpleTfidfVectorizer(max_features=1000)
        start = time.perf_counter()
        vec.fit_transform(docs)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"TF-IDF fit_transform too slow: {elapsed:.4f}s"

    def test_classifier_latency(self):
        from app.services.training_pipeline import (
            SimpleTfidfVectorizer,
            NearestCentroidClassifier,
        )

        docs = [f"sample text number {i}" for i in range(200)]
        labels = ["a", "b", "c", "d", "e"] * 40
        vec = SimpleTfidfVectorizer(max_features=500)
        X = vec.fit_transform(docs)
        clf = NearestCentroidClassifier()
        start = time.perf_counter()
        clf.fit(X, labels)
        clf.predict(X)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Classifier too slow: {elapsed:.4f}s"


# ─── Endpoint Latency ───────────────────────────────────────────


class TestEndpointPerformance:
    """Endpoint response time benchmarks."""

    async def test_health_check_latency(self, client: AsyncClient):
        start = time.perf_counter()
        resp = await client.get("/health")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 1.0, f"Health check too slow: {elapsed:.4f}s"

    async def test_root_endpoint_latency(self, client: AsyncClient):
        start = time.perf_counter()
        resp = await client.get("/")
        elapsed = time.perf_counter() - start
        assert resp.status_code == 200
        assert elapsed < 0.5, f"Root endpoint too slow: {elapsed:.4f}s"

    async def test_concurrent_requests(self, client: AsyncClient, admin_user):
        """Multiple sequential requests should maintain performance."""
        import asyncio

        headers = auth_header(admin_user)

        async def _request():
            return await client.get("/analytics/overview", headers=headers)

        start = time.perf_counter()
        results = []
        for _ in range(10):
            results.append(await _request())
        elapsed = time.perf_counter() - start

        for r in results:
            assert r.status_code == 200
        assert elapsed < 5.0, f"10 sequential requests too slow: {elapsed:.4f}s"
