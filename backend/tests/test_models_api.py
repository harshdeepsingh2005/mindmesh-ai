"""Tests for model management API endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


class TestModelListEndpoints:
    """GET /models/ — list registered models."""

    async def test_list_all_models(self, client: AsyncClient, admin_user):
        resp = await client.get("/models/", headers=auth_header(admin_user))
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "total" in data
        assert data["total"] >= 3  # emotion, sentiment, risk defaults

    async def test_list_filtered_by_name(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/models/?model_name=emotion_detection",
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 200
        data = resp.json()
        for m in data["models"]:
            assert m["model_name"] == "emotion_detection"

    async def test_teacher_cannot_list_models(
        self, client: AsyncClient, teacher_user
    ):
        resp = await client.get("/models/", headers=auth_header(teacher_user))
        assert resp.status_code == 403


class TestActiveModels:
    """GET /models/active"""

    async def test_list_active_models(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/models/active", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "active_models" in data
        assert "emotion_detection" in data["active_models"]
        assert "sentiment_analysis" in data["active_models"]


class TestModelVersion:
    """GET /models/{model_name}/{version}"""

    async def test_get_existing_version(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/models/emotion_detection/1.0.0",
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "emotion_detection"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"

    async def test_get_nonexistent_version(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/models/emotion_detection/99.0.0",
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 404


class TestTrainingEndpoint:
    """POST /models/train"""

    async def test_trigger_training(self, client: AsyncClient, admin_user):
        resp = await client.post(
            "/models/train",
            json={
                "model_name": "emotion_detection",
                "version": "3.0.0-test",
                "dataset_size": 50,
                "auto_promote": False,
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_models"] == 1
        assert data["results"][0]["model_name"] == "emotion_detection"
        assert "accuracy" in data["results"][0]["metrics"]

    async def test_train_invalid_model(self, client: AsyncClient, admin_user):
        resp = await client.post(
            "/models/train",
            json={
                "model_name": "nonexistent_model",
                "version": "1.0.0",
                "dataset_size": 50,
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 400


class TestEvaluationEndpoint:
    """POST /models/evaluate"""

    async def test_evaluate_emotion_model(self, client: AsyncClient, admin_user):
        resp = await client.post(
            "/models/evaluate",
            json={
                "model_name": "emotion_detection",
                "model_version": "1.0.0",
                "dataset_size": 50,
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "emotion_detection"
        assert "accuracy" in data["metrics"]

    async def test_evaluate_nonexistent_model(
        self, client: AsyncClient, admin_user
    ):
        resp = await client.post(
            "/models/evaluate",
            json={
                "model_name": "emotion_detection",
                "model_version": "99.0.0",
                "dataset_size": 50,
            },
            headers=auth_header(admin_user),
        )
        assert resp.status_code == 404
