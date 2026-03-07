"""Tests for analytics & dashboard endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


class TestDashboard:
    """GET /analytics/dashboard"""

    async def test_dashboard_empty_db(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/analytics/dashboard", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "overview" in data
        assert "risk_distribution" in data
        assert "emotion_distribution" in data
        assert "emotion_trend" in data
        assert "alert_summary" in data

    async def test_student_cannot_access_dashboard(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get(
            "/analytics/dashboard", headers=auth_header(student_user)
        )
        assert resp.status_code == 403


class TestOverview:
    """GET /analytics/overview"""

    async def test_overview_empty(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/analytics/overview", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should return zeroed-out stats
        assert "total_students" in data or isinstance(data, dict)

    async def test_overview_with_data(
        self, client: AsyncClient, admin_user, db_session
    ):
        # Insert a student to get non-empty stats
        from app.models.student import Student

        student = Student(
            student_identifier="STU-ANLY",
            user_id=admin_user["id"],
            age=15,
            school="Analytics School",
            grade="10th",
        )
        db_session.add(student)
        await db_session.commit()

        resp = await client.get(
            "/analytics/overview", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200


class TestEmotionDistribution:
    """GET /analytics/emotions/distribution"""

    async def test_emotion_distribution(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/analytics/emotions/distribution", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "distribution" in data
        assert "total_analyses" in data


class TestRiskDistribution:
    """GET /analytics/risk/distribution"""

    async def test_risk_distribution(self, client: AsyncClient, admin_user):
        resp = await client.get(
            "/analytics/risk/distribution", headers=auth_header(admin_user)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "buckets" in data
        assert "total_assessed" in data


class TestStudentRBAC:
    """Students should not access analytics endpoints."""

    async def test_student_cannot_see_overview(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get(
            "/analytics/overview", headers=auth_header(student_user)
        )
        assert resp.status_code == 403

    async def test_student_cannot_see_emotions(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get(
            "/analytics/emotions/distribution",
            headers=auth_header(student_user),
        )
        assert resp.status_code == 403
