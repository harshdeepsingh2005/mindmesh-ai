"""Integration tests — counselor risk-assessment workflow."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


class TestCounselorRiskWorkflow:
    """End-to-end counselor risk assessment integration."""

    async def test_risk_assessment_nonexistent_student(
        self, client: AsyncClient, db_session
    ):
        """Assessing a nonexistent student should return a zero-risk result."""
        # Create a counselor user
        from tests.conftest import _make_user

        counselor = await _make_user(
            db_session,
            name="Counselor",
            email="counselor@test.com",
            role="counselor",
        )
        resp = await client.post(
            "/counselor/risk/assess",
            json={"student_id": "nonexistent-id", "lookback_days": 30},
            headers=auth_header(counselor),
        )
        # No data for this student, so endpoint returns a low-risk result
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "low"
        assert data["composite_score"] == 0

    async def test_risk_thresholds_endpoint(
        self, client: AsyncClient, db_session
    ):
        """GET /counselor/risk/config/thresholds should return config."""
        from tests.conftest import _make_user

        counselor = await _make_user(
            db_session,
            name="Counselor2",
            email="counselor2@test.com",
            role="counselor",
        )
        resp = await client.get(
            "/counselor/risk/config/thresholds",
            headers=auth_header(counselor),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thresholds" in data
        assert "factor_weights" in data
        # Verify threshold structure
        for level in ("low", "medium", "high"):
            assert level in data["thresholds"]
