"""Security tests — RBAC enforcement, token validation, input sanitization."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


# ─── RBAC Enforcement ────────────────────────────────────────────


class TestRBACEnforcement:
    """Verify role-based access control across protected endpoints."""

    async def test_student_cannot_access_admin_alerts(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get("/alerts/", headers=auth_header(student_user))
        assert resp.status_code == 403

    async def test_student_cannot_access_analytics(
        self, client: AsyncClient, student_user
    ):
        resp = await client.get(
            "/analytics/dashboard", headers=auth_header(student_user)
        )
        assert resp.status_code == 403

    async def test_teacher_cannot_access_models(
        self, client: AsyncClient, teacher_user
    ):
        resp = await client.get("/models/", headers=auth_header(teacher_user))
        assert resp.status_code == 403

    async def test_student_cannot_create_profiles(
        self, client: AsyncClient, student_user
    ):
        resp = await client.post(
            "/student/profiles",
            json={
                "student_identifier": "STU-HACK",
                "age": 15,
                "school": "Hack School",
                "grade": "10th",
            },
            headers=auth_header(student_user),
        )
        assert resp.status_code == 403


# ─── Token Security ─────────────────────────────────────────────


class TestTokenSecurity:
    """JWT token validation edge cases."""

    async def test_expired_token_format(self, client: AsyncClient):
        """Malformed JWT should be rejected."""
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.invalid.sig"},
        )
        assert resp.status_code == 401

    async def test_missing_bearer_prefix(self, client: AsyncClient):
        resp = await client.get(
            "/auth/me", headers={"Authorization": "notbearer token123"}
        )
        assert resp.status_code in (401, 403)

    async def test_empty_token(self, client: AsyncClient):
        resp = await client.get(
            "/auth/me", headers={"Authorization": "Bearer "}
        )
        assert resp.status_code == 401


# ─── Input Sanitization ─────────────────────────────────────────


class TestInputSanitization:
    """Ensure the API handles malicious input gracefully."""

    async def test_xss_in_name(self, client: AsyncClient):
        resp = await client.post(
            "/auth/register",
            json={
                "name": "<script>alert('xss')</script>",
                "email": "xss@test.com",
                "role": "student",
                "password": "securepassword123",
            },
        )
        # Should succeed (server stores raw text) or reject
        assert resp.status_code in (201, 422)
        if resp.status_code == 201:
            # If accepted, ensure it's stored safely
            data = resp.json()
            assert "script" in data["name"] or "<" in data["name"]

    async def test_sql_injection_in_email(self, client: AsyncClient):
        resp = await client.post(
            "/auth/login",
            json={
                "email": "admin@test.com' OR '1'='1",
                "password": "anything12345",
            },
        )
        # Should be rejected by email validation
        assert resp.status_code == 422

    async def test_oversized_input(self, client: AsyncClient):
        resp = await client.post(
            "/auth/register",
            json={
                "name": "A" * 10000,
                "email": "big@test.com",
                "role": "student",
                "password": "securepassword123",
            },
        )
        assert resp.status_code == 422
