"""Tests for authentication & user management endpoints."""

import pytest
from httpx import AsyncClient

from tests.conftest import auth_header


# ─── Registration ────────────────────────────────────────────────


class TestRegistration:
    """POST /auth/register"""

    async def test_register_success(self, client: AsyncClient):
        resp = await client.post(
            "/auth/register",
            json={
                "name": "New User",
                "email": "new@example.com",
                "role": "student",
                "password": "securepassword123",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "new@example.com"
        assert data["role"] == "student"
        assert "id" in data

    async def test_register_duplicate_email(self, client: AsyncClient):
        payload = {
            "name": "User A",
            "email": "dup@example.com",
            "role": "student",
            "password": "securepassword123",
        }
        await client.post("/auth/register", json=payload)
        resp = await client.post("/auth/register", json=payload)
        assert resp.status_code == 409

    async def test_register_invalid_role(self, client: AsyncClient):
        resp = await client.post(
            "/auth/register",
            json={
                "name": "Bad",
                "email": "bad@example.com",
                "role": "hacker",
                "password": "securepassword123",
            },
        )
        assert resp.status_code == 422

    async def test_register_short_password(self, client: AsyncClient):
        resp = await client.post(
            "/auth/register",
            json={
                "name": "Short",
                "email": "short@example.com",
                "role": "student",
                "password": "abc",
            },
        )
        assert resp.status_code == 422


# ─── Login ───────────────────────────────────────────────────────


class TestLogin:
    """POST /auth/login"""

    async def test_login_success(self, client: AsyncClient):
        # Register first
        await client.post(
            "/auth/register",
            json={
                "name": "Login User",
                "email": "login@example.com",
                "role": "student",
                "password": "securepassword123",
            },
        )
        resp = await client.post(
            "/auth/login",
            json={"email": "login@example.com", "password": "securepassword123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "student"

    async def test_login_wrong_password(self, client: AsyncClient):
        await client.post(
            "/auth/register",
            json={
                "name": "User",
                "email": "wrong@example.com",
                "role": "student",
                "password": "securepassword123",
            },
        )
        resp = await client.post(
            "/auth/login",
            json={"email": "wrong@example.com", "password": "badpassword1"},
        )
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        resp = await client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "anything123"},
        )
        assert resp.status_code == 401


# ─── Auth Protection ─────────────────────────────────────────────


class TestAuthProtection:
    """GET /auth/me — requires valid JWT."""

    async def test_no_token(self, client: AsyncClient):
        resp = await client.get("/auth/me")
        assert resp.status_code == 401

    async def test_invalid_token(self, client: AsyncClient):
        resp = await client.get(
            "/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"}
        )
        assert resp.status_code == 401

    async def test_valid_token(self, client: AsyncClient, admin_user):
        resp = await client.get("/auth/me", headers=auth_header(admin_user))
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == admin_user["email"]
        assert data["role"] == "admin"
