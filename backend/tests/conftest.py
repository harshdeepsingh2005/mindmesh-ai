"""Shared test fixtures for the MindMesh AI test suite."""

import asyncio
from typing import AsyncGenerator, Dict

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.database.db import Base, get_db
from app.main import app
from app.services.auth import hash_password, create_access_token

# ── In-memory SQLite for tests ──────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


# ── DB session fixture ──────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create tables and yield a fresh async session, then tear down."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ── Override the FastAPI DB dependency ──────────────────────────


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an httpx AsyncClient wired to the test database."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── User factories ──────────────────────────────────────────────


async def _make_user(
    db: AsyncSession,
    *,
    name: str = "Test User",
    email: str = "test@example.com",
    role: str = "admin",
    password: str = "securepassword123",
) -> Dict:
    """Insert a user directly into the test database and return its data."""
    from app.models.user import User

    user = User(
        name=name,
        email=email,
        role=role,
        password_hash=hash_password(password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "password": password,
    }


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> Dict:
    return await _make_user(db_session, email="admin@test.com", role="admin")


@pytest_asyncio.fixture
async def teacher_user(db_session: AsyncSession) -> Dict:
    return await _make_user(
        db_session, name="Teacher", email="teacher@test.com", role="teacher"
    )


@pytest_asyncio.fixture
async def student_user(db_session: AsyncSession) -> Dict:
    return await _make_user(
        db_session, name="Student", email="student@test.com", role="student"
    )


# ── Auth helpers ────────────────────────────────────────────────


def auth_header(user: Dict) -> Dict[str, str]:
    """Generate an Authorization header for the given user dict."""
    token = create_access_token(data={"sub": user["id"], "role": user["role"]})
    return {"Authorization": f"Bearer {token}"}
