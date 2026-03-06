"""MindMesh AI Backend Entry Point."""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from sqlalchemy import text

from .config import settings
from .database.db import AsyncSessionLocal, init_db, close_db
from .database.schemas import HealthCheckResponse
from .logging_config import logger
from .routes import student, counselor, analytics
from .routes import auth, users
from .routes import analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    logger.info("Application started successfully.")
    yield
    await close_db()
    logger.info("Application shut down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered mental health monitoring and intervention platform for school ecosystems.",
    lifespan=lifespan,
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["User Management"])
app.include_router(student.router, prefix="/student", tags=["Student"])
app.include_router(analysis.router, prefix="/emotion", tags=["AI Analysis"])
app.include_router(counselor.router, prefix="/counselor", tags=["Counselor"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """Health check endpoint to verify API and database status."""
    db_status = "disconnected"
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"error: {str(e)}"

    return HealthCheckResponse(
        status="ok" if db_status == "connected" else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database=db_status,
        timestamp=datetime.utcnow(),
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
