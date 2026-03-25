"""MindMesh AI Backend Entry Point."""

import os
import shutil
import glob
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database.db import AsyncSessionLocal, init_db, close_db
from .database.schemas import HealthCheckResponse
from .logging_config import logger
from .routes import student, counselor, analytics, auth, users, analysis, alerts, models

# ─── Serverless bootstrap: copy pre-trained models to writable /tmp ───
if os.getenv("VERCEL"):
    _tmp_models = "/tmp/saved_models"
    os.makedirs(_tmp_models, exist_ok=True)
    _repo_models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    if os.path.isdir(_repo_models):
        for _src in glob.glob(os.path.join(_repo_models, "*.joblib")):
            _dst = os.path.join(_tmp_models, os.path.basename(_src))
            if not os.path.exists(_dst):
                shutil.copy2(_src, _dst)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # On Vercel: copy pre-trained models from read-only repo to writable /tmp
    if os.getenv("VERCEL"):
        tmp_models = "/tmp/saved_models"
        os.makedirs(tmp_models, exist_ok=True)
        # The repo's saved_models/ lives at /var/task/saved_models/
        repo_models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
        if os.path.isdir(repo_models):
            for src in glob.glob(os.path.join(repo_models, "*.joblib")):
                dst = os.path.join(tmp_models, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    logger.info(f"Copied pre-trained model: {os.path.basename(src)}")

    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")
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

@app.get("/system/init_db", tags=["System"])
async def trigger_init_db():
    """Manual trigger to initialize Database tables avoiding lambda boot limits."""
    await init_db()
    return {"message": "Database tables created successfully."}

# CORS Middleware — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if hasattr(settings, "CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["User Management"])
app.include_router(student.router, prefix="/student", tags=["Student"])
app.include_router(analysis.router, prefix="/emotion", tags=["AI Analysis"])
app.include_router(counselor.router, prefix="/counselor", tags=["Counselor"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics & Dashboard"])
app.include_router(models.router, prefix="/models", tags=["Model Management"])


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
