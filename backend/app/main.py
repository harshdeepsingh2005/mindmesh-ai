# MindMesh AI Backend Entry Point
from fastapi import FastAPI
from .routes import student, counselor, analytics

app = FastAPI(title="MindMesh AI")

app.include_router(student.router, prefix="/student", tags=["Student"])
app.include_router(counselor.router, prefix="/counselor", tags=["Counselor"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
