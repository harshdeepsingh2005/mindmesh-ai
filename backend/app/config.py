# MindMesh AI Backend Config
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mindmesh.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change_this_secret")
