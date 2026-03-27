"""MindMesh AI — FastAPI Dependencies for Auth & RBAC."""

from typing import List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database.db import get_db
from .models.user import User
from .services.auth import decode_access_token
from .logging_config import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency to extract and validate the current authenticated user.

    Args:
        token: JWT bearer token from request header.
        db: Database session.

    Returns:
        The authenticated User ORM object.

    Raises:
        HTTPException: If token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        logger.warning("Invalid or expired JWT token received.")
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        logger.warning("JWT token missing 'sub' claim.")
        raise credentials_exception

    # Serverless Mock Users
    if (user_id in ("admin-demo-uuid", "student-demo-uuid", "teacher-demo-uuid") or 
        user_id.startswith("admin-demo-uuid-") or 
        user_id.startswith("student-demo-uuid-") or
        user_id.startswith("teacher-demo-uuid-")):
        role = payload.get("role", "student")
        return User(id=user_id, email=f"{role}@mindmesh.ai", name=f"{role.capitalize()} Demo", role=role)

    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
    except Exception:
        user = None

    if user is None:
        logger.warning(f"User not found for id={user_id}")
        raise credentials_exception

    return user


def require_roles(allowed_roles: List[str]):
    """Dependency factory for role-based access control.

    Args:
        allowed_roles: List of role strings that are permitted.

    Returns:
        A dependency function that validates the user's role.
    """

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            logger.warning(
                f"Access denied for user={current_user.id} role={current_user.role} "
                f"(required: {allowed_roles})"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return role_checker
