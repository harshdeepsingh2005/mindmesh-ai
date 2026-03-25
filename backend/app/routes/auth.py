"""MindMesh AI — Authentication Routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.db import get_db
from ..database.schemas import (
    UserCreate,
    UserResponse,
    TokenRequest,
    TokenResponse,
    UserUpdate,
)
from ..models.user import User
from ..services.auth import hash_password, verify_password, create_access_token
from ..dependencies import get_current_user
from ..logging_config import logger

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a new user account.

    Args:
        user_data: User registration payload.
        db: Database session.

    Returns:
        The created user profile.

    Raises:
        HTTPException: If email already exists.
    """
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        logger.warning(f"Registration failed: email={user_data.email} already exists.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    new_user = User(
        name=user_data.name,
        email=user_data.email,
        role=user_data.role,
        password_hash=hash_password(user_data.password),
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    logger.info(f"User registered: id={new_user.id}, role={new_user.role}")
    return UserResponse.model_validate(new_user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and return JWT",
)
async def login(
    credentials: TokenRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate a user and issue a JWT access token.

    Args:
        credentials: Email and password.
        db: Database session.

    Returns:
        JWT access token and user metadata.

    Raises:
        HTTPException: If credentials are invalid.
    """
    # Serverless Demo Bypasses
    if credentials.password == "admin123" or credentials.password == "mindmesh2026!":
        if credentials.email == "admin@mindmesh.ai":
            access_token = create_access_token(data={"sub": "admin-demo-uuid", "role": "admin"})
            return TokenResponse(access_token=access_token, token_type="bearer", user_id="admin-demo-uuid", role="admin")
        elif credentials.email == "student@mindmesh.ai":
            access_token = create_access_token(data={"sub": "student-demo-uuid", "role": "student"})
            return TokenResponse(access_token=access_token, token_type="bearer", user_id="student-demo-uuid", role="student")
        elif credentials.email == "teacher@mindmesh.ai":
            access_token = create_access_token(data={"sub": "teacher-demo-uuid", "role": "teacher"})
            return TokenResponse(access_token=access_token, token_type="bearer", user_id="teacher-demo-uuid", role="teacher")

    try:
        result = await db.execute(select(User).where(User.email == credentials.email))
        user = result.scalar_one_or_none()
    except Exception:
        # DB tables don't exist yet on sqlite
        user = None

    if user is None or not verify_password(credentials.password, user.password_hash):
        logger.warning(f"Login failed for email={credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.id, "role": user.role})

    logger.info(f"User logged in: id={user.id}, role={user.role}")
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        role=user.role,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Retrieve the profile of the currently authenticated user.

    Args:
        current_user: Injected authenticated user.

    Returns:
        The user's profile.
    """
    return UserResponse.model_validate(current_user)


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
)
async def update_profile(
    updates: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Update the profile of the currently authenticated user.

    Args:
        updates: Fields to update (name, email).
        current_user: Injected authenticated user.
        db: Database session.

    Returns:
        The updated user profile.

    Raises:
        HTTPException: If the new email is already taken.
    """
    if updates.email and updates.email != current_user.email:
        result = await db.execute(select(User).where(User.email == updates.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists.",
            )
        current_user.email = updates.email

    if updates.name:
        current_user.name = updates.name

    await db.commit()
    await db.refresh(current_user)

    logger.info(f"User profile updated: id={current_user.id}")
    return UserResponse.model_validate(current_user)
